# Minimal GPU helpers for the turbulent-flow compression pipeline.
# Randomized eigendecomposition, reduced spatial modes, and block loaders.

import numpy as np
import os
import logging
import dask
from dask import delayed
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from dask.distributed import get_client, wait
import dask.array as da
from hpda_utils import *
import cupy as cp
from collections import defaultdict
import gc
import re

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def get_all_gpu_memory():
    results = []
    for i in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(i):
            mempool = cp.get_default_memory_pool()
            used = mempool.used_bytes()
            free, total = cp.cuda.Device().mem_info
            results.append((i, used, free, total))
    return results


def cpu_dense_eigh(B, k, mkl_threads=32):
    # limita solo dentro questo blocco
    with threadpool_limits(limits=mkl_threads, user_api='blas'):
        evals, evecs = eigh(B, driver="evd", overwrite_a=True)
    idx = np.argsort(evals)[::-1][:k]
    return evals[idx], evecs[:, idx]


@delayed
def read_npy_file(f):
    return np.load(f)


def load_phi(field_name, dump_dir, n_row_blocks, n_col_blocks,
             row_chunk_sizes, col_chunk_sizes):
    grid = []
    for a in range(n_row_blocks):
        row_blocks = []
        for b in range(n_col_blocks):
            fname = os.path.join(
                dump_dir,
                f"phi_{field_name}_a{a}_b{b}.npy"
            )
            shape = (
                row_chunk_sizes[a],
                col_chunk_sizes[b]
            )
            block_da = da.from_delayed(
                read_npy_file(fname),
                shape=shape,
                dtype=np.float32
            )
            row_blocks.append(block_da)
        grid.append(row_blocks)
    Phi = da.block(grid)
    log_variable_details(Phi)
    return Phi


def Compute_POD_redV(X, V, dump_dir, field_name, dtype=np.float32):    
    client = get_client()
    ngpu = cp.cuda.runtime.getDeviceCount()
    logging.info(f"GPU detected {ngpu}")
    worker_addresses = list(client.scheduler_info()['workers'].keys())
    node_workers = defaultdict(list)
    for worker in worker_addresses:
        ip_match = re.match(r"tcp://([\d\.]+):\d+", worker)
        if ip_match:
            ip = ip_match.group(1)  
            node_workers[ip].append(worker) 
    worker_node_map = {w: node for node, workers in node_workers.items() for w in workers}
    active_nodes = sorted(set(worker_node_map.values()))
    cp.get_default_memory_pool().free_all_blocks()
    @delayed
    def gram_partial_block(X, V, device_id, a, b, dump_dir, field_name):
        with cp.cuda.Device(device_id):
            X_gpu = cp.asarray(X, dtype=cp.float32)
            V_gpu = cp.asarray(V, dtype=cp.float32)
            Phi = X_gpu @ V_gpu
            Phi_cpu = cp.asnumpy(Phi)
            result_gpu = get_all_gpu_memory()
            for i, (dev, used, free, total) in enumerate(result_gpu):
                logger.info(f"GPU {dev}: Used = {used / 1e6:.2f} MB | Free = {free / 1e6:.2f} MB | Total = {total / 1e6:.2f} MB" )
            del X_gpu, V_gpu, Phi
            cp.get_default_memory_pool().free_all_blocks()
            filename = os.path.join(dump_dir, f"phi_{field_name}_a{a}_b{b}.npy")
            np.save(filename, Phi_cpu)
        return filename
    n_row_blocks = len(X.chunks[0])
    n_col_blocks = len(V.chunks[1])
    partials = defaultdict(list)
    # ---- Create partial tasks ----
    gpu_slots = [(node, dev) for node in active_nodes for dev in range(ngpu)]
    total_gpus = len(gpu_slots)
    total_blocks=n_row_blocks*n_col_blocks
    task_id = 0
    batch = []
    batch_workers = []
    task_meta = []
    for a in range(n_row_blocks):
        for b in range(n_col_blocks):
            task_id += 1
            worker, device_id = gpu_slots[len(batch) % total_gpus]
            X_ra = X.blocks[a, :]
            V_rb = V.blocks[:, b]
            task = gram_partial_block(X_ra, V_rb, device_id, a, b, dump_dir, field_name)
            batch.append(task)
            batch_workers.append(worker)
            if (task_id) % total_gpus == 0 or (task_id) == total_blocks:
                logger.info(f"Computing batch {task_id - len(batch) + 1} to {task_id} of {total_blocks}")
                # Assign each task in batch to corresponding worker
                futures = [
                    client.compute(task, workers=[worker], allow_other_workers=False)
                    for task, worker in zip(batch, batch_workers)
                ]
                wait(futures)
                batch = []
                batch_workers = []
    logger.info("All GPU blocks computed and saved to disk.")
    logger.info(f"You can now load files from {dump_dir} and aggregate them.")
    return None


def rand_eigsh_mini(C, k, oversample=20, dtype=np.float32):    
    client = get_client()
    ngpu = cp.cuda.runtime.getDeviceCount()
    logging.info(f"GPU detected {ngpu}")
    worker_addresses = list(client.scheduler_info()['workers'].keys())
#    workers_info = client.scheduler_info()['workers']
#    gpu_workers = [
#        w for w in worker_addresses
#        if workers_info[w]['resources'].get('GPU', 0) > 0
#    ]
    node_workers = defaultdict(list)
    for worker in worker_addresses:
        ip_match = re.match(r"tcp://([\d\.]+):\d+", worker)
        if ip_match:
            ip = ip_match.group(1)  
            node_workers[ip].append(worker) 
      # Flatten list of workers and store node IP for each worker
    worker_node_map = {w: node for node, workers in node_workers.items() for w in workers}
    active_nodes = sorted(set(worker_node_map.values()))
    N = C.shape[0]
    l = k + oversample
    logging.info(f"RANDOMIZED MINIMAL CPU | N={N} k={k} l={l}")
    rng = np.random.default_rng(42)
    Omega_da = da.random.standard_normal(
        (N, l),
        chunks=(C.chunks[1],l)
    ).astype(np.float32)
#    Omega = rng.standard_normal((int(N), int(l))).astype(dtype)
#    log_variable_details(Omega)
#    Omega_da = da.from_array(Omega, chunks=(C.chunks[1], (l,))).persist()
    Omega_da, = dask.persist(Omega_da)
    wait(Omega_da)
    Y = C.dot(Omega_da)
    del Omega_da
    log_variable_details(Y)
    Q_blocks = []
    R_blocks = []
    gpu_slots = [(node, dev) for node in active_nodes for dev in range(ngpu)]
    total_gpus = len(gpu_slots)
    logging.info(f"Active nodes and GPU slots: {gpu_slots}")

    @delayed
    def gpu_qr_block(Y_block, device_id):
        with cp.cuda.Device(device_id):
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            Y_gpu = cp.asarray(Y_block, dtype=cp.float32)
            Q_gpu, R_gpu = cp.linalg.qr(Y_gpu, mode="reduced")
            result_gpu = get_all_gpu_memory()
            for i, (dev, used, free, total) in enumerate(result_gpu):
                logger.info(f"GPU {dev}: Used = {used / 1e6:.2f} MB | Free = {free / 1e6:.2f} MB | Total = {total / 1e6:.2f} MB" )
            Q_np = cp.asnumpy(Q_gpu)
            R_np = cp.asnumpy(R_gpu)
            del Y_gpu, Q_gpu, R_gpu
            cp.get_default_memory_pool().free_all_blocks()
        return Q_np, R_np

    row_sizes = Y.chunks[0]
    row_offsets = np.concatenate(([0], np.cumsum(row_sizes)))
    total_blocks = len(row_sizes)
    batch = []
    batch_workers = []
    result_gpu = []
    for i in range(total_blocks):
        row_start = row_offsets[i]
        row_end = row_offsets[i+1]
        worker, device = gpu_slots[i % total_gpus]
        # Slice lazy
        Y_block = Y.blocks[i, 0]
        #Y_block = Y[row_start:row_end, :]
        # Delayed QR
        task = gpu_qr_block(Y_block, device)
        batch.append(task)
        batch_workers.append(worker)
        # Launch when batch full or last
        if (i + 1) % total_gpus == 0 or (i + 1) == total_blocks:
            logging.info(f"Computing batch {i+1-len(batch)+1} to {i+1}")
            futures = [
                client.compute(t, workers=[w], allow_other_workers=False)
                for t, w in zip(batch, batch_workers)
            ]
            wait(futures)
            results = client.gather(futures)
            for Q_np_block, R_np_block in results:
                Q_blocks.append(Q_np_block)
                R_blocks.append(R_np_block)
            batch = []
            batch_workers = []
    del Y
    R_stacked = np.vstack(R_blocks)
    log_variable_details(R_stacked)
    with cp.cuda.Device(0):
        cp.get_default_memory_pool().free_all_blocks()
        R_gpu = cp.asarray(R_stacked, dtype=cp.float32)
        Qr_gpu, R_final_gpu = cp.linalg.qr(R_gpu, mode="reduced")
        # libera R_gpu subito
        del R_gpu, R_final_gpu
        cp.get_default_memory_pool().free_all_blocks()
        Qr = cp.asnumpy(Qr_gpu)
        del Qr_gpu
        cp.get_default_memory_pool().free_all_blocks()
    del R_stacked
    Q_np_blocks = []
    row_offset = 0
    for i in range(len(Q_blocks)):
        rows = R_blocks[i].shape[0]
        Qr_slice = Qr[row_offset:row_offset+rows, :]
        row_offset += rows
        Qi_final = Q_blocks[i] @ Qr_slice
        Q_np_blocks.append(Qi_final)
    Q_futures = [client.scatter(Qi, broadcast=False) for Qi in Q_np_blocks]
    Q_da_blocks = [
        da.from_delayed(
            dask.delayed(lambda x: x)(fut),
            shape=Qi.shape,
            dtype=Qi.dtype
        )
        for fut, Qi in zip(Q_futures, Q_np_blocks)
    ]
    Q = da.concatenate(Q_da_blocks, axis=0)
    Q = Q.rechunk((C.chunks[0], l//5)) 
    log_variable_details(Q)
    del Q_np_blocks
    #Q_np_test = Q.compute()
    #print(np.linalg.norm(Q_np_test.T @ Q_np_test - np.eye(Q_np_test.shape[1])))
    logging.info("Compute B = Q^T C Q")
    CQ = C.dot(Q)
    B = (Q.T).dot(CQ)
#    if l > 35000:
#        B = (B + B.T) / 2
    B = B.persist()
    wait(B)

    #@delayed
    def gpu_eigh_block(B_da, k):
        cp.get_default_memory_pool().free_all_blocks()
        # materializza B sul worker
        # B_np = B_da.compute()
        # manda su GPU
        B_gpu = cp.asarray(B_da, dtype=cp.float32)
        # eigendecomposition
        evals_gpu, evecs_gpu = cp.linalg.eigh(B_gpu)
        # porta su CPU
        evals = cp.asnumpy(evals_gpu)
        evecs = cp.asnumpy(evecs_gpu)
        # libera GPU
        del B_gpu, evals_gpu, evecs_gpu
        cp.get_default_memory_pool().free_all_blocks()
        # ordina top-k
        idx = np.argsort(evals)[::-1][:k]
        return evals[idx], evecs[:, idx]
#            from cupyx.scipy.sparse.linalg import eigsh
#            B_gpu = cp.array(B_da, dtype=cp.float64, order='C')
#            logging.info(f"B_gpu shape: {B_gpu.shape}")
#            logging.info(f"C-contiguous: {B_gpu.flags.c_contiguous}")
#            cp.cuda.Stream.null.synchronize()
#            gc.collect()
#            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
#            logging.info(f"GPU memory BEFORE allocation: "
#                         f"{free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total")
#            evals, evecs = eigsh(B_gpu, k=k, which='LM')
#            #evecs_gpu, evals_gpu, _ = cp.linalg.svd(B_gpu, full_matrices=False)
#            #evals_gpu, evecs_gpu = cp.linalg.eigh(B_gpu)
#            evals = cp.asnumpy(evals_gpu)
#            evecs = cp.asnumpy(evecs_gpu)
#            del B_gpu, evals_gpu, evecs_gpu
#            cp.get_default_memory_pool().free_all_blocks()
#            return evals[:k], evecs[:, :k]
    if B.shape[0] < 32100:
        #worker = gpu_slots[0][0]  # ad esempio primo worker GPU
        worker = worker_addresses[1]
        #target_worker = gpu_workers[0]
        #task = gpu_eigh_block(B, k)
        ##future = client.compute(task, workers=[target_worker], allow_other_workers=False)
        #future = client.compute(task, workers=[worker], allow_other_workers=False) # ad esempio primo worker GPU
        future = client.submit(gpu_eigh_block,B,k,workers=[worker],allow_other_workers=False)
        eigvals, evecs_k = client.gather(future)
        cp.get_default_memory_pool().free_all_blocks()
    else:
        logging.info("Switching to CPU dense eigh (MKL)")
        # Porta la matrice in RAM
        logging.info(f"B_cpu shape: {B.shape}")
        #logging.info(f"C-contiguous: {B_da.flags['C_CONTIGUOUS']}")
        gc.collect()
        # Decomposizione completa
        eigvals, evecs_k = cpu_dense_eigh(B,k)
        # Ordina in ordine decrescente e prendi top-k
        idx = np.argsort(eigvals)[::-1][:k]
        eigvals = eigvals[idx]
        evecs_k = evecs_k[:, idx]
    log_variable_details(evecs_k)
    # E_future = client.scatter(evecs_k, broadcast=True)
    E_future = client.scatter(evecs_k, workers=[worker]) # ad esempio primo worker GPU
    E_da = da.from_delayed(
        dask.delayed(lambda x: x)(E_future),
        shape=evecs_k.shape,
        dtype=evecs_k.dtype
    )
    E_da = E_da.rechunk((l//5,k//5))
    #E_da = da.from_array(evecs_k, chunks=(l//5,k//5))
    V = Q.dot(E_da)
    log_variable_details(V)
    V, = dask.persist(V)
    wait(V)
    logging.info("DONE")
    return eigvals, V