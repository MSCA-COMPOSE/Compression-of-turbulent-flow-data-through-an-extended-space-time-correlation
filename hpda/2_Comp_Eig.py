import pandas as pd
import argparse
import numpy as np
import os
import sys
import logging
import dask
from dask import delayed
from scipy.linalg import eigh, qr
from dask.distributed import Client, get_client, wait
from dask_jobqueue import SLURMCluster
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
import socket
from hpda_utils import *
from comp_utils import *
import scipy.io
import cupy as cp
from collections import defaultdict
import gc
import json
import re
from dask_cuda import LocalCUDACluster
"""
This script reads DNS snapshots (x, y, z, t), builds the fluctuation
matrix X (space × time), and computes the temporal cross-correlation matrix:
    C = X^T X
with: Ns = nx * ny (non-homogeneous directions),Nt = nz * snaps (homogeneous)
so that: X.shape = (Ns, Nh),  C.shape = (Nh, Nh)
-------------------------------------------------------------------------------
CHUNKING NOTE
-------------------------------------------------------------------------------
If X has chunks:
    X.chunks = (space_chunk, hom_chunk)
and:
    n_space_chunks = Ns / space_chunk
    n_hom_chunks  = Nh / hom_chunk
then the computational cost of C = X^T X scales approximately as:
    ~ n_space_chunks × (n_hom_chunks)^2
Each block of C has size:
    (hom_chunk × hom_chunk)
Therefore:
- Smaller hom_chunk → quadratic increase in tasks.
- Smaller space_chunk → linear increase in tasks.
- hom_chunk must be chosen so that hom_chunk^2 fits in worker memory.
Proper chunking of X is essential to control memory usage and task count.
OPTIMAL CHUNKING ON LEONARDO: Hom_chunk approx 10000
"""

### READ FILES AND COMPUTE POD
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
    )

if __name__ == '__main__':
    # Retained-energy fraction (compression level) is parsed from the CLI (see below).
    oversample = 30
    kl_C=32000 ### 32000 in production
    nx_chunk,ny_chunk,nz_chunk=[50,50,128] #[50,50,256]
    nx_chunk1 = nx_chunk // 2
    ny_chunk1 = ny_chunk // 2                                                                                               
    nz_chunk1 = nz_chunk
    L_Z = 0.3
    nblock = 2 # 2 for debug 6
    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", help="DASK scheduler file")
    parser.add_argument("input_dir", help="Directory containing input csv files")
    parser.add_argument("sequence", help="sequenceOfSteps.dat file path")
    parser.add_argument("output_dir", help="Directory where to write output parquet files")
    #parser.add_argument("structured_mesh", help="file with structured mesh data")
    parser.add_argument("target", nargs="?", type=float, default=0.95,
                        help="Compression level: retained-energy fraction in (0, 1], e.g. 0.95")
    args = parser.parse_args()
    target = float(args.target)
    
    logging.info('Starting main')
    logging.info(f"Compression energy target: {target}")
    client = Client(scheduler_file=args.scheduler)
    client.upload_file('../hpda/hpda_utils.py')
    client.upload_file('../hpda/comp_utils.py')
    print(dask.config.config)
    
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "login.cluster.address" # Provide address/domain of login node
    current_user = os.environ.get('USER')
    logging.info(f"ssh -N -L {port}:{host}:{port} {current_user}@{login_node_address}") # print ssh tunnel to use locally to access dashboard
    logging.info('Started')

    sequence_file = os.path.join(args.input_dir, args.sequence)
    seq = np.genfromtxt(sequence_file, dtype=int)
    snaps = len(seq)

#### READ DATA MATRIX and NOT PARQUET ######
    logging.info("Reading C matrix from h5")
    C = load_C_from_chunk_files(args.output_dir)
    C_chunks=C.shape[0]//8
    C = C.rechunk((C_chunks,C_chunks))

########## UN^COMMENT FOR PRODUCITON ########
#    logging.info("Reading C matrix from parquet")
#    C_df  = dd.read_parquet(
#        os.path.join(args.output_dir, "C_matrix.parquet"),
#        engine="pyarrow"
#    )
#    # Convert back to dask array
#    C = C_df.to_dask_array(lengths=True)
#    C_chunks=C.shape[0]//8
#    C = C.rechunk((C_chunks,C_chunks))
######### FOr debug random C matrix ###########
#    kl_C=5000
#    M = da.random.random((5000, 14000),chunks=(5000, 1750)).astype(np.float32)
#    C = M.T @ M
#    C_chunks = 2000
########################
    log_variable_details(C)
    C, = dask.persist(C)
    wait([C])
#    C_np = C[:1000,:1000].compute()
#    diag = da.diagonal(C).compute()
#    logging.info(f"Diag min: {diag.min()}, max: {diag.max()}")
#    logging.info(f"Symmetry check (||C - C^T||): {np.linalg.norm(C_np - C_np.T)}")

    total_energy = da.diagonal(C).sum().compute()
    max_iter = 1
    iter_count = 0
    k_min = None
    while iter_count < max_iter:
        eigvals, V = rand_eigsh_mini(C, k=kl_C)
        # eigvals, V = randomized_eigsh_on_C(C, k=kl_C,n_iter=0)
        cum_energy = np.cumsum(eigvals) / total_energy
        logging.info("Energy reached: %s", cum_energy[-1])
        idx = np.searchsorted(cum_energy, target)
        if idx < len(cum_energy):
            k_min = idx + 1
            logging.info("Energy reached: %s", cum_energy[k_min-1])
            break
        # target not reached → increase l
        kl_C += 2000
        iter_count += 1
        logging.info("Increasing l to %s (iteration %s)", kl_C, iter_count)
    if k_min is None:
        logging.warning("Target not reached after max_iter. Using all computed eigenvalues.")
        k_min = len(eigvals)
    logging.info("k_min: %s", k_min)
    logging.info("Energy reached: %s", cum_energy[k_min-1])
    V_out = V[:, :k_min].rechunk((C_chunks, -1))
    logging.info('Persisting SVD')
    V_out, =dask.persist(V_out)
    wait([V_out])
    write_V_by_rows(V_out, args.output_dir, file_row_block=C_chunks, col_chunk=5000)
    #### WRITING JSON FILES WITH INFO ############
    targets = [0.99, 0.995, 0.999]
    k_targets = {}
    cum_energy = np.cumsum(eigvals) / total_energy
    def compute_k_for_target(cum_energy, target):
        idx = np.searchsorted(cum_energy, target)
        return int(idx + 1) if idx < len(cum_energy) else None
    # Calcolo k per ciascun target
    k_targets = {str(t): compute_k_for_target(cum_energy, t) for t in targets}
    # Aggiungiamo info globali
    results = {
        "targets": targets,
        "k_per_target": k_targets,
        "k_max": int(len(eigvals)),
        "max_energy_reached": float(cum_energy[-1]),
        "kl_C_used": int(kl_C)
    }

    meta_file = os.path.join(args.output_dir, "k_per_target.json")
    with open(meta_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Saved results: %s", results)
    #column_list = [f"mode_{i}" for i in range(k_min)]
    #dd.from_dask_array(V_out, columns=column_list).to_parquet(os.path.join(args.output_dir,f'V_REDg_{snaps}.parquet'), engine='pyarrow')
    logging.info('Completed')
