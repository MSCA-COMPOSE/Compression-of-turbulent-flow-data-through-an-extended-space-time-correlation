import pandas as pd
import argparse
import numpy as np
import os
import sys
import logging
import dask
from dask.distributed import Client, get_client, wait
from dask_jobqueue import SLURMCluster
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
import socket
from hpda_utils import *
import scipy.io
import json
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
    kl_C=5000
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
    parser.add_argument("snaps", help="snapshot number")
    #parser.add_argument("structured_mesh", help="file with structured mesh data")
    parser.add_argument("target", nargs="?", type=float, default=0.95,
                        help="Compression level: retained-energy fraction in (0, 1], e.g. 0.95")
    args = parser.parse_args()
    target = float(args.target)
    
    logging.info('Starting main')
    logging.info(f"Compression energy target: {target}")
    client = Client(scheduler_file=args.scheduler)
    client.upload_file('../hpda/hpda_utils.py')
    print(dask.config.config)
    
    host = client.run_on_scheduler(socket.gethostname)
    port = client.scheduler_info()['services']['dashboard']
    login_node_address = "login.cluster.address" # Provide address/domain of login node
    current_user = os.environ.get('USER')

    logging.info(f"ssh -N -L {port}:{host}:{port} {current_user}@{login_node_address}") # print ssh tunnel to use locally to access dashboard

    logging.info('Started')

    # sequence_file = os.path.join(args.sequence)
    sequence_file = os.path.join(args.input_dir, args.sequence)
    seq = np.genfromtxt(sequence_file, dtype=int)
    snaps = len(seq)
    logging.info(f"Number of elements: {snaps}")
    SUBSPACE_NAME = "FLOW_phys"
    chunk_size=100000
    UTOT = []
    VTOT = []
    WTOT = []

    #### start loop on blocks
    
    # for blockid in range(1, nblock + 1):
    blockid = 2
    files = []

    # For debug #
    snaps = int(args.snaps) #48 #560
    #############

    for num in seq[:snaps]:
        filename =  f"{SUBSPACE_NAME}_{blockid}_{num}.raw"
        # filename = 'POD_1_'+format(num, '04d')+'.csv'
        # logging.info(os.path.join(args.input_dir , filename))
        files.append(os.path.join(args.input_dir , filename))
    
    logging.info('Reading mesh')
    gridname = f"{SUBSPACE_NAME}_GRID_{blockid}.xyz"
    # structured_mesh_coord, _ = read_structured_mesh(args.structured_mesh, unroll=True)
    nxp,nyp,nzp = read_grid_header(os.path.join(args.input_dir , gridname))
    totdim = nxp*nyp*nzp
    xyz = read_grid(os.path.join(args.input_dir , gridname),nxp,nyp,nzp)
    logging.info(xyz.shape)
    vol = compute_grid_volume(xyz,L_Z)
    logging.info(vol.shape)

    lazy_read = [read_file(filename, nxp, nyp, nzp) for filename in files]

    logging.info('Started Read')

    Us = []
    Vs = []
    Ws = []
    for item in lazy_read:
        da_temp = da.from_delayed( item, dtype=np.float32, shape=(nxp,nyp,nzp, 5) )
        Us.append(da_temp[:, :, :, 1].reshape(nxp*nyp,nzp))
        Vs.append(da_temp[:, :, :, 2].reshape(nxp*nyp,nzp))
        Ws.append(da_temp[:, :, :, 3].reshape(nxp*nyp,nzp))
    u = da.concatenate(Us, axis=-1).rechunk((nx_chunk*ny_chunk,nzp*snaps))
    v = da.concatenate(Vs, axis=-1).rechunk((nx_chunk*ny_chunk,nzp*snaps))
    w = da.concatenate(Ws, axis=-1).rechunk((nx_chunk*ny_chunk,nzp*snaps))
    del Us, Vs, Ws, da_temp

    # Remove row mean from u,v,w
    u_mean = da.mean(u, axis=1, keepdims=True)
    v_mean = da.mean(v, axis=1, keepdims=True)
    w_mean = da.mean(w, axis=1, keepdims=True)
    u = (u - u_mean).astype(np.float32)
    v = (v - v_mean).astype(np.float32)
    w = (w - w_mean).astype(np.float32)
    log_variable_details(u)
    u,v,w,u_mean,v_mean,w_mean=dask.persist(u,v,w,u_mean,v_mean,w_mean)
    wait([u,v,w,u_mean,v_mean,w_mean])

    #save the mean value
    logging.info('Persisting matrices')

    #### UNCOMMENT TO REWRITE ON PARQUET FORMAT
#        u, v, w, u_mean, v_mean, w_mean = dask.persist(u,v,w,u_mean,v_mean,w_mean)        
#        logging.info('Matrices persisted')
#        column_list = seq[:snaps].astype(str).tolist()
#        dd.from_dask_array(u, columns=column_list).to_parquet(os.path.join(args.output_dir,'u_{}.parquet'.format(blockid)), engine='pyarrow')
#        dd.from_dask_array(v, columns=column_list).to_parquet(os.path.join(args.output_dir,'v_{}.parquet'.format(blockid)), engine='pyarrow')
#        dd.from_dask_array(w, columns=column_list).to_parquet(os.path.join(args.output_dir,'w_{}.parquet'.format(blockid)), engine='pyarrow')
#    columns = [f"col_{i}" for i in range(C.shape[1])]
    dd.from_dask_array(u_mean, columns=['1']).to_parquet(os.path.join(args.output_dir,'mean_u_{}.parquet'.format(blockid)), engine='pyarrow')
    dd.from_dask_array(v_mean, columns=['1']).to_parquet(os.path.join(args.output_dir,'mean_v_{}.parquet'.format(blockid)), engine='pyarrow')
    dd.from_dask_array(w_mean, columns=['1']).to_parquet(os.path.join(args.output_dir,'mean_w_{}.parquet'.format(blockid)), engine='pyarrow')
    ###########################################
    # u,v,w = dask.persist(u,v,w)
    u=u.rechunk((nx_chunk*ny_chunk*8,nz_chunk*snaps//7))
    v=v.rechunk((nx_chunk*ny_chunk*8,nz_chunk*snaps//7))
    w=w.rechunk((nx_chunk*ny_chunk*8,nz_chunk*snaps//7))    
    X=da.concatenate([u, v, w])
    log_variable_details(X)
    C = da.dot(X.T,X)
    log_variable_details(C)
    C, = dask.persist(C)
    wait([C])
    log_variable_details(C)
    C_chunks=nz_chunk*snaps
    C = C.rechunk((C_chunks,C_chunks))
    n_files=C.shape[0]//C_chunks
    C, = dask.persist(C)
    wait([C])
    metadata = {
        "row_chunks": list(C.chunks[0]),
        "col_chunks": list(C.chunks[1]),
        "dtype": "float32",
        "shape": C.shape,
        "nx_chunk": nx_chunk,
        "ny_chunk": ny_chunk,
        "nz_chunk": nz_chunk
    }
    meta_file = os.path.join(args.output_dir, "C_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f)
    write_C_by_chunks_parallel(C, args.output_dir, client)
    logging.info('Completed')