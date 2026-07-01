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
from comp_utils import *
import scipy.io
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

    nx_chunk,ny_chunk,nz_chunk=[500,200,32] #[50,50,256]
    nx_chunk1 = nx_chunk // 4
    ny_chunk1 = ny_chunk // 4                                                                                               
    nz_chunk1 = nz_chunk
    L_Z = 0.3
    nblock = 2 # 2 for debug 6
    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", help="DASK scheduler file")
    parser.add_argument("input_dir", help="Directory containing input csv files")
    parser.add_argument("sequence", help="sequenceOfSteps.dat file path")
    parser.add_argument("output_dir", help="Directory where to write output parquet files")
    parser.add_argument("snaps", help="number snapshsots")
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
    meta_file = os.path.join(args.output_dir, "C_metadata.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    nx_chunk = meta["nx_chunk"]
    ny_chunk = meta["ny_chunk"]
    nz_chunk = meta["nz_chunk"]
    row_chunks = tuple(meta["row_chunks"])
    col_chunks = tuple(meta["col_chunks"])
    dtype = np.dtype(meta["dtype"])    
    # for blockid in range(1, nblock + 1):
    blockid = 2

    # For debug #
    snaps = int(args.snaps) #560
    #############
 
    logging.info('Reading mesh')
    gridname = f"{SUBSPACE_NAME}_GRID_{blockid}.xyz"
    # structured_mesh_coord, _ = read_structured_mesh(args.structured_mesh, unroll=True)
    nxp,nyp,nzp = read_grid_header(os.path.join(args.input_dir , gridname))
    totdim = nxp*nyp*nzp
    xyz = read_grid(os.path.join(args.input_dir , gridname),nxp,nyp,nzp)
    logging.info(xyz.shape)
    vol = compute_grid_volume(xyz,L_Z)
    logging.info(vol.shape)
    ###### END MESH ########## 
    meta_file = os.path.join(args.output_dir, "EV_metadata.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    n_row_blocks = tuple(meta["Phi_row_chunks"])
    n_col_blocks = tuple(meta["Phi_col_chunks"])

    EV_c=load_V_as_dask(args.output_dir)
    EV_c = EV_c.rechunk((EV_c.shape[0]//8, EV_c.shape[1]))
    # EV_c=EV_c.rechunk((-1,EV_c.shape[1]//4)) ## RIGA SBAGLIATA!
    EV_c, =dask.persist(EV_c)
    wait([EV_c])
    log_variable_details(EV_c)
    snap_ids = [1, 10, 11, 13, 20, 25, 98, 99, 122, 340, 347, 444, 500, 511]
    snap_ids = [i for i in snap_ids if i < snaps]  # keep only snapshots present in this run
    logging.info(f"ncol, nrow: {n_col_blocks}, {n_row_blocks}")
    logging.info(f"nx_chunk, ny_chunk: {nx_chunk}, {ny_chunk}")
    phi_u = load_phi('u',args.output_dir,len(n_row_blocks),len(n_col_blocks),n_row_blocks,n_col_blocks)
    nxy_re = phi_u.shape[0]//len(n_row_blocks)
    phi_u = phi_u.rechunk((nxy_re//4,-1))
    log_variable_details(phi_u)
    u_recon = phi_u @ EV_c.T
    u_mean = load_mean_from_parquet(args.output_dir, "u", blockid)
    u_recon=u_recon+u_mean
    log_variable_details(u_recon)
    u_recon=u_recon.reshape(nxp,nyp,nzp,snaps).rechunk((nx_chunk,ny_chunk,nz_chunk,snaps))
    log_variable_details(u_recon)
    u_recon_sel = u_recon[..., snap_ids]
    u_recon_sel, = dask.persist(u_recon_sel)
    wait([u_recon_sel])
    phi_v = load_phi('v',args.output_dir,len(n_row_blocks),len(n_col_blocks),n_row_blocks,n_col_blocks)
    phi_v = phi_v.rechunk((nxy_re//4,-1))
    v_recon = phi_v @ EV_c.T
    v_mean = load_mean_from_parquet(args.output_dir, "v", blockid)
    v_recon=v_recon+v_mean
    v_recon=v_recon.reshape(nxp,nyp,nzp,snaps).rechunk((nx_chunk,ny_chunk,nz_chunk,snaps))
    v_recon_sel = v_recon[..., snap_ids]
    log_variable_details(v_recon)
    v_recon_sel, = dask.persist(v_recon_sel)
    wait([v_recon_sel])
    phi_w = load_phi('w',args.output_dir,len(n_row_blocks),len(n_col_blocks),n_row_blocks,n_col_blocks)
    phi_w = phi_w.rechunk((nxy_re//4,-1))
    w_recon = phi_w @ EV_c.T
    w_mean = load_mean_from_parquet(args.output_dir, "w", blockid)
    w_recon=w_recon+w_mean
    w_recon=w_recon.reshape(nxp,nyp,nzp,snaps).rechunk((nx_chunk,ny_chunk,nz_chunk,snaps))
    log_variable_details(w_recon)
    w_recon_sel = w_recon[..., snap_ids]
    w_recon_sel, = dask.persist(w_recon_sel)
    wait([w_recon_sel])

    files = []
    for num in seq[snap_ids]:
        filename =  f"{SUBSPACE_NAME}_{blockid}_{num}.raw"
        # filename = 'POD_1_'+format(num, '04d')+'.csv'
        # logging.info(os.path.join(args.input_dir , filename))
        files.append(os.path.join(args.input_dir , filename))

    lazy_read = [read_file(filename, nxp, nyp, nzp) for filename in files]

    logging.info('Started Read')

    Us = []
    Vs = []
    Ws = []
    for item in lazy_read:
        da_temp = da.from_delayed( item, dtype=np.float32, shape=(nxp,nyp,nzp, 5) )
        Us.append(da_temp[:, :, :, 1].reshape(nxp,nyp,nzp,1))
        Vs.append(da_temp[:, :, :, 2].reshape(nxp,nyp,nzp,1))
        Ws.append(da_temp[:, :, :, 3].reshape(nxp,nyp,nzp,1))
    u = da.concatenate(Us, axis=-1).rechunk((nx_chunk1,ny_chunk1,nzp,len(snap_ids)))
    v = da.concatenate(Vs, axis=-1).rechunk((nx_chunk1,ny_chunk1,nzp,len(snap_ids)))
    w = da.concatenate(Ws, axis=-1).rechunk((nx_chunk1,ny_chunk1,nzp,len(snap_ids)))
    del Us, Vs, Ws, da_temp

    log_variable_details(u)
    u,v,w=dask.persist(u,v,w)
    wait([u,v,w])

    #save the mean value
    logging.info('Persisting matrices')
    diff_u = u_recon_sel - u
    diff_v = v_recon_sel - v
    diff_w = w_recon_sel - w
    # ---- U ----
    num_u = da.sqrt(da.sum(diff_u**2, axis=(0,1,2)))
    den_u = da.sqrt(da.sum(u**2, axis=(0,1,2)))
    rel_err_u = num_u / den_u

    # ---- V ----
    num_v = da.sqrt(da.sum(diff_v**2, axis=(0,1,2)))
    den_v = da.sqrt(da.sum(v**2, axis=(0,1,2)))
    rel_err_v = num_v / den_v

    # ---- W ----
    num_w = da.sqrt(da.sum(diff_w**2, axis=(0,1,2)))
    den_w = da.sqrt(da.sum(w**2, axis=(0,1,2)))
    rel_err_w = num_w / den_w
    rel_err_u, rel_err_v, rel_err_w = dask.compute(
        rel_err_u, rel_err_v, rel_err_w
    )
    logging.info(f"Snapshots tested: {snap_ids}")

    # ---- U ----
    logging.info(f"[U] Relative errors per snapshot: {rel_err_u}")
    logging.info(f"[U] Min error: {np.min(rel_err_u):.6e}")
    logging.info(f"[U] Max error: {np.max(rel_err_u):.6e}")

    # ---- V ----
    logging.info(f"[V] Relative errors per snapshot: {rel_err_v}")
    logging.info(f"[V] Min error: {np.min(rel_err_v):.6e}")
    logging.info(f"[V] Max error: {np.max(rel_err_v):.6e}")

    # ---- W ----
    logging.info(f"[W] Relative errors per snapshot: {rel_err_w}")
    logging.info(f"[W] Min error: {np.min(rel_err_w):.6e}")
    logging.info(f"[W] Max error: {np.max(rel_err_w):.6e}")
    min_u = diff_u.min()
    max_u = diff_u.max()

    min_v = diff_v.min()
    max_v = diff_v.max()

    min_w = diff_w.min()
    max_w = diff_w.max()
    (min_u, max_u,
     min_v, max_v,
     min_w, max_w) = dask.compute(
        min_u, max_u,
        min_v, max_v,
        min_w, max_w
    )
    logging.info(f"[U] diff min: {min_u:.6e}")
    logging.info(f"[U] diff max: {max_u:.6e}")

    logging.info(f"[V] diff min: {min_v:.6e}")
    logging.info(f"[V] diff max: {max_v:.6e}")

    logging.info(f"[W] diff min: {min_w:.6e}")
    logging.info(f"[W] diff max: {max_w:.6e}")
    logging.info('Completed')