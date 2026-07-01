# Minimal helper functions for the turbulent-flow compression pipeline.
# Grid/snapshot readers, cell volumes, correlation-matrix and eigenvector I/O.

import dask
import numpy as np
import glob
import os
import h5py
import json
import logging
import dask.array as da
import dask.dataframe as dd
from dask.distributed import wait


def load_mean_from_parquet(output_dir, field_name, blockid):    
    path = os.path.join(output_dir,f"mean_{field_name}_{blockid}.parquet")
    # Leggi come Dask DataFrame
    df = dd.read_parquet(path, engine="pyarrow")
    # Converti in Dask Array
    mean_array = df.to_dask_array(lengths=True)
    # Assicurati shape (s,1) per broadcasting corretto
    if mean_array.ndim == 1:
        mean_array = mean_array[:, None]
    return mean_array


def write_block_direct(C_block, filename):
    block = C_block          # compute locale al worker
    n_rows, n_cols = block.shape
    with h5py.File(filename, "w") as f:
        dset = f.create_dataset(
            "C",
            shape=(n_rows, n_cols),
            dtype=np.float32,
            chunks=(n_rows, n_cols),   # già chunk-aligned
            compression=None
        )
        dset[:, :] = block
    del block
    return filename


def write_C_by_chunks_parallel(C, output_dir, client):
    row_chunks = C.chunks[0]
    col_chunks = C.chunks[1]
    tasks = []
    r_start = 0
    for i_r, r_size in enumerate(row_chunks):
        c_start = 0
        for i_c, c_size in enumerate(col_chunks):
            r_end = r_start + r_size
            c_end = c_start + c_size
            fname = os.path.join(
                output_dir,
                f"C_matrix_r{i_r}_c{i_c}.h5"
            )
            # slice perfettamente allineato ai chunk reali
            C_block = C[r_start:r_end, c_start:c_end]
            task = dask.delayed(write_block_direct)(
                C_block,
                fname
            )
            tasks.append(task)
            c_start = c_end
        r_start = r_end
    futures = client.compute(tasks)
    wait(futures)


@dask.delayed
def read_h5_C_block(fname):
    with h5py.File(fname, "r") as f:
        return f["C"][:]


def load_C_from_chunk_files(input_dir):
    # ---- Leggi metadata una sola volta ----
    meta_file = os.path.join(input_dir, "C_metadata.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    row_chunks = tuple(meta["row_chunks"])
    col_chunks = tuple(meta["col_chunks"])
    dtype = np.dtype(meta["dtype"])
    C_grid = []
    for r_idx, r_size in enumerate(row_chunks):
        row_blocks = []
        for c_idx, c_size in enumerate(col_chunks):
            fname = os.path.join(
                input_dir,
                f"C_matrix_r{r_idx}_c{c_idx}.h5"
            )
            block = da.from_delayed(
                read_h5_C_block(fname),
                shape=(r_size, c_size),
                dtype=dtype
            )
            row_blocks.append(block)
        C_grid.append(row_blocks)
    C = da.block(C_grid)
    return C


def write_V_by_rows(V, output_dir,file_row_block=20000,col_chunk=6000):
    n_rows, n_cols = V.shape
    file_idx = 0
    for r0 in range(0, n_rows, file_row_block):
        r1 = min(r0 + file_row_block, n_rows)
        print(f"Writing rows {r0}:{r1}")
        V_block = V[r0:r1, :]
        fname = os.path.join(output_dir, f"V_rows_{file_idx}.h5")
        with h5py.File(fname, "w") as f:
            dset = f.create_dataset(
                "V",
                shape=(r1 - r0, n_cols),
                dtype=np.float32,
                chunks=(r1 - r0, min(col_chunk, n_cols)),
                compression=None
            )
            # Qui non spezzettiamo le righe
            block = V_block.compute()
            dset[:, :] = block
            del block
        file_idx += 1
    print("V written by rows (no row chunking).")


def load_V_as_dask(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "V_rows_*.h5")))
    blocks = []
    for f in files:
        h5f = h5py.File(f, "r")
        dset = h5f["V"]
        arr = da.from_array(dset, chunks=dset.chunks)
        blocks.append(arr)
    V = da.concatenate(blocks, axis=0)
    return V


def read_grid_header(fname):
    return np.fromfile(fname,count=3,dtype='int32')


def read_grid(fname,nxp,nyp,nzp):
    return np.fromfile(fname,offset=12,dtype='float32').reshape((nxp,nyp,nzp,3),order='F')


@dask.delayed
def read_file(fname,nxp,nyp,nzp):
    data = np.fromfile(fname, offset=28, dtype='float32').reshape((nxp, nyp, nzp, 5), order='F')
    data = np.nan_to_num(data, nan=0.0)  # Convert NaNs to 0
    return data


def polygon_area(x, y):
    """Computing area of elements."""
    return 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))


def compute_grid_volume(grid,L_Z):
    nxp,nyp,nzp = grid.shape[:3]
    dz = L_Z/nzp
    x = np.zeros([nxp,nyp,nzp])
    y = np.zeros([nxp,nyp,nzp])
    volumes = np.zeros([nxp,nyp,nzp]) #approx for consistent dimensions
#    for j in range(nyp-1):
#        for i in range(nxp-1):
#            x_tmp = [grid[i,j,0,0],
#                     grid[i+1,j,0,0],
#                     grid[i,j+1,0,0],
#                     grid[i+1,j+1,0,0]]
#            y_tmp = [grid[i,j,0,1],
#                     grid[i+1,j,0,1],
#                     grid[i,j+1,0,1],
#                     grid[i+1,j+1,0,1]]
#            volumes[i,j] = dz*polygon_area(x_tmp,y_tmp)
    x[:,:,0] = grid[...,0,0]
    y[:,:,0] = grid[...,0,1]
    for j in range(nyp-1):
        for i in range(nxp-1):
            xv = np.array([x[i,j,0],x[i+1,j,0],x[i+1,j+1,0],x[i,j+1,0]])
            yv = np.array([y[i,j,0],y[i+1,j,0],y[i+1,j+1,0],y[i,j+1,0]])
            volumes[i,j,0] = dz*polygon_area(xv, yv)    
    for k in range(nzp):
        volumes[:,:,k]=volumes[:,:,0]
    return volumes


def log_variable_details(variable):
    logging.info(f"Type of variable: {type(variable)}")

    # Check if the variable is a Dask array and log its details
    if isinstance(variable, da.Array):
        logging.info(f"Shape of Dask array: {variable.shape}")
        logging.info(f"Data type (dtype) of Dask array: {variable.dtype}")
        logging.info(f"Number of partitions (chunks) in Dask array: {variable.npartitions}")
        logging.info(f"Chunk shape (chunk size) of Dask array: {variable.chunksize}")