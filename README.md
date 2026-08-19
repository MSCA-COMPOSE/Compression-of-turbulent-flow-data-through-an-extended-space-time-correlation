# Compression of turbulent flow data through an extended space-time correlation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the computational framework presented in the manuscript currently under review on the GPU-accelerated compression of large turbulent flow datasets through an extended space-time correlation.

This public repository provides the GPU-enabled Dask implementation of the compression workflow together with a reconstruction-error check.

## Algorithm

<p align="center">
  <img src="tsqr_algorithm.png" alt="Compression and reconstruction algorithm" width="840">
</p>

The snapshots `u_i(x, t)` are arranged into the fluctuation matrix `X` of size `N_s x N_h`, with `N_s = 3 n_x n_y` (the three velocity components over the non-homogeneous plane) and `N_h = n_z n_t` (the spanwise-homogeneous direction folded together with the snapshots). The correlation operator `C = X^T X` of size `N_h x N_h`, coupling the spanwise direction and time, is factorised with a randomized eigensolver: a tall-skinny QR of `Y = C Ω` yields an orthonormal basis `Q`, the small matrix `B = Q^T C Q` is diagonalised with `eigh`, and the `r` leading modes needed to reach the energy target are retained. Only the reduced spatial modes `Φ_r` (`N_s x r`) and the reduced temporal eigenvectors `V_r` (`N_h x r`) are stored; any snapshot is recovered as `X_r = Φ_r V_r^T`.

## Repository structure

- `hpda/`: Python source files for the four workflow stages and the associated helper functions.
- `submission/`: example SLURM submission scripts and a driver that submits the four stages in sequence.
- `FLOW/`: pointer to the public input dataset (raw flow snapshots, grid and sequence file). Because of its size the dataset is not stored in this GitHub repository but archived on Zenodo at [10.5281/zenodo.19481070](https://doi.org/10.5281/zenodo.19481070) (see `FLOW/README.md`).

## Workflow overview

The compression workflow follows four stages, each mapped to one Python script and one SLURM submission script:

1. **Read and correlation matrix** (`1_RED_dask_read_POD.py`, `J1_RED_read_GPU.sh`).
   The raw snapshots are read and the velocity components `u, v, w` are assembled into a fluctuation matrix `X = [u; v; w]` after removing the temporal mean. The correlation operator `C = X^T X` of size `N_h x N_h`, coupling the spanwise direction and time, is factorised with a randomized eigensolver: The mean fields are saved in Parquet format.

2. **Eigendecomposition** (`2_Comp_Eig.py`, `J2_gpu_comp.sh`).
   A randomized GPU eigensolver decomposes `C`, the cumulative energy content is evaluated, and the number of modes `k` required to reach the target energy is selected. The reduced temporal eigenvectors `V` are written by rows in HDF5, and the number of modes retained for several energy targets is stored in `k_per_target.json`.

3. **Reduced spatial modes / compression** (`3_RED_write_GPU.py`, `J3_gpu_phi_comp.sh`).
   The fluctuation fields are recomputed, the reduced eigenvectors `V` are loaded, and the reduced spatial modes `Phi = X . V` are formed block-wise on the GPUs. The blocks `phi_{u,v,w}` are written to disk together with `EV_metadata.json`. The pair `(Phi, V)`, together with the mean fields, is the compressed representation of the dataset.

4. **Reconstruction check** (`4_check_RED.py`, `J4_gpu_check.sh`).
   A set of selected snapshots is reconstructed as `Phi . V^T + mean` and compared against the original snapshots. The relative reconstruction error per snapshot, and the minimum/maximum differences, are reported in the job log.

The submission scripts in `submission/` show how these four stages can be run sequentially on an HPC system.

## Public dataset

The input dataset used in this study — the velocity snapshots, grid and sequence file for the controlled-diffusion-airfoil cascade — is openly archived on Zenodo at [https://doi.org/10.5281/zenodo.19481070](https://doi.org/10.5281/zenodo.19481070). The `FLOW/` directory documents the expected input layout and how to point the workflow at the downloaded data (see `FLOW/README.md`).

## Installation on an HPC system

The workflow was developed and tested in a module-based HPC environment using Anaconda, Dask and CuPy. The GPU stages require a Python environment with CUDA-aware packages.

A representative environment setup is:

1. Load the system modules required by the cluster:

    ```bash
    module purge
    module load openmpi
    module load nvhpc
    module load cuda
    module load anaconda3/2022.05
    ```

2. Initialize Conda in the shell:

    ```bash
    source $(conda info --base)/etc/profile.d/conda.sh
    eval "$(conda shell.bash hook)"
    ```

3. Create and activate a dedicated environment:

    ```bash
    conda create --yes --prefix $HOME/.conda-envs/ghpda -c conda-forge --override-channels python=3.9
    conda activate $HOME/.conda-envs/ghpda
    ```

4. Install the required packages:

    ```bash
    conda install --yes -c conda-forge --override-channels \
        dask distributed dask-jobqueue dask-cuda \
        cupy cuda-cudart cuda-version=12 \
        numpy scipy pandas pyarrow h5py threadpoolctl
    ```

This is a cleaned version of the environment used for the GPU workflow. Cluster-specific details such as account names, partitions, module names and the conda environment prefix must be adapted by the user. The submission scripts use the placeholders `YOUR_ACCOUNT`, `YOUR_PARTITION` and `YOUR_GPU_CONDA_ENV` for exactly these values.

## Running the workflow

The scripts in `submission/` illustrate the intended execution order. Edit the paths, the number of snapshots, the compression level and the node counts at the top of `run_comp_pipeline.sh`, then submit the whole chain from the `submission/` directory:

```bash
cd submission
bash run_comp_pipeline.sh
```

The driver exports the input/output locations, the number of snapshots and the compression level (energy target), then submits the four stages with SLURM `afterok` dependencies so that each stage starts only if the previous one succeeded. The individual stages can also be submitted manually in the same order.

Input is read from the dataset directory set as `INPUTDIR`, and all outputs are written to the directory set as `OUTDIR`; both are configured at the top of `run_comp_pipeline.sh`.

## Outputs

Running the pipeline produces, in the output directory:

- `mean_{u,v,w}_<block>.parquet` — temporal mean fields,
- `C_matrix_r*_c*.h5`, `C_metadata.json` — the temporal correlation matrix, stored by chunks,
- `V_rows_*.h5` — the reduced temporal eigenvectors,
- `phi_{u,v,w}_a*_b*.npy`, `EV_metadata.json` — the reduced spatial modes (compressed output),
- `k_per_target.json` — number of modes retained for each energy target,
- reconstruction errors for the selected snapshots, reported in the stage-4 job log.

## Example: reconstructed flow fields

<p align="center">
  <img src="recon_fields.png" alt="Reference, reconstructed and error turbulent kinetic energy fields" width="820">
</p>

Turbulent kinetic energy `k/U^2` on the blade for three representative cases (`Lam999`, `Turb999`, `Turb950`, where the label combines the inlet condition with the retained-energy target). Each column compares the reference field (top), the field reconstructed from the compressed representation (middle), and the pointwise error `Δk/U^2` (bottom). The reconstruction stays close to the reference even at the more aggressive energy targets, with the residual error concentrated in the smallest turbulent scales.

## Citation

If you use this repository, please cite the archived Zenodo release:

> Lopes, G., Lengani, D., & Henningson, D. (2026). *MSCA-COMPOSE/Compression-of-turbulent-flow-data-through-an-extended-space-time-correlation: v1.0.0* (v1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX)

**DOI**: [10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX)

The scientific context, methodology, and discussion of results are described in the associated manuscript:

> Lopes, G., Henningson, D., & Lengani, D. *Compression of turbulent flow data through an extended space-time correlation.* Submitted to the *Journal of Fluid Mechanics* (under review), 2026.

This entry will be updated with the journal reference and DOI upon acceptance.

## License

This repository is distributed under the MIT License.
