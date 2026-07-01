#!/bin/bash
# Stage 3 - Load the reduced eigenvectors V and form the reduced spatial modes
#           Phi = X . V on the GPUs. Phi, together with V and the mean fields,
#           is the compressed representation of the dataset.
# The number of nodes is passed by run_comp_pipeline.sh (production used ~21).
##SBATCH --nodes=2                 # set here for standalone runs, or via run_comp_pipeline.sh
#SBATCH --ntasks-per-node=3        # DASK total tasks = workers + 2 (client and scheduler).
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION   # e.g. boost_usr_prod on Leonardo (CINECA)
##SBATCH --qos=YOUR_QOS              # optional, e.g. boost_qos_dbg
#SBATCH --output=DASKmpi-%x.%j.out
#SBATCH --error=DASKmpi-%x.%j.out
#SBATCH --mail-user=EMAIL
#SBATCH --mail-type=FAIL
#SBATCH --propagate=STACK
#SBATCH --mem-per-cpu=20gb

module load openmpi
module load nvhpc
module load cuda
module load anaconda3/2022.05
source $(conda info --base)/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
source activate YOUR_GPU_CONDA_ENV

dask-scheduler --scheduler-file ./$SLURM_JOB_ID-scheduler.json &
sleep 5s

srun dask-worker --interface ib0 --nthreads 8 --memory-limit "160 GiB" \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-scheduler --worker-class distributed.Worker &
sleep 5s

srun dask-cuda-worker --interface ib0 --nthreads 1 --memory-limit 0 \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-dashboard --resources "GPU=1" &
sleep 5s

python ../hpda/3_RED_write_GPU.py ./$SLURM_JOB_ID-scheduler.json $INPUTDIR $SEQFILE $OUTDIR $SNAPS $TARGET
