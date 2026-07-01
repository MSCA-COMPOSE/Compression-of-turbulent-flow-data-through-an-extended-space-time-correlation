#!/bin/bash
# Stage 1 - Read snapshots, build the fluctuation matrix X = [u; v; w]
#           and compute the temporal correlation matrix C = X^T X.
# The number of nodes is passed by run_comp_pipeline.sh (production used ~21).
##SBATCH --nodes=2                 # set here for standalone runs, or via run_comp_pipeline.sh
#SBATCH --ntasks-per-node=4        # DASK total tasks = workers + 2 (client and scheduler).
                                   # Cyclic allocation keeps client/scheduler off the same node.
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION   # e.g. boost_usr_prod on Leonardo (CINECA)
##SBATCH --qos=YOUR_QOS              # optional quality of service, e.g. boost_qos_dbg
#SBATCH --output=DASKmpi-%x.%j.out
#SBATCH --error=DASKmpi-%x.%j.out
#SBATCH --mail-user=EMAIL
#SBATCH --mail-type=FAIL
#SBATCH --propagate=STACK
#SBATCH --mem-per-cpu=17gb

# nthreads and memory-limit for the dask workers should be adjusted to the node size.
module load openmpi
module load nvhpc
module load cuda
module load anaconda3/2022.05
source $(conda info --base)/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
source activate YOUR_GPU_CONDA_ENV

mkdir -p $OUTDIR

# Launch a Dask scheduler as an extra process; the scheduler file is used for rendez-vous.
dask-scheduler --scheduler-file ./$SLURM_JOB_ID-scheduler.json &
sleep 5s

# CPU workers handle the general Dask tasks.
srun dask-worker --interface ib0 --nthreads 8 --memory-limit "119 GiB" \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-scheduler --worker-class distributed.Worker &
sleep 5s

# GPU workers handle the CuPy contractions.
srun dask-cuda-worker --interface ib0 --nthreads 1 --memory-limit 0 \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-dashboard --resources "GPU=1" &
sleep 5s

python ../hpda/1_RED_dask_read_POD.py ./$SLURM_JOB_ID-scheduler.json $INPUTDIR $SEQFILE $OUTDIR $SNAPS $TARGET
