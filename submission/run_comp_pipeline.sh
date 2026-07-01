#!/bin/bash
# Compression pipeline driver: submits the four stages in order, each starting
# only if the previous one finished (SLURM afterok dependency).
#   1) J1_RED_read_GPU.sh   -> read snapshots + correlation matrix C
#   2) J2_gpu_comp.sh       -> eigendecomposition of C + reduced eigenvectors V
#   3) J3_gpu_phi_comp.sh   -> reduced spatial modes Phi (compressed output)
#   4) J4_gpu_check.sh      -> reconstruction error check

# ---- Edit these paths to your locations on the cluster ----------------------
export INPUTDIR=/path/to/FLOW        # raw snapshots, grid and sequence file
export SEQFILE=hip_seq_turb.txt     # sequence file, located inside INPUTDIR
export SNAPS=30                     # number of snapshots to process
export TARGET=0.95                  # compression level: retained-energy fraction (0-1)
export OUTDIR=/path/to/STATS         # all pipeline outputs are written here
# -----------------------------------------------------------------------------

mkdir -p $OUTDIR

# Node counts for the reduced test case (production used 21 / 2 / 21 / 21).
jid1=$(sbatch --nodes=2 J1_RED_read_GPU.sh | awk '{print $4}')
jid2=$(sbatch --nodes=2 --dependency=afterok:$jid1 J2_gpu_comp.sh | awk '{print $4}')
jid3=$(sbatch --nodes=2 --dependency=afterok:$jid2 J3_gpu_phi_comp.sh | awk '{print $4}')
jid4=$(sbatch --nodes=2 --dependency=afterok:$jid3 J4_gpu_check.sh | awk '{print $4}')

echo "Pipeline: $jid1 -> $jid2 -> $jid3 -> $jid4"