#!/bin/bash
#SBATCH --job-name=cleanrltest
#SBATCH --time=15
#SBATCH -N 1

module load gnu9/9.4.0

conda init
conda activate anaconda-test

which python

cd /var/scratch/fht800/cleanrl/
python cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000
