#!/bin/bash
#SBATCH --job-name=pythonscript
#SBATCH --time=1
#SBATCH -N 1

module load gnu9/9.4.0

conda init
conda activate bpai
cp -r $HOME/slurm-bpai-tutorial/anaconda $TMPDIR/dfe340/
cd $TMPDIR/dfe340/anaconda
python script.py
which python