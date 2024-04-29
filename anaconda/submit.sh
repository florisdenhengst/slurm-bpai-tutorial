#!/bin/bash
#SBATCH --job-name=pythonscript
#SBATCH --time=1
#SBATCH -N 1

module load gnu9/9.4.0

conda init
conda activate anaconda-test
cp -r $HOME/slurm-bpai-tutorial/anaconda $TMPDIR/fht800/
cd $TMPDIR/fht800/anaconda
python script.py
which python
