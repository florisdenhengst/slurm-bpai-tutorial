#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1

mkdir -p $TMPDIR/dfe340
cp -r $HOME/slurm-bpai-tutorial/python $TMPDIR/dfe340
cd $TMPDIR/dfe340/python
python script.py
rm -rf $TMPDIR/dfe340/python