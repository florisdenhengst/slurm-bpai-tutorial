#!/bin/sh
#SBATCH –t 0:20:00
#SBATCH –N 1
–c 24

module load python/
cp–r $HOME/run3 $TMPDIR
cd $TMPDIR/run3
python myscript.pyinput.dat
mkdir–p $HOME/run3/results
cpresult.dat run3.log $HOME/run3/results