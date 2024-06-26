#!/bin/sh
#SBATCH --time=03:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2

mkdir -p $TMPDIR/dfe340
cp -r $HOME/slurm-bpai-tutorial/anaconda $TMPDIR/dfe340
cd $TMPDIR/dfe340/anaconda/
conda init
conda activate bpai
# coopdqn.py is the actual one, this is just a test
python evalcomp.py --learning_rate 2.5e-3 --seed 2 --env-id entombed_competitive_v3 --total-timesteps 1500000 --track --wandb-project-name cleanrldas --capture_video 

#python dqnslurm.py --seed $SLURM_ARRAY_TASK_ID --env-id entombed_competitive_v3 --total-timesteps 500000 --track --wandb-project-name cleanrldas --capture_video
mkdir -p $HOME/slurm-bpai-tutorial/anaconda/results
cp -r $TMPDIR/dfe340/anaconda/wandb $HOME/slurm-bpai-tutorial/anaconda/results
cp -r $TMPDIR/dfe340/anaconda/runs $HOME/slurm-bpai-tutorial/anaconda/results
rm -rf $TMPDIR/dfe340/anaconda