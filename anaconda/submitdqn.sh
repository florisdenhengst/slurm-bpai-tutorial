#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2

mkdir -p $TMPDIR/dfe340
cp -r $HOME/slurm-bpai-tutorial/anaconda $TMPDIR/dfe340
cd $TMPDIR/dfe340/anaconda/
conda init
conda activate bpai
python dqnslurm.py  --env-id CartPole-v1 --total-timesteps 500000 

#python dqnslurm.py --seed $SLURM_ARRAY_TASK_ID --env-id CartPole-v1 --total-timesteps 500000 --track --wandb-project-name cleanrldas --capture_video
mkdir -p $HOME/slurm-bpai-tutorial/results
cp -r $TMPDIR/dfe340/python/wandb $HOME/slurm-bpai-tutorial/results
rm -rf $TMPDIR/dfe340/python