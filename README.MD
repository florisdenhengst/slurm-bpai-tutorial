This is a repository for a mini-tutorial on running jobs with SLURM.

To schedule a job, run
```bash
sbatch path/to/job.sh
```

Common commands
```bash 
sbatch path/to/job.sh   # submit job to scheduler
squeue 12345            # inspect status of job 12345
squeue -u abc123        # inspect all jobs of user abc123
scancel 12345           # remove job 12345 from the queue
scontrol show job 12345 # show estiamted job start
```