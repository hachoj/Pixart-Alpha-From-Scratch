#!/bin/bash 

#SBATCH --job-name=shard_2
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=6
#SBATCH --mem=60gb 
#SBATCH --account=weishao 
#SBATCH --qos=weishao 
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

#SBATCH --output=logs/session_2.log
#SBATCH --error=logs/session_2.err
#SBATCH --time=120:00:00 

#SBATCH --mail-user=7023555729@tmomail.net 
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-type=REQUEUE,INVALID_DEPEND,STAGE_OUT 
#SBATCH --mail-type=TIME_LIMIT
 
hostname;date;pwd 
export XDG_RUNTIME_DIR=${SLURM_TMPDIR} 

module load mamba
mamba activate ml

for i in {1114..1670}
do
    echo "Running rank $i"
    python -u data/shard_large.py --rank $i --world_size 5573
done
