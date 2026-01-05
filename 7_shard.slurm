#!/bin/bash

#SBATCH --job-name=7_shard
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --account=weishao
#SBATCH --qos=weishao
#SBATCH --partition=hpg-b200

#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
mkdir -p logs

#SBATCH --mail-user=7023555729@tmomail.net
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=REQUEUE,INVALID_DEPEND,STAGE_OUT
#SBATCH --mail-type=TIME_LIMIT,TIME_LIMIT_80,TIME_LIMIT_50

hostname;date;pwd

module purge
module load mamba
mamba activate core

python data/shard_imgnet_21_vllm.py --rank 7 --world-size 8
