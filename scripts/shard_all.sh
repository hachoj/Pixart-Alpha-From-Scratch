#!/bin/bash 

module load mamba
mamba activate ml

for i in {0..9}
do
    echo "Submitting script $i"
    sbatch scripts/${i}_shard.slurm
done
