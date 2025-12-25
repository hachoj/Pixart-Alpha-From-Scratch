#!/bin/bash 

module load mamba
mamba activate ml

# Define as an array using parentheses
RANKS=(3346 3365 4022 4023 4025 4033 4727 4728 4729 4731 4732 4733 4734 4735 4743 4750 5493 5494 5495 5496 5497 5498 5499 5500 5501 5502 5503 5504 5511 5512 5513 5514 5515 5523 5524 5525 5526)

# Iterate over the array using "${RANKS[@]}"
for i in "${RANKS[@]}"
do
    echo "Running rank $i"
    python -u data/shard_large.py --rank "$i" --world_size 5573
done
