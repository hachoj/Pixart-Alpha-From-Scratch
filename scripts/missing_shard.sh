#!/bin/bash 

module load mamba
mamba activate ml

# Define as an array using parentheses
RANKS=(693 886 991 1346 1429 2025 2448 2449 2647 3013 3084 3125 3177 3365 3543 3558 3576 3801 4003 4015 4022 4023 4025 4033 4128 4175 4203 4610 4647 4694 4727 4728 4729 4731 4732 4733 4734 4735 4743 4750 4756 5250 5302 5493 5494 5495 5496 5497 5498 5499 5500 5501 5502 5503 5504 5511 5512 5513 5514 5515 5523 5524 5525 5526)

# Iterate over the array using "${RANKS[@]}"
for i in "${RANKS[@]}"
do
    echo "Running rank $i"
    python -u data/shard_large.py --rank "$i" --world_size 5573
done
