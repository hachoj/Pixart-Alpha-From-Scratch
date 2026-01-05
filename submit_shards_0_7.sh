#!/bin/bash
set -euo pipefail

mkdir -p logs

for r in 0 1 2 3 4 5 6 7; do
  script="${r}_shard.slurm"
  if [[ ! -f "$script" ]]; then
    echo "Missing $script" >&2
    exit 1
  fi
  echo "Submitting $script"
  sbatch "$script"
done
