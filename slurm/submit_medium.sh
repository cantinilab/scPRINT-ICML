#!/bin/bash

# SBATCH --cpus-per-task=24
# SBATCH --hint=nomultithread
# SBATCH --signal=SIGUSR1@180

# run script from above
eval "srun scprint fit $1"
exit 99