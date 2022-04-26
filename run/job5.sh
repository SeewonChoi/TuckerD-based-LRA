#!/bin/bash -l
#SBATCH -o job%j.out
#SBATCH -t 72:00:00   
#SBATCH --partition=C072M0512G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
source activate torch
cd ..

python main.py --sketchType Dlearning2 --dataType brain --bestdone 0 --N 100 --N_train 20 --side 2

