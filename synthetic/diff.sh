#!/bin/bash
#SBATCH --partition=alrodri-a100
#SBATCH --time=00-09:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB
#SBATCH --account=alrodri
#SBATCH --output=/home/muwenhao/workplace/DiffOpt/synthetic/log/%j.out
#SBATCH --error=/home/muwenhao/workplace/DiffOpt/synthetic/log/%j.err

nvidia-smi
CONFIG="$1"
TASK="$2"
seeds="1469983670"
Coefficients="0"


for seed in $seeds; do
  for Coefficient in $Coefficients; do
    echo $seed
    echo $TASK
    echo $Coefficient
    # python DiffOpt/DiffOpt.py --config $CONFIG --seed $seed --mode 'train' --task $TASK --coefficient $Coefficient --which_gpu 6
    python DiffOpt/DiffOpt.py --config $CONFIG --seed $seed --use_gpu --mode 'eval' --task $TASK --coefficient $Coefficient --which_gpu 0
  done
done
