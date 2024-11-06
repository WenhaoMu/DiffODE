#!/usr/bin/env bash

CONFIG="$1"
TASK="$2"
seeds="123"


for seed in $seeds; do
  echo $seed
  echo $TASK
  python DiffOpt/surrogate.py --config $CONFIG --seed $seed --use_gpu --mode 'train' --task $TASK --which_gpu 3
done
