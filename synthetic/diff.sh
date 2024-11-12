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
