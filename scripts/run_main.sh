ID=0
DEVICE=0
DSR=0.0
DATA_SEED=0
MODEL='xresnet'
CONFIG_PATH=""
ACCELERATOR='cpu'
EPOCHS=1

DATA_SET='./data/ptb_xl_fs100'

python ./code/train_ecg_model.py  --dataset="${DATA_SET}" --epochs=${EPOCHS} --data_seed=${DATA_SEED} --config_path="${CONFIG_PATH}" --id=${ID} --accelerator="${ACCELERATOR}" --device=${DEVICE} --down_sample_rate=${DSR} --logdir ./logs/"${MODEL}" --model "${MODEL}"
