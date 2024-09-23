ID=0
DEVICE=0
ACCELERATOR='cpu'
DSR=0.0
DATA_SEED=0
MODEL='cnn'
SOTA_CONFIG=False
CONFIG_PATH="./ray/cnn/top1/single/ASHA/False/0.0/0/top1.pkl"
EPOCHS=1

DATA_SET='./data/ptb_xl_fs100'

python ./code/train_ecg_model.py  --dataset="${DATA_SET}" --epochs=${EPOCHS} --data_seed=${DATA_SEED} --config_path="${CONFIG_PATH}" --id=${ID} --device=${DEVICE} --accelerator="${ACCELERATOR}" --down_sample_rate=${DSR} --logdir ./logs/"${MODEL}" --model "${MODEL}"
