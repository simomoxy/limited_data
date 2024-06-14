ID=0
DEVICE=0
ACCELERATOR='gpu'
DSR=0.0
DATA_SEED=0
MODEL='cnn'
SOTA_CONFIG=False
CONFIG_PATH="/Users/simonjaxy/Documents/vub/WP1/limited_data/optuna/cnn/pareto/0.0/0/pareto0.pkl"
EPOCHS=1

DATA_SET='/Users/simonjaxy/Documents/vub/WP1/ssm_ecg/data/ptb_xl_fs100'

python ./code/train_ecg_model.py  --dataset="${DATA_SET}" --epochs=${EPOCHS} --data_seed=${DATA_SEED} --config_path="${CONFIG_PATH}" --id=${ID} --device=${DEVICE} --accelerator="${ACCELERATOR}" --down_sample_rate=${DSR} --logdir ./logs/"${MODEL}" --model "${MODEL}"