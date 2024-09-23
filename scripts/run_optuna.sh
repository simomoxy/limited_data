OBJECTIVE='multi'
OPTIMIZATION='bayesian'
NUM_GPUS=1
ACCELERATOR='gpu'
DEVICE=0
DSR=0.0
DATA_SEED=0
MODEL='cnn'
SOTA_CONFIG=False
STUDY_NAME="optuna_${MODEL}_${OBJECTIVE}_${OPTIMIZATION}_${DSR}_${SOTA_CONFIG}_${DATA_SEED}"
NUM_STUDY_SAMPLES=1
EPOCHS=1

DATA_SET='./data/ptb_xl_fs100'

python ./code/tune_optuna.py --dataset="${DATA_SET}" --data_seed=${DATA_SEED} --num_study_samples ${NUM_STUDY_SAMPLES} --study_name="${STUDY_NAME}" --objectives"=${OBJECTIVE}" --optimization="${OPTIMIZATION}" --num_gpus=${NUM_GPUS} --epochs ${EPOCHS}  --accelerator="${ACCELERATOR}" --device=${DEVICE} --down_sample_rate=${DSR} --study_path ./optuna/"${MODEL}" --logdir ./logs/optuna/"${MODEL}" --model "${MODEL}"
