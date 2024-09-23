OBJECTIVE='single'
OPTIMIZATION='ASHA'
NUM_WORKER=1
GPUS_PER_WORKER=0
CPUS_PER_WORKER=8
ACCELERATOR='auto'
DEVICE=0
DSR=0.0
DATA_SEED=0
MODEL='cnn'
SOTA_CONFIG=False
STUDY_NAME="ray_${MODEL}_${OBJECTIVE}_${OPTIMIZATION}_${DSR}_${SOTA_CONFIG}_${DATA_SEED}"
NUM_STUDY_SAMPLES=1
EPOCHS=1

DATA_SET='./data/ptb_xl_fs100'

python ./code/tune_ray.py --dataset="${DATA_SET}" --num_study_samples "${NUM_STUDY_SAMPLES}" --data_seed=${DATA_SEED} --study_name="${STUDY_NAME}" --objectives="${OBJECTIVE}" --optimization="${OPTIMIZATION}" --num_worker=${NUM_WORKER} --gpus_per_worker=${GPUS_PER_WORKER} --cpus_per_worker=${CPUS_PER_WORKER} --epochs "${EPOCHS}" --device ${DEVICE} --accelerator="${ACCELERATOR}" --down_sample_rate=${DSR} --study_path ./ray/"${MODEL}" --logdir ./logs/ray/"${MODEL}"  --model "${MODEL}"
