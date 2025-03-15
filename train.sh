MODEL=rvt # rvt , rvt_s5, yolox
MODEL_SIZE=tiny # tiny small base
DATASET='gen4' # gen1 gen4 
GPU_IDS=0 ## [0, 1] 0 ...
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
PROJECT='RVT'
DT=5
T_BIN=1
CHANNEL=2
SEQUENCE_LENGTH=5
DATA_DIR=/home/aten-22/dataset/gen4_preprocessed/dt_${DT} # path to preprocessed dataset
GROUP=duration_${DT}

python3 train.py dataset=${DATASET} model=${MODEL} +model/${MODEL}=${MODEL_SIZE}.yaml +exp=train \
dataset.path=${DATA_DIR} dataset.ev_repr_name="'stacked_histogram_dt=${DT}_nbins=${T_BIN}'" dataset.sequence_length=${SEQUENCE_LENGTH} \
hardware.gpus=${GPU_IDS} model.backbone.input_channels=${CHANNEL} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
wandb.project_name=${PROJECT} wandb.group_name=${GROUP}