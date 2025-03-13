# Event Model 
this repository integrate 3 models, RVT, RVT-S5, YOLOX and anable to train and evaluate.

## 1. setup
tested python3.11 and venv on ubuntu22.04

### 1-1. virtual env
```shell
python3.11 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

### 1-2. data preprocess
```shell
DATASET='' # gen1 or gen4 or VGA
DATA_DIR=''
DEST_DIR=''
NUM_PROCESSES=20  # set to the number of parallel processes to use
python preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_${DATASET}.yaml -ds ${DATASET} -np ${NUM_PROCESSES}
```

## 2. Train
```shell
MODEL='' # rvt , rvt_s5, yolox
MODEL_SIZE='' # tiny small base
DATASET='' # gen1 gen4 
DATA_DIR='' # path to preprocessed dataset
GPU_IDS=0 ## [0, 1] 0 ...
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
PROJECT='RVT'
GROUP='duration_50'
python3 train.py dataset=${DATASET} model=${MODEL} +model/${MODEL_SIZE}=${MODEL_SIZE}.yaml +exp=train \
dataset.path=${DATA_DIR} hardware.gpus=${GPU_IDS} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
wandb.project_name=${PROJECT} wandb.group_name=${GROUP}
```

## 3. Eval
```shell
MODEL='' # rvt , rvt_s5, yolox
MODEL_SIZE='' # tiny small base
DATASET='' # gen1 gen4 
DATA_DIR='' # path to preprocessed dataset
GPU_ID=0 ## [0, 1] 0 ...
BATCH_SIZE_PER_GPU=8
EVAL_WORKERS_PER_GPU=2
CKPT='' ## path to .ckpt
python3 validation.py dataset=${DATASET} model=${MODEL} +model/${MODEL_SIZE}=${MODEL_SIZE}.yaml +exp=val \
dataset.path=${DATA_DIR} hardware.gpus=${GPU_ID} \
hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
checkpoint=${CKPT}
```