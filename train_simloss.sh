RUN=$1
GPU=$2

CUDA_VISIBLE_DEVICES=$GPU python run.py \
    --exp exp_train \
    --run-name $RUN \
    --sim-loss \
    --dataroot /HDD/DATASETS/   
