GPU=$1
EMBED_TYPE=$2
RUN=$3

CUDA_VISIBLE_DEVICES=$GPU python run.py \
    --exp exp_train \
    --embed_type $EMBED_TYPE \
    --run $RUN   
