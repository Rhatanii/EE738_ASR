#!/bin/bash

BS=64
GPU=3
LR=5e-5
# 1e-3 50 -> 5e-4 70 -> 3e-4 77 -> 1.5e-4 82 -> 5e-5 
EPOCH=90
ACCUMULATE_ITER=2
MAX_NORM=5
SAVE_PATH=eval_torch_conformer_bs_${BS}_accum_${ACCUMULATE_ITER}_lr_${LR}_no_augmentation_max_norm_${MAX_NORM}
START_EPOCH=83
INIT_MODEL=eval_torch_conformer_bs_${BS}_accum_${ACCUMULATE_ITER}_lr_5e-5_no_augmentation_max_norm_5/model00086.pt #${SAVE_PATH}/model00029.pt
# Xavier init and batch size up
LOG_DIR=runs/${SAVE_PATH}

python train_20247013.py \
    --batch_size ${BS} \
    --lr ${LR} \
    --max_epoch ${EPOCH} \
    --accumulate_iters ${ACCUMULATE_ITER} \
    --max_norm ${MAX_NORM} \
    --gpu ${GPU} \
    --save_path ${SAVE_PATH} \
    --log_dir ${LOG_DIR} \
    --start_epoch ${START_EPOCH} \
    --initial_model ${INIT_MODEL} \
    --eval

