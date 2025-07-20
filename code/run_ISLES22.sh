#!/bin/bash
# Description: Run the training code for DyCON-ISLES22

# Main training script
python train_DyCON_ISLES22.py \
--root_dir "../data/ISLES22" \
--exp "ISLES22" \
--model "unet_3D" \
--max_iterations 20000 \
--temp 0.6 \
--batch_size 8 \
--labelnum 10 \
--gpu_id 0 \
--patch_size 96 96 64