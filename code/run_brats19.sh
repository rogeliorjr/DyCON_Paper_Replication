# Description: Run the training code for DyCON-BraTS2019

# Main training script
python train_DyCON_BraTS19.py \
--root_dir "../data/BraTS2019" \
--exp "BraTS2019" \
--model "unet_3D" \
--max_iterations 20000 \
--temp 0.6 \
--batch_size 8 \
--labelnum 25 \
--gpu_id 0 
