#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for DyCON on ISLES22 dataset
Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation
"""

import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from dataloaders.isles22 import ISLESDataset, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from utils import dycon_losses

# ========== Argument Parser ========== #
parser = argparse.ArgumentParser()

# === Basic Parameters === #
parser.add_argument('--root_dir', type=str, default='../data/ISLES22', help='Root directory containing H5 files')
parser.add_argument('--exp', type=str, default='ISLES22', help='Experiment name')
parser.add_argument('--model', type=str, default='unet_3D', help='Model architecture')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use')

# === Training Parameters === #
parser.add_argument('--max_iterations', type=int, default=20000, help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='Maximum learning rate')

parser.add_argument('--labelnum', type=int, default=10, help='Number of labeled samples')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher model')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Ramp-up duration for consistency weight')

# === DyCon-specific Parameters === #
parser.add_argument('--gamma', type=float, default=2.0,
                    help='Focusing parameter for hard positives/negatives in FeCL (γ)')
parser.add_argument('--beta_min', type=float, default=0.5, help='Minimum value for entropy weighting (β)')
parser.add_argument('--beta_max', type=float, default=5.0, help='Maximum value for entropy weighting (β)')
parser.add_argument('--s_beta', type=float, default=None,
                    help='If provided, use this static beta for UnCLoss instead of adaptive beta.')
parser.add_argument('--temp', type=float, default=0.6, help='Temperature for contrastive learning')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight for labeled losses')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight for UnCL loss')
parser.add_argument('--use_focal', type=int, default=1, help='Use focal mechanism in FeCL')
parser.add_argument('--use_teacher_loss', type=int, default=1, help='Use teacher features in FeCL')

# === Architecture Parameters === #
parser.add_argument('--patch_size', type=int, nargs='+', default=[96, 96, 64], help='Patch size for training')
parser.add_argument('--in_ch', type=int, default=1, help='Input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
parser.add_argument('--feature_scaler', type=int, default=4, help='Downscaling factor for feature maps')

# === Misc Parameters === #
parser.add_argument('--deterministic', type=int, default=1, help='Whether to set deterministic options')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')

args = parser.parse_args()

# ========== Path Setup ========== #
snapshot_path = f"../model/{args.exp}/DyCON_{args.model}_{args.consistency_type}_temp{args.temp}" \
                f"_labelnum{args.labelnum}_max_iterations{args.max_iterations}"

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== Training Configuration ========== #
batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = True
    cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

num_classes = args.num_classes
patch_size = tuple(args.patch_size)


# ========== Helper Functions ========== #
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def patients_to_slices(dataset_path, labelnum):
    """Convert patient numbers to slice numbers for ISLES dataset"""
    # For ISLES22, each H5 file contains the full 3D volume
    # So the number of labeled samples equals the labelnum directly
    return labelnum


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


# ========== Main Training ========== #
if __name__ == "__main__":
    # Create snapshot directory
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    # Logging setup
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ========== Model and EMA Model Setup ========== #
    model = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes)
    model = model.to(device)

    # Initialize EMA model
    ema_model = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes)
    ema_model = ema_model.to(device)
    for param in ema_model.parameters():
        param.detach_()

    # Log model size
    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Total params of model: {model_params:.2f}M")

    # ========== Data Loading ========== #
    # Create dataset
    db_train = ISLESDataset(h5_dir=args.root_dir,
                            split='train',
                            transform=T.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    db_val = ISLESDataset(h5_dir=args.root_dir,
                          split='val',
                          transform=T.Compose([
                              RandomCrop(patch_size),
                              ToTensor()
                          ]))

    # Log data splits
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_dir, args.labelnum)
    logging.info(f"Total slices is: {total_slices}, labeled slices is: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()
    ema_model.train()

    # ========== Optimizer and Loss Setup ========== #
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} Itertations per epoch".format(len(trainloader)))

    # Initialize DyCON loss functions
    uncl_criterion = dycon_losses.UnCLoss()
    fecl_criterion = dycon_losses.FeCLoss(device=device, temperature=args.temp, gamma=args.gamma,
                                          use_focal=bool(args.use_focal), rampup_epochs=1500)

    # ========== Training Loop ========== #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        # Adaptive beta for UnCLoss
        if args.s_beta is not None:
            beta = args.s_beta
        else:
            beta = dycon_losses.adaptive_beta(epoch=epoch_num, total_epochs=max_epoch,
                                              max_beta=args.beta_max, min_beta=args.beta_min)

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)

            # Add noise to create augmented inputs for EMA model
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise

            # Forward pass through student and teacher models
            _, stud_logits, stud_features = model(volume_batch)
            with torch.no_grad():
                _, ema_logits, ema_features = ema_model(ema_inputs)

            # Apply softmax for probability outputs
            stud_probs = F.softmax(stud_logits, dim=1)
            ema_probs = F.softmax(ema_logits, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # Calculate the supervised loss
            loss_seg = F.cross_entropy(stud_logits[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_dice = losses.dice_loss(stud_probs[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            # Prepare embeddings for contrastive learning
            B, C, _, _, _ = stud_features.shape
            stud_embedding = stud_features.view(B, C, -1)
            stud_embedding = torch.transpose(stud_embedding, 1, 2)
            stud_embedding = F.normalize(stud_embedding, dim=-1)

            ema_embedding = ema_features.view(B, C, -1)
            ema_embedding = torch.transpose(ema_embedding, 1, 2)
            ema_embedding = F.normalize(ema_embedding, dim=-1)

            # Create mask for contrastive learning with dynamic downsampling factor
            # Calculate the actual downsampling factor based on feature dimensions
            _, _, H_f, W_f, D_f = stud_features.shape
            H_in, W_in, D_in = patch_size
            downsample_factor = H_in // H_f  # This should give you the correct factor

            # Create mask with correct downsampling
            mask_con = F.avg_pool3d(label_batch.float(), kernel_size=downsample_factor,
                                    stride=downsample_factor)
            mask_con = (mask_con > 0.5).float()
            mask_con = mask_con.reshape(B, -1)
            mask_con = mask_con.unsqueeze(1)

            # Calculate DyCON losses
            teacher_feat = ema_embedding if args.use_teacher_loss else None
            f_loss = fecl_criterion(feat=stud_embedding,
                                    mask=mask_con,
                                    teacher_feat=teacher_feat,
                                    epoch=epoch_num)
            u_loss = uncl_criterion(stud_logits, ema_logits, beta)
            consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()

            # Combine all losses
            loss = args.l_weight * (loss_seg + loss_seg_dice) + \
                   consistency_weight * consistency_loss + \
                   args.u_weight * (f_loss + u_loss)

            # Check for NaN or Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf found in loss at iteration {iter_num}")
                continue

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # Update EMA model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # ========== Logging ========== #
            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_seg', loss_seg, iter_num)
            writer.add_scalar('info/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/f_loss', f_loss, iter_num)
            writer.add_scalar('info/u_loss', u_loss, iter_num)
            writer.add_scalar('info/beta', beta, iter_num)

            logging.info(f'iteration {iter_num} : loss : {loss.item():.4f}, '
                         f'loss_seg: {loss_seg.item():.4f}, loss_seg_dice: {loss_seg_dice.item():.4f}, '
                         f'f_loss: {f_loss.item():.4f}, u_loss: {u_loss.item():.4f}')

            # ========== Validation ========== #
            if iter_num % 100 == 0:
                model.eval()
                metric_list = []
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = metrics.test_single_volume(sampled_batch["image"], sampled_batch["label"],
                                                          model, classes=num_classes, patch_size=patch_size)
                    metric_list.append(metric_i)

                metric_list = np.array(metric_list)  # (val_size, num_classes, 4)
                for class_i in range(1, num_classes):
                    writer.add_scalar(f'info/val_dice_class_{class_i}',
                                      np.mean(metric_list[:, class_i - 1, 0]), iter_num)
                    writer.add_scalar(f'info/val_hd95_class_{class_i}',
                                      np.mean(metric_list[:, class_i - 1, 1]), iter_num)
                    writer.add_scalar(f'info/val_asd_class_{class_i}',
                                      np.mean(metric_list[:, class_i - 1, 2]), iter_num)

                performance = np.mean(metric_list[:, :, 0])  # mean dice
                mean_hd95 = np.mean(metric_list[:, :, 1])
                mean_asd = np.mean(metric_list[:, :, 2])

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', mean_asd, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_best = os.path.join(snapshot_path, f'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info(f'iteration {iter_num}, best model saved with dice: {best_performance:.4f}')

                logging.info(f'iteration {iter_num} : mean_dice : {performance:.4f}, '
                             f'mean_hd95 : {mean_hd95:.4f}, mean_asd : {mean_asd:.4f}')
                model.train()

            # Save checkpoint
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    logging.info(f"Training completed. Best performance: {best_performance:.4f}")