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
from networks.net_factory import net_factory_3d
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
parser.add_argument('--temp', type=float, default=0.6,
                    help='Temperature for contrastive softmax scaling (optimal: 0.6)')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight for supervised losses')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight for unsupervised losses (UnCL + FeCL)')
parser.add_argument('--use_focal', type=int, default=1, help='Whether to use focal weighting (1 for True, 0 for False)')
parser.add_argument('--use_teacher_loss', type=int, default=1,
                    help='Use teacher-based auxiliary loss (1 for True, 0 for False)')

# === Data Augmentation === #
parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 96, 64], help='Patch size for training')

# === Network Parameters === #
parser.add_argument('--in_ch', type=int, default=1, help='Input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
parser.add_argument('--feature_scaler', type=int, default=4, help='Downscaling factor for feature maps')

# === Misc Parameters === #
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

args = parser.parse_args()

# ========== Deterministic Settings ========== #
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# ========== Path Setup ========== #
snapshot_path = f"../model/{args.exp}/DyCON_{args.model}_{args.consistency_type}_temp{args.temp}" \
                f"_labelnum{args.labelnum}_max_iterations{args.max_iterations}"
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = args.batch_size
labeled_bs = args.labeled_bs
max_iterations = args.max_iterations
base_lr = args.base_lr

# ========== Create Model ========== #
model = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=args.num_classes,
                       scaler=args.feature_scaler)
model = model.to(device)

ema_model = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=args.num_classes,
                           scaler=args.feature_scaler)
ema_model = ema_model.to(device)
ema_model.eval()

# Copy weights from model to ema_model
for param, ema_param in zip(model.parameters(), ema_model.parameters()):
    ema_param.data.copy_(param.data)

print(f"Total params of model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


# ========== Utility Functions ========== #
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def plot_samples(image, mask, epoch):
    """Plot sample slices of the image/preds and mask"""
    # image: (C, H, W, D), mask: (H, W, D)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image[1][:, :, image.shape[-1] // 2], cmap='gray')  # access the class at index 1
    ax[1].imshow(mask[:, :, mask.shape[-1] // 2], cmap='viridis')
    plt.savefig(f'../misc/train_preds/ISLES22_sample_slice_{str(epoch)}.png')
    plt.close()


def patients_to_slices(dataset_path, labelnum):
    """Convert patient numbers to slice numbers for ISLES dataset"""
    # For ISLES22, each H5 file contains the full 3D volume
    # So the number of labeled samples equals the labelnum directly
    return labelnum


# ========== Main Training Function ========== #
if __name__ == "__main__":
    # Create directories
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    # ========== Logging Setup ========== #
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ========== Dataset Setup ========== #
    train_transform = T.Compose([
        RandomCrop(args.patch_size),
        RandomRotFlip(),
        ToTensor()
    ])

    db_train = ISLESDataset(h5_dir=args.root_dir,
                            split='train',
                            transform=train_transform)

    db_val = ISLESDataset(h5_dir=args.root_dir,
                          split='val',
                          transform=T.Compose([ToTensor()]))

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_dir, args.labelnum)
    print(f"Total slices is: {total_slices}, labeled slices is: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    # ========== Loss Setup ========== #
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)

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
            loss_seg = ce_loss(stud_logits[:labeled_bs], label_batch[:labeled_bs].long())
            loss_seg_dice = dice_loss(stud_probs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))

            # Compute FeCL loss (batch-wise contrastive learning)
            # Prepare embeddings properly
            B, C, _, _, _ = stud_features.shape
            stud_embedding_fecl = stud_features.view(B, C, -1)
            stud_embedding_fecl = torch.transpose(stud_embedding_fecl, 1, 2)
            stud_embedding_fecl = F.normalize(stud_embedding_fecl, dim=-1)

            ema_embedding_fecl = ema_features.view(B, C, -1)
            ema_embedding_fecl = torch.transpose(ema_embedding_fecl, 1, 2)
            ema_embedding_fecl = F.normalize(ema_embedding_fecl, dim=-1)

            # Create mask for contrastive learning
            mask_con_fecl = F.avg_pool3d(label_batch.float(), kernel_size=args.feature_scaler * 4,
                                         stride=args.feature_scaler * 4)
            mask_con_fecl = (mask_con_fecl > 0.5).float()
            mask_con_fecl = mask_con_fecl.reshape(B, -1)
            mask_con_fecl = mask_con_fecl.unsqueeze(1)

            teacher_feat = ema_embedding_fecl if args.use_teacher_loss else None
            f_loss = fecl_criterion(feat=stud_embedding_fecl,
                                    mask=mask_con_fecl,
                                    teacher_feat=teacher_feat,
                                    gambling_uncertainty=None,
                                    epoch=epoch_num)

            # Compute UnCL loss (uncertainty-aware consistency)
            u_loss = uncl_criterion(stud_logits, ema_logits, beta)
            consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()

            # Gather losses
            loss = args.l_weight * (
                    loss_seg + loss_seg_dice) + consistency_weight * consistency_loss + args.u_weight * (
                           f_loss + u_loss)

            # Check for NaN or Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf found in loss at iteration {iter_num}")
                continue

            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            # Logging
            writer.add_scalar('info/loss', loss, iter_num)
            writer.add_scalar('info/f_loss', f_loss, iter_num)
            writer.add_scalar('info/u_loss', u_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_seg, iter_num)
            writer.add_scalar('info/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            # Clean up
            del noise, stud_embedding_fecl, ema_embedding_fecl, ema_logits, ema_features, ema_probs, mask_con_fecl

            # Calculate training metrics
            with torch.no_grad():
                outputs_bin = (stud_probs[:, 1, :, :, :] > 0.5).float()
                dice_score = metrics.compute_dice(outputs_bin, label_batch)
                H, W, D = stud_logits.shape[-3:]
                max_dist = np.linalg.norm([H, W, D])
                hausdorff_score = metrics.compute_hd95(outputs_bin, label_batch, max_dist)

            writer.add_scalar('train/Dice', dice_score.mean().item(), iter_num)
            writer.add_scalar('train/HD95', hausdorff_score.mean().item(), iter_num)

            # Periodic logging
            if iter_num % 50 == 0:
                logging.info('iteration %d : loss : %f loss_seg: %f loss_seg_dice: %f f_loss: %f u_loss: %f'
                             % (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(),
                                f_loss.item(), u_loss.item()))

            # Periodic validation and checkpointing
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = []

                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(db_val):
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(0).to(device), \
                            sampled_batch['label'].unsqueeze(0).to(device)

                        # Get model predictions
                        _, outputs = model(volume_batch)
                        outputs_soft = F.softmax(outputs, dim=1)
                        outputs_bin = (outputs_soft[:, 1, :, :, :] > 0.5).float()

                        # Calculate metrics
                        dice_score = metrics.compute_dice(outputs_bin, label_batch).item()

                        # For HD95, handle edge cases
                        if outputs_bin.sum() > 0 and label_batch.sum() > 0:
                            hd95_score = metrics.compute_hd95(outputs_bin, label_batch).item()
                        else:
                            hd95_score = 0.0

                        avg_metric.append([dice_score, hd95_score])

                avg_metric = np.array(avg_metric)
                avg_dice = np.mean(avg_metric[:, 0])
                avg_hd95 = np.mean(avg_metric[:, 1])

                writer.add_scalar('val/Dice', avg_dice, iter_num)
                writer.add_scalar('val/HD95', avg_hd95, iter_num)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, avg_dice, avg_hd95))

                if avg_dice > best_performance:
                    best_performance = avg_dice
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                model.train()

            # Early stopping check
            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()