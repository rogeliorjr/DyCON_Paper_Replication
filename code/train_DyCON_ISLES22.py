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
def patients_to_slices(dataset_dir, patiens_num):
    ref_dict = None
    if "ISLES" in dataset_dir:
        ref_dict = {"1": 36, "2": 38, "3": 27, "4": 53, "5": 60,
                    "6": 25, "7": 25, "8": 38, "9": 38, "10": 45,
                    "11": 27, "12": 29, "13": 32, "14": 29, "15": 44,
                    "16": 38, "17": 29, "18": 23, "19": 48, "20": 42,
                    "21": 31, "22": 48, "23": 42, "24": 23, "25": 29}
    else:
        print("Error")

    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# ========== Main Training ========== #
if __name__ == "__main__":
    # Copy current code to snapshot folder
    shutil.copy('../code/train_DyCON_ISLES22.py', snapshot_path + '/train_DyCON_ISLES22.py')
    shutil.copy('../code/run_ISLES22.sh', snapshot_path + '/run_ISLES22.sh')
    shutil.copytree('../code/dataloaders', snapshot_path + '/dataloaders',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    shutil.copytree('../code/networks', snapshot_path + '/networks', shutil.ignore_patterns(['.git', '__pycache__']))
    shutil.copytree('../code/utils', snapshot_path + '/utils', shutil.ignore_patterns(['.git', '__pycache__']))

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
            # Get the spatial dimensions after encoding
            B, C, H, W, D = stud_features.shape

            # Debug prints to understand dimensions
            if iter_num == 0:
                print(f"Input shape: {volume_batch.shape}")
                print(f"Feature shape: {stud_features.shape}")
                print(f"Expected N: {H * W * D}")

            # Reshape features to (B, N, C) where N is the number of spatial locations
            stud_embedding_fecl = stud_features.reshape(B, C, -1)  # (B, C, N)
            stud_embedding_fecl = stud_embedding_fecl.transpose(1, 2)  # (B, N, C)
            stud_embedding_fecl = F.normalize(stud_embedding_fecl, dim=-1)

            ema_embedding_fecl = ema_features.reshape(B, C, -1)  # (B, C, N)
            ema_embedding_fecl = ema_embedding_fecl.transpose(1, 2)  # (B, N, C)
            ema_embedding_fecl = F.normalize(ema_embedding_fecl, dim=-1)

            # Create mask for contrastive learning
            # Calculate the actual downsampling factor based on feature dimensions
            downsample_factor_h = volume_batch.shape[2] // stud_features.shape[2]
            downsample_factor_w = volume_batch.shape[3] // stud_features.shape[3]
            downsample_factor_d = volume_batch.shape[4] // stud_features.shape[4]

            mask_con_fecl = F.avg_pool3d(label_batch.float(),
                                         kernel_size=(downsample_factor_h, downsample_factor_w, downsample_factor_d),
                                         stride=(downsample_factor_h, downsample_factor_w, downsample_factor_d))
            mask_con_fecl = (mask_con_fecl > 0.5).float()

            # Ensure mask has the correct shape
            N = H * W * D  # Should match the N in embeddings
            mask_con_fecl = mask_con_fecl.reshape(B, -1)  # (B, N)

            # Verify the shapes match
            assert mask_con_fecl.shape[1] == stud_embedding_fecl.shape[1], \
                f"Mask shape {mask_con_fecl.shape} doesn't match embedding shape {stud_embedding_fecl.shape}"

            # Add the batch dimension for the mask
            mask_con_fecl = mask_con_fecl.unsqueeze(1)  # (B, 1, N)

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

            # Learning rate schedule
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # ========== Logging ========== #
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_consistency', consistency_loss, iter_num)
            writer.add_scalar('loss/loss_fecl', f_loss, iter_num)
            writer.add_scalar('loss/loss_uncl', u_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_seg: %f, loss_seg_dice: %f, loss_consistency: %f, loss_fecl: %f, loss_uncl: %f' %
                (iter_num, loss.item(), loss_seg.item(), loss_seg_dice.item(),
                 consistency_loss.item(), f_loss.item(), u_loss.item()))

            # ========== Validation ========== #
            if iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(db_val):
                        metric_i = []
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(0).to(device), \
                            sampled_batch['label'].unsqueeze(0).to(device)

                        # Pad if necessary
                        if volume_batch.shape[2] < args.patch_size[0] or \
                                volume_batch.shape[3] < args.patch_size[1] or \
                                volume_batch.shape[4] < args.patch_size[2]:
                            pw = max((args.patch_size[0] - volume_batch.shape[2]) // 2 + 1, 0)
                            ph = max((args.patch_size[1] - volume_batch.shape[3]) // 2 + 1, 0)
                            pd = max((args.patch_size[2] - volume_batch.shape[4]) // 2 + 1, 0)
                            volume_batch = F.pad(volume_batch, [pd, pd, ph, ph, pw, pw], mode='constant', value=0)
                            label_batch = F.pad(label_batch, [pd, pd, ph, ph, pw, pw], mode='constant', value=0)

                        outputs, _, _ = model(volume_batch)
                        outputs_soft = torch.softmax(outputs, dim=1)
                        outputs = torch.argmax(outputs_soft, dim=1).squeeze(0).cpu().numpy()
                        y = label_batch.squeeze(0).squeeze(0).cpu().numpy()

                        # Calculate metrics
                        metric_i.append(metrics.dice(outputs == 1, y == 1))
                        metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                for class_i in range(1):
                    writer.add_scalar('info/val_dice_class_{}'.format(class_i + 1), metric_list[class_i], iter_num)

                performance = np.mean(metric_list, axis=0)
                writer.add_scalar('info/val_dice_mean', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_num_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            # Save every 3000 iterations
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    logging.info("Training completed!")