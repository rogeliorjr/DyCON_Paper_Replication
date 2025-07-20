#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing script for DyCON on ISLES22 dataset
Evaluates the trained model on the validation set
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataloaders.isles22 import ISLESDataset, ToTensor
from networks.net_factory import net_factory_3d
from utils import metrics

# ========== Argument Parser ========== #
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='../data/ISLES22', help='Root directory containing H5 files')
parser.add_argument('--model', type=str, default='unet_3D', help='Model architecture')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use')
parser.add_argument('--labelnum', type=int, default=10, help='Number of labeled samples used in training')
parser.add_argument('--in_ch', type=int, default=1, help='Input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
parser.add_argument('--feature_scaler', type=int, default=4, help='Downscaling factor for feature maps')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--temp', type=float, default=0.6, help='Temperature for contrastive learning')
parser.add_argument('--exp', type=str, default='ISLES22', help='Experiment name')

args = parser.parse_args()

# ========== Path Setup ========== #
snapshot_path = f"../model/{args.exp}/DyCON_{args.model}_{args.consistency_type}_temp{args.temp}" \
                f"_labelnum{args.labelnum}_max_iterations20000"
test_save_path = f"../model/{args.exp}/predictions/"

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = args.num_classes


# ========== Model Setup ========== #
def create_model():
    net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes, scaler=args.feature_scaler)
    model = net.to(device)
    return model


model = create_model()

# Load the best checkpoint
best_model_path = os.path.join(snapshot_path, 'best_model.pth')
if os.path.exists(best_model_path):
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
else:
    print(f"Best model not found at {best_model_path}. Please check the path.")
    exit(1)

model.eval()

# ========== Dataset Setup ========== #
db_test = ISLESDataset(h5_dir=args.root_dir,
                       split='val',
                       transform=T.Compose([ToTensor()]))

testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

# ========== Testing Loop ========== #
print(f"Testing on {len(db_test)} samples...")
metric_list = {'dice': [], 'hd95': [], 'asd': [], 'sensitivity': [], 'specificity': []}

with torch.no_grad():
    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        volume_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)

        # Forward pass
        outputs = model(volume_batch)[1]  # Get logits
        outputs_soft = F.softmax(outputs, dim=1)

        # Convert to binary predictions
        outputs_bin = (outputs_soft[:, 1, :, :, :] > 0.5).float()

        # Move to CPU for metric calculation
        outputs_bin_np = outputs_bin.squeeze(0).cpu().numpy()
        label_np = label_batch.squeeze(0).cpu().numpy()

        if np.sum(outputs_bin_np) == 0 and np.sum(label_np) == 0:
            # Both prediction and ground truth are empty
            metric_list['dice'].append(1.0)
            metric_list['hd95'].append(0.0)
            metric_list['asd'].append(0.0)
            metric_list['sensitivity'].append(1.0)
            metric_list['specificity'].append(1.0)
        elif np.sum(outputs_bin_np) == 0 or np.sum(label_np) == 0:
            # One is empty, the other is not
            metric_list['dice'].append(0.0)
            # For HD95 and ASD, use maximum possible distance
            H, W, D = label_np.shape
            max_dist = np.linalg.norm([H, W, D])
            metric_list['hd95'].append(max_dist)
            metric_list['asd'].append(max_dist)

            if np.sum(label_np) == 0:
                # No ground truth positives
                metric_list['sensitivity'].append(0.0)
                metric_list['specificity'].append(1.0 if np.sum(outputs_bin_np) == 0 else 0.0)
            else:
                # No predicted positives
                metric_list['sensitivity'].append(0.0)
                metric_list['specificity'].append(1.0)
        else:
            # Normal case: both have positive voxels
            dice = metrics.compute_dice(outputs_bin_np, label_np).item()
            metric_list['dice'].append(dice)

            # Calculate HD95
            hd95 = metrics.compute_hd95(outputs_bin_np, label_np).item()
            metric_list['hd95'].append(hd95)

            # Calculate ASD
            asd = metrics.compute_asd(outputs_bin_np, label_np).item()
            metric_list['asd'].append(asd)

            # Calculate sensitivity and specificity
            tp = np.sum((outputs_bin_np == 1) & (label_np == 1))
            tn = np.sum((outputs_bin_np == 0) & (label_np == 0))
            fp = np.sum((outputs_bin_np == 1) & (label_np == 0))
            fn = np.sum((outputs_bin_np == 0) & (label_np == 1))

            sensitivity = tp / (tp + fn + 1e-10)
            specificity = tn / (tn + fp + 1e-10)

            metric_list['sensitivity'].append(sensitivity)
            metric_list['specificity'].append(specificity)

# ========== Calculate and Display Results ========== #
print("\n" + "=" * 60)
print("TESTING RESULTS FOR ISLES22")
print("=" * 60)
print(f"Model: {args.model}")
print(f"Labeled samples: {args.labelnum}")
print(f"Number of test samples: {len(db_test)}")
print("-" * 60)

for metric_name, values in metric_list.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric_name.upper():12s} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")

print("=" * 60)

# Save detailed results
results_file = os.path.join(test_save_path, f'test_results_labelnum{args.labelnum}.txt')
with open(results_file, 'w') as f:
    f.write("ISLES22 Test Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: {args.model}\n")
    f.write(f"Labeled samples: {args.labelnum}\n")
    f.write(f"Model path: {best_model_path}\n")
    f.write("-" * 60 + "\n")

    for metric_name, values in metric_list.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        f.write(f"{metric_name.upper():12s} | Mean: {mean_val:.4f} | Std: {std_val:.4f}\n")

    f.write("\nPer-sample results:\n")
    f.write("-" * 60 + "\n")
    for i in range(len(metric_list['dice'])):
        f.write(f"Sample {i:3d} | ")
        for metric_name in metric_list.keys():
            f.write(f"{metric_name}: {metric_list[metric_name][i]:.4f} | ")
        f.write("\n")

print(f"\nDetailed results saved to: {results_file}")