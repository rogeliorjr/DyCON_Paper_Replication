#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
import torch
from medpy import metric
from medpy.metric import hd95


def cal_dice(prediction, label, num=2):
    """Calculate dice for multi-class segmentation"""
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp) + 1e-8)
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    """Calculate comprehensive metrics using medpy"""
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    """
    Calculate dice coefficient - handles both PyTorch tensors and NumPy arrays

    Args:
        input: Prediction (tensor or numpy array)
        target: Ground truth (tensor or numpy array)
        ignore_index: Index to ignore in calculation

    Returns:
        Dice coefficient
    """
    smooth = 1.

    # Convert numpy arrays to torch tensors if needed
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input.astype(np.float32))
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.float32))

    # Ensure tensors are on the same device
    if hasattr(input, 'device') and hasattr(target, 'device'):
        if input.device != target.device:
            target = target.to(input.device)

    # Clone and flatten
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)

    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0

    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


# Metrics to track training performance
def compute_dice(output, label):
    """Batch-wise dice calculation"""
    # Handle numpy arrays
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output.astype(np.float32))
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label.astype(np.float32))

    intersection = (output * label).sum(dim=(1, 2, 3))
    dice_score = (2. * intersection) / (output.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) + 1e-8)
    return dice_score


def compute_jaccard(output, label):
    """Batch-wise Jaccard/IoU calculation"""
    # Handle numpy arrays
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output.astype(np.float32))
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label.astype(np.float32))

    intersection = (output * label).sum(dim=(1, 2, 3))
    union = output.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) - intersection
    jaccard_score = intersection / (union + 1e-8)
    return jaccard_score


def compute_hd95(pred, target, max_dist):
    """Compute Hausdorff Distance 95 percentile"""
    hd95_scores = []

    # Convert to numpy if needed
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    for p, t in zip(pred, target):
        if np.sum(p) == 0 or np.sum(t) == 0:
            hd95_scores.append(max_dist)  # Return max distance if either set is empty
        else:
            try:
                hd95_scores.append(hd95(p, t))
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                hd95_scores.append(max_dist)
    return hd95_scores