import os
import sys
import shutil
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from networks.net_factory_3d import net_factory_3d
from utils import ramps, metrics, losses, test_3d_patch
from dataloaders.isles22 import ISLESDataset, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

# Argument parsing
parser = argparse.ArgumentParser(description="Training DyCON on ISLES22 Dataset")

parser.add_argument('--root_dir', type=str, default="../data/ISLES22", help='Path to ISLES-2022 dataset')
parser.add_argument('--patch_size', type=list, default=[96, 96, 64], help='Input image patch size')

parser.add_argument('--exp', type=str, default='ISLES22', help='Experiment name')
parser.add_argument('--gpu_id', type=str, default=0, help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training (0 or 1)')

parser.add_argument('--model', type=str, choices=['unet_3D', 'vnet'], default='unet_3D', help='Model architecture')
parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--feature_scaler', type=int, default=2, help='Feature scaling factor for contrastive loss')

parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum number of training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='Total batch size per GPU')
parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')
parser.add_argument('--labelnum', type=int, default=10, help='Number of labeled samples')

# DyCON specific parameters
parser.add_argument('--temp', type=float, default=0.6, help='Temperature parameter for contrastive learning')
parser.add_argument('--consistency', type=float, default=0.1, help='Consistency loss weight')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Consistency rampup epochs')

# Loss weights
parser.add_argument('--sup_loss_weight', type=float, default=1.0, help='Supervised loss weight')
parser.add_argument('--uncl_loss_weight', type=float, default=1.0, help='UnCL loss weight')
parser.add_argument('--fecl_loss_weight', type=float, default=1.0, help='FeCL loss weight')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    """Get consistency weight with rampup"""
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def create_model(ema=False):
    """Create model"""
    # FIX: Remove feature_scaler argument
    model = net_factory_3d(net_type=args.model, in_chns=args.in_ch,
                           class_num=args.num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def patients_to_slices(dataset, patiens_num):
    """Convert patient indices to slice indices"""
    ref_dict = None
    if "ISLES22" in dataset:
        ref_dict = {"1": 23, "2": 23, "3": 23, "4": 26, "5": 21, "6": 28,
                    "7": 23, "8": 25, "9": 24, "10": 25}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def worker_init_fn(worker_id):
    """Worker initialization function for reproducibility"""
    random.seed(args.seed + worker_id)


def update_ema_variables(model, ema_model, alpha, global_step):
    """Update EMA model parameters"""
    # Use alpha here to control the moving average decay
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    """Main training function"""
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    patch_size = args.patch_size

    # Create models
    model = create_model()
    ema_model = create_model(ema=True)

    # Move models to GPU
    model = model.cuda()
    ema_model = ema_model.cuda()

    # Data transformations
    train_transform = T.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        ToTensor(),
    ])

    # Load datasets
    print("Loading ISLES22 dataset...")
    db_train = ISLESDataset(h5_dir=args.root_dir,
                            split='train',
                            transform=train_transform)

    db_val = ISLESDataset(h5_dir=args.root_dir,
                          split='val',
                          transform=T.Compose([ToTensor()]))

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_dir, args.labelnum)
    print("Total slices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                          batch_size, args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    # Optimizer and loss functions
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    # Use PyTorch's built-in loss functions
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # Logging setup
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # Get labeled and unlabeled batches
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_label_batch = label_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            # Noise for consistency training
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            # Forward pass
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # EMA model forward pass
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            # Supervised loss (only on labeled data)
            loss_ce = ce_loss(outputs[:args.labeled_bs], labeled_label_batch[:].long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], labeled_label_batch.unsqueeze(1))
            supervised_loss = 0.5 * (loss_ce + loss_dice)

            # Consistency weight
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # Consistency loss (simplified version instead of UnCL)
            if len(unlabeled_volume_batch) > 0:
                unlabeled_outputs_soft = outputs_soft[args.labeled_bs:]

                # MSE consistency loss between student and teacher predictions
                consistency_loss = F.mse_loss(unlabeled_outputs_soft, ema_output_soft)
                consistency_loss = consistency_weight * consistency_loss
            else:
                consistency_loss = torch.tensor(0.0).cuda()

            # Total loss (simplified without FeCL for now)
            loss = args.sup_loss_weight * supervised_loss + args.uncl_loss_weight * consistency_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA model
            update_ema_variables(model, ema_model, 0.99, iter_num)

            # Learning rate decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # Logging
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            # Print progress every 100 iterations
            if iter_num % 100 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                             (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # Validation and model saving
            if iter_num % 500 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(db_val):
                    metric_i = test_3d_patch.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)