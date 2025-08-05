import os
import argparse

import torch
from networks.net_factory import net_factory_3d
from utils.test_3d_patch import test_all_case_BraTS19

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="../data/BraTS2019", help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='BraTS2019', help='exp_name')
parser.add_argument('--model', type=str, choices=['unet_3D', 'vnet'], default='unet_3D', help='Model architecture')
parser.add_argument('--gpu_id', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing?')
parser.add_argument('--labelnum', type=int, default=25, help='labeled data')
parser.add_argument('--gamma', type=float, default=2.0, help='Focusing parameter for hard positives/negatives in FeCL (γ)')
parser.add_argument('--beta_min', type=float, default=0.5, help='Minimum value for entropy weighting (β)')
parser.add_argument('--beta_max', type=float, default=5.0, help='Maximum value for entropy weighting (β)')
parser.add_argument('--s_beta', type=float, default=None, help='If provided, use this static beta for UnCLoss instead of adaptive beta.')
parser.add_argument('--temp', type=float, default=0.6, help='Temperature for contrastive softmax scaling (optimal: 0.6)')
parser.add_argument('--use_focal', type=int, default=1, help='Whether to use focal weighting (1 for True, 0 for False)')
parser.add_argument('--use_teacher_loss', type=int, default=1, help='Use teacher-based auxiliary loss (1 for True, 0 for False)')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--max_iterations', type=int, default=20000, help='Maximum number of training iterations')
parser.add_argument('--in_ch', type=int, default=1, help='Input channels')
parser.add_argument('--feature_scaler', type=float, default=2, help='Feature scaler for the model')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if args.s_beta is not None:
    beta_str = f"_beta{args.s_beta}"
else:
    beta_str = f"_beta{args.beta_min}-{args.beta_max}"

focal_str = "Focal" if bool(args.use_focal) else "NoFocal"
gamma_str = f"_gamma{args.gamma}" if bool(args.use_focal) else ""
teacher_str = "Teacher" if bool(args.use_teacher_loss) else "NoTeacher"

snapshot_path = (
    f"../models/{args.exp}/{args.model.upper()}_{args.labelnum}labels_"
    f"{args.consistency_type}{gamma_str}_{focal_str}_{teacher_str}_temp{args.temp}"
    f"{beta_str}_max_iterations{args.max_iterations}"
)
test_save_path = "{}/{}_predictions/".format(snapshot_path, args.exp, args.labelnum, args.model)

num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

image_list = []
with open(os.path.join(args.root_path, "val.txt"), 'r') as f:
    case_ids = [line.strip() for line in f if line.strip()]
    image_list = [os.path.join(args.root_path, "data", f"{case_id}.h5") for case_id in case_ids]    

def test_calculate_metric():
    net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes, scaler=args.feature_scaler)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    model.load_state_dict(torch.load(save_model_path, map_location='cpu'))
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case_BraTS19(model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=16, stride_z=4,
                           save_result=False, test_save_path=test_save_path,
                           metric_detail=args.detail, nms=args.nms)

    return avg_metric

if __name__ == '__main__':
    print("Running BraTS19 Test Evaluation...")
    print("=" * 60)
    metric = test_calculate_metric()
    print("=" * 60)
    print("FINAL AVERAGE METRICS:")
    print(f"{'Dice':<8} {'Jaccard':<8} {'HD95':<8} {'ASD':<8}")
    print(f"{metric[0]:<8.5f} {metric[1]:<8.5f} {metric[2]:<8.5f} {metric[3]:<8.5f}")
    print("=" * 60)

# python test_BraTS19.py --labelnum 25
# python test_BraTS19.py --labelnum 25 --use_focal 0 --use_teacher_loss 0 --s_beta 3.0




