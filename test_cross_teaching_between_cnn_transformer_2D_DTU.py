
import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from config import get_config
from dataloaders import utils
from dataloaders.dataset import BaseDataSets, BaseDataSetsDTU, RandomGenerator, TwoStreamBatchSampler, ResizeGenerator
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume

from scipy.ndimage import zoom
from val_2D import calculate_metric_percase


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


# load the val/test dataset for visual check

# set seed
seed = 42
cudnn.benchmark = True
cudnn.deterministic = False

metric1_list = []
metric1_supervised_list = []
metric2_list = []
metric2_supervised_list = []

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


root_path = "/home/yifan/studium/WindEnergy/published_work/semisupervised/SSL4MIS/data/DTU_Coupoin"
patch_size = [224, 224]
db_val = BaseDataSetsDTU(base_dir=root_path, split="val", transform=transforms.Compose([ResizeGenerator(patch_size)]))
db_train = BaseDataSetsDTU(base_dir=root_path, split="train", supervised=True, transform=transforms.Compose([ResizeGenerator(patch_size)]))

image_size=[1003, 972]
batch_size = 16
labeled_bs = 8
seed = 42
def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                        num_workers=1)
trainloader = DataLoader(db_train, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=False, worker_init_fn=worker_init_fn)


# unet
def create_model(ema=False):
        # Network definition
        model = net_factory(net_type='unet', in_chns=1,
                            class_num=2)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
model1 = create_model() # set num_class to 2?
model1 = model1.cpu()
model1_supervised = create_model()
model1_supervised = model1_supervised.cpu()

unet_weights_path = "/home/yifan/studium/WindEnergy/published_work/semisupervised/SSL4MIS/model/DTU/Cross_teaching_between_cnn_transformer_2D_size1003_iter9400_colab/unet/unet_best_model1.pth"
model1.load_state_dict(torch.load(unet_weights_path))
unet_supervised_weights_path = "/home/yifan/studium/WindEnergy/published_work/semisupervised/SSL4MIS/model/DTU/fully_supervised_unet_size1003_iter9000_colab/unet/unet_best_model.pth"
model1_supervised.load_state_dict(torch.load(unet_supervised_weights_path))

# swin-unet
model2 = ViT_seg(config, img_size=patch_size,
                     num_classes=2).cpu()
model2.load_from(config, eva=True)
model2_supervised = ViT_seg(config, img_size=patch_size,
                     num_classes=2).cpu()
model2_supervised.load_from(config, eva=True, supervised=True)


model2 = model2.cpu()
model2_supervised = model2_supervised.cpu()




def overlap(image_org_norm, prediction):
    rgb_image_org = cv2.cvtColor(image_org_norm, cv2.COLOR_GRAY2BGR)
    r_channel = rgb_image_org[:, :, 0]
    r_channel[prediction == 1] = np.clip(r_channel[prediction == 1] + 100, 0, 255)
    rgb_image_org[:, :, 0] = r_channel

    return rgb_image_org

def metric_calculation(prediction, label):
    metric1_list = []
    for i in range(1, 2):
        metric1_list.append(calculate_metric_percase(
            prediction == i, label == i
        ))

    return metric1_list[0][2]

def inference(model, input,):
    out = torch.softmax(model(input), dim=1)
    # resize
    out = F.interpolate(out, size=(1003, 1003), mode='bilinear', align_corners=False)
    out = torch.argmax(out, dim=1).squeeze() # remove batch size 1 -> 224 224
    out = out.detach().numpy()

    return out

for j, sampled_batch in enumerate(tqdm(valloader, desc="Processing Validation Data", unit="batch")):
# for i, sampled_batch in enumerate(valloader):
    image, label = sampled_batch['image'], sampled_batch['label']
    image_org = image.squeeze(0).squeeze(0).numpy()

    image = F.interpolate(image, size=(224, 224), mode='bicubic', align_corners=False)

    # print("iteration: ", i)
    # print("image: ", batch['image'].shape)
    # print("label: ", batch['label'].shape)
    # print("idx: ", batch['idx'])
    # print("--------------------------------")

    # inference

    image, label = image.squeeze(0).numpy(), label.squeeze(0).numpy() # (1, 224, 224) / (224, 224)

    assert image.shape == (1, 224, 224)
    assert label.shape == (1003, 1003)
    
    prediction1, prediction1_supervised, prediction2, prediction2_supervised = [np.zeros_like(label) for _ in range(4)]
    input = torch.from_numpy(image).unsqueeze(0).float()
    
    model1.eval()
    model1_supervised.eval()
    model2.eval()
    model2_supervised.eval()

    with torch.no_grad():
        prediction1 = inference(model1, input)
        prediction1_supervised = inference(model1_supervised, input)
        prediction2 = inference(model2, input)
        prediction2_supervised = inference(model2_supervised, input)

    # metric



    metric1_list.append(metric_calculation(prediction1, label))
    metric1_supervised_list.append(metric_calculation(prediction1_supervised, label))
    metric2_list.append(metric_calculation(prediction2, label))
    metric2_supervised_list.append(metric_calculation(prediction2_supervised, label))

    # visualization

    # norm
    min_val = np.min(image_org)
    max_val = np.max(image_org)

    image_org_norm = ((image_org - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    prediction1_mark = overlap(image_org_norm, prediction1)
    prediction1_supervised_mark = overlap(image_org_norm, prediction1_supervised)
    prediction2_mark = overlap(image_org_norm, prediction2)
    prediction2_supervised_mark = overlap(image_org_norm, prediction2_supervised)
    gt_mark = overlap(image_org_norm, label)


    # plot
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    # plt.text(0, 0, str(metric_list), fontsize=12, ha='right', va='top', color='red')

    axes[0].imshow(image_org, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(prediction1_mark)
    axes[1].set_title('Cross Teaching Unet')
    axes[1].axis('off')

    axes[2].imshow(prediction2_mark)
    axes[2].set_title('Cross Teaching Swin-Unet')
    axes[2].axis('off')

    axes[3].imshow(gt_mark)
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')

    axes[4].imshow(prediction1_supervised_mark)
    axes[4].set_title('Supervised training Unet')
    axes[4].axis('off')

    axes[5].imshow(prediction2_supervised_mark)
    axes[5].set_title('Supervised training Swin-Unet')
    axes[5].axis('off')

    
    text_labels = ['', str(metric1_list[j]), str(metric2_list[j]), '', str(metric1_supervised_list[j]), str(metric2_supervised_list[j])]
    for ax, text in zip(axes.flatten(), text_labels):
        ax.text(0, 0, text, fontsize=12, color='r', ha='left', va='top', bbox=dict(facecolor='black', alpha=0.5))



    plt.tight_layout()
    plt.savefig('/home/yifan/studium/WindEnergy/published_work/semisupervised/SSL4MIS/model/DTU/result_image_testset/test_{}.png'.format(j), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

print("evaluation done")

data = {
    'Cross Teaching Unet': metric1_list,
    'Supervised training Unet': metric1_supervised_list,
    'Cross Teaching Swin-Unet': metric2_list,
    'Supervised training Swin-Unet': metric2_supervised_list
}

df = pd.DataFrame(data)
mean_values = df.mean()
print("mIoU:")
for column, mean_value in mean_values.items():
    print(f"{column}: {mean_value:.4f}")

df.to_csv('/home/yifan/studium/WindEnergy/published_work/semisupervised/SSL4MIS/model/DTU/metrics_data_testset.csv', index=False)


plt.figure(figsize=(10, 6))
sns.boxplot(data=df)

plt.title('Evaluation on testset')
plt.ylabel('IoU')


plt.savefig('/home/yifan/studium/WindEnergy/published_work/semisupervised/SSL4MIS/model/DTU/box_plot_metrics_testset.png', bbox_inches='tight', dpi=300)

# plt.grid()
# plt.show()