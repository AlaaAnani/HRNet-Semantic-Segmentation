# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss, 0), outputs

class GaussianAugModel(FullModel):
  def __init__(self, model, loss, sigma=0, std=1):
    super(GaussianAugModel, self).__init__(model, loss)
    self.sigma = sigma
    self.std = torch.nn.Parameter(torch.tensor(std).reshape(1, 3, 1, 1), requires_grad=False)

  def forward(self, inputs, labels, sigma=0, *args, **kwargs):
    if sigma > 0:
      inputs = inputs + sigma * torch.randn_like(inputs) / self.std
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss, 0), outputs

def kl_div(input, targets):
  return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def find_boundaries_torch(segmentation_map, margin=1):
    # Ensure segmentation_map is a 2D tensor with shape [Height, Width]
    device = segmentation_map.device
    
    # Expand segmentation_map to have batch and channel dimensions
    segmentation_map_expanded = segmentation_map.unsqueeze(0).unsqueeze(0).to(torch.float32)
    
    # Create a kernel with a single '1' in the center (to compare with neighbors)
    kernel_size = 2 * margin + 1
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    kernel[margin, margin] = 1.
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Use 'valid' convolution to compare center against its surroundings
    center_values = F.conv2d(segmentation_map_expanded, kernel, padding=margin)
    
    # Dilation to represent spread of values across specified margin
    dilated_values = F.max_pool2d(segmentation_map_expanded, kernel_size, stride=1, padding=margin)

    # Boundary condition: pixels where dilated value does not match center value
    boundaries = (dilated_values != center_values).squeeze()

    return boundaries

def get_confusion_matrix(label, pred, size, num_class, ignore=-1, abstain=None):
    """
    Calcute the confusion matrix by given label and pred
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    if len(pred.shape) == 4:
        output = pred.cpu().numpy().transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    else:
        seg_pred = pred.cpu().numpy()
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int_)

    ignore_index = seg_gt != ignore
    # if abstain is not None:
    #     certify_index = np.array(pred != abstain)
    #     ignore_index = ignore_index & certify_index
        
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def acc(pred, label, ignore_label=255, stats=False):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    pred = pred.flatten()
    label = label.flatten()
    
    non_ignore_idx = label != ignore_label
    pos = (pred == label)[non_ignore_idx].sum()
    count = non_ignore_idx.sum()
    if stats:
        d = {}
        d['pixels_count'] = count
        d['positive_count'] = pos
        return pos/count, d
    return pos/count

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr