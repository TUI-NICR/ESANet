# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

This code is partially adapted from RedNet
(https://github.com/JinDongJiang/RedNet)
"""
import os
import sys

import pandas as pd
import numpy as np
from torch import nn
import torch


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='none',
            ignore_index=-1
        )
        self.ce_loss.to(device)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            # mask = targets > 0
            targets_m = targets.clone()
            targets_m -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())

            number_of_pixels_per_class = \
                torch.bincount(targets.flatten().type(self.dtype),
                               minlength=self.num_classes)
            divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
            # losses.append(torch.sum(loss_all) / torch.sum(mask.float()))

        return losses


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='sum',
            ignore_index=-1
        )
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None,
            reduction='sum',
            ignore_index=-1
        )
        self.ce_loss.to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets_m >= 0)  # only non void pixels

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.nr_pixels.cpu().numpy().item()

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0


def print_log(epoch, local_count, count_inter, dataset_size, loss, time_inter,
              learning_rates):
    print_string = 'Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]'.format(
        epoch, local_count, dataset_size,
        100. * local_count / dataset_size)
    for i, lr in enumerate(learning_rates):
        print_string += '   lr_{}: {:>6}'.format(i, round(lr, 10))
    print_string += '   Loss: {:0.6f}'.format(loss.item())
    print_string += '  [{:0.2f}s every {:>4} data]'.format(time_inter,
                                                          count_inter)
    print(print_string, flush=True)


def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
                          best_miou_epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou': best_miou,
        'best_miou_epoch': best_miou_epoch
    }
    ckpt_model_filename = "ckpt_latest.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
            print('Best mIoU:', best_miou)
        else:
            best_miou = 0

        if 'best_miou_epoch' in checkpoint:
            best_miou_epoch = checkpoint['best_miou_epoch']
            print('Best mIoU epoch:', best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)


def get_best_checkpoint(ckpt_dir, key='mIoU_test'):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, 'logs.csv')
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        miou = data[key][idx]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pth')
    assert ckpt_path is not None, f'No trainings found at {ckpt_dir}'
    assert os.path.exists(ckpt_path), \
        f'There is no weights file named {ckpt_path}'
    print(f'Best mIoU: {100*miou:0.2f} at epoch: {epoch}')
    return ckpt_path
