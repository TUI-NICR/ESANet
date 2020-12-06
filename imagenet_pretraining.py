# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Pretraining of ResNet with exchanged encoder blocks on ImageNet. This script
uses Tensorflow Datasets, as the dataset is already available as tfrecords.
Part of this code is copied from:
https://github.com/pytorch/examples/blob/master/imagenet/main.py

"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds

from src.models.resnet import ResNet34, ResNet18
from src.logger import CSVLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Encoder ImageNet Training')
    parser.add_argument('--data_dir',
                        help='path to ImageNet data')
    parser.add_argument('--results_dir',
                        default='./trained_models/imagenet')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34'],
                        help='Wich resnet to train')
    parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='Set this if you have pretrained weights that '
                             'only need to be adapted')
    parser.add_argument('--weight_file', type=str,
                        help='path to weight file for finetuning')
    args = parser.parse_args()

    if args.finetune:
        args.lr = 0.001

    # default learning rate is for batch_size 256
    args.lr = args.lr * args.batch_size / 256
    return args


def main():
    args = parse_args()

    ckpt_dir = os.path.join(args.results_dir, f'{args.encoder}_NBt1D')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    model, device = build_model(args)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.weight_file:
        if device.type == 'cuda':
            checkpoint = torch.load(args.weight_file)
        else:
            checkpoint = torch.load(args.weight_file,
                                    map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print("=> loaded checkpoint '{}' (epoch {})"
              "".format(args.weight_file, checkpoint['epoch']))
    else:
        start_epoch = 0

    train_batches, validation_batches, dataset_info = get_data(args)
    n_train_images = dataset_info.splits['train'].num_examples
    n_val_images = dataset_info.splits['validation'].num_examples

    log_keys = ['acc_train_top-1', 'acc_train_top-5', 'acc_val_top-1',
                'acc_val_top-5']

    log_keys_for_csv = log_keys.copy()
    log_keys_for_csv.extend(['loss_train', 'loss_val', 'epoch', 'lr'])
    csvlogger = CSVLogger(log_keys_for_csv, os.path.join(ckpt_dir, 'logs.csv'),
                          append=True)

    best_acc1 = -1

    for epoch in range(start_epoch, args.epochs):
        if not args.finetune:
            lr = adjust_learning_rate(optimizer, epoch, args)
        else:
            lr = args.lr

        # train for one epoch
        logs = train(train_batches, model, criterion, optimizer, epoch, device,
                     n_train_images, args)

        # evaluate on validation set
        logs = validate(validation_batches, model, criterion, device,
                        n_val_images, logs, args)

        # remember best acc@1 and save checkpoint
        is_best = logs['acc_val_top-1'] > best_acc1
        best_acc1 = max(logs['acc_val_top-1'], best_acc1)

        save_ckpt(ckpt_dir, model, optimizer, epoch, is_best)

        logs['epoch'] = epoch
        logs['lr'] = lr
        logs.pop('time', None)
        csvlogger.write_logs(logs)

    print('done')


def get_data(args):
    print('Preparing data...')
    data, dataset_info = tfds.load(name='imagenet2012',
                                   with_info=True,
                                   as_supervised=True,
                                   download=False,
                                   data_dir=args.data_dir)

    train_batches = (data['train']
                     .shuffle(buffer_size=10000)
                     .map(preprocess_training_image,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
                     .batch(args.batch_size)
                     .prefetch(tf.data.experimental.AUTOTUNE)
                     )

    validation_batches = (data['validation']
                          .map(preprocess_validation_image,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
                          .batch(args.batch_size * 3)
                          .prefetch(tf.data.experimental.AUTOTUNE)
                          )

    return train_batches, validation_batches, dataset_info


def preprocess_training_image(image, label):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    bbox_begin, bbox_size, bbox = tf.image.sample_distorted_bounding_box(
        image_size=tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[3. / 4, 4. / 3],
        area_range=[0.08, 1.0],
        max_attempts=1,
        use_image_if_no_bounding_boxes=True)

    image = tf.slice(image, bbox_begin, bbox_size)
    image = tf.image.resize(image, (224, 224)) / 255.0

    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean

    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    std = tf.reshape(std, [1, 1, 3])
    image = image / std

    image = tf.image.random_flip_left_right(image)
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label


def preprocess_validation_image(image, label):
    image = tf.image.resize(image, (256, 256)) / 255.0
    image = tf.image.central_crop(image, central_fraction=224 / 256)

    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean

    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    std = tf.reshape(std, [1, 1, 3])
    image = image / std

    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label


def build_model(args):
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            if args.encoder == 'resnet34':
                Encoder = ResNet34
            elif args.encoder == 'resnet18':
                Encoder = ResNet18
            self.encoder = Encoder(block='NonBottleneck1D',
                                   pretrained_on_imagenet=False,
                                   activation=nn.ReLU(inplace=True))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def forward(self, images):
            encoder_outs = self.encoder(images)
            enc_down_32, enc_down_16, enc_down_8, enc_down_4 = encoder_outs
            out = self.avgpool(enc_down_32)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out

    model = Classifier()
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Device:', device)
    model.to(device)

    return model, device


def train(train_batches, model, criterion, optimizer, epoch, device,
          n_train_images, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        n_train_images,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, train_batch in enumerate(train_batches):
        images = torch.from_numpy(train_batch[0].numpy())
        target = torch.from_numpy(train_batch[1].numpy())

        # do not train on the last smaller batch
        current_batch_size = len(target)
        if current_batch_size < args.batch_size:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display((i+1)*args.batch_size)

    logs = dict()
    logs['acc_train_top-1'] = top1.avg.cpu().numpy().item()
    logs['acc_train_top-5'] = top5.avg.cpu().numpy().item()
    logs['loss_train'] = losses.avg
    return logs


def validate(validation_batches, model, criterion, device, n_val_images, logs,
             args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        n_val_images,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        examples_done = 0
        for i, validation_batch in enumerate(validation_batches):
            images = torch.from_numpy(validation_batch[0].numpy())
            target = torch.from_numpy(validation_batch[1].numpy())

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            examples_done += len(target)
            if i % args.print_freq == 0:
                progress.display(examples_done)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), flush=True)
        logs['acc_val_top-1'] = top1.avg.cpu().numpy().item()
        logs['acc_val_top-5'] = top5.avg.cpu().numpy().item()
        logs['loss_val'] = losses.avg

    return logs


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_ckpt(ckpt_dir, model, optimizer, epoch, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save best checkpoint with epoch number
    if is_best:
        ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
        path = os.path.join(ckpt_dir, ckpt_model_filename)
        torch.save(state, path)
        print('{:>2} has been successfully saved'.format(path), flush=True)
    # always save latest checkpoint
    torch.save(state, os.path.join(ckpt_dir, 'ckpt_latest.pth'))


if __name__ == '__main__':
    main()
