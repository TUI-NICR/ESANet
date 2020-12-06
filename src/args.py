# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse


class ArgumentParserRGBDSegmentation(argparse.ArgumentParser):
    def set_common_args(self):
        # paths
        self.add_argument('--results_dir',
                          default='./results')
        self.add_argument('--last_ckpt', default='', type=str, metavar='PATH',
                          help='path to latest checkpoint')
        self.add_argument('--pretrained_dir',
                          default='./trained_models/imagenet',
                          help='path to the pretrained resnets with different'
                               'encoder block')

        # pretraining
        self.add_argument('--pretrained_scenenet', default='',
                          help='the path to the weights pretrained on '
                               'SceneNet')
        self.add_argument('--no_imagenet_pretraining',
                          dest='pretrained_on_imagenet', default=True,
                          action='store_false',
                          help='Encoder will be initialized randomly. '
                               '(If not set encoder will be initialized with '
                               'weights pretrained on ImageNet)')
        self.add_argument('--finetune', default=None, type=str,
                          help='path to the weights you want to finetune on.')
        self.add_argument('--freeze', default=0, type=int,
                          help='number of epochs the whole model (except for '
                               'the output layer(s) are freezed. Might be '
                               'useful when using a pretrained '
                               'model on scenenet.')

        # input dimensions
        self.add_argument('--batch_size', type=int, default=8,
                          help='batch size for training')
        self.add_argument('--batch_size_valid', type=int, default=None,
                          help='batch size for validation. Can be typically '
                               '2-3 times as large as the batch size for '
                               'training. If None it will be the same as '
                               '--batch_size.')
        self.add_argument('--height', type=int, default=480,
                          help='height of the training images. '
                               'Images will be resized to this height.')
        self.add_argument('--width', type=int, default=640,
                          help='width of the training images. '
                               'Images will be resized to this width.')

        # epochs
        self.add_argument('--epochs', default=500, type=int, metavar='N',
                          help='number of total epochs to run')

        # training hyper parameters
        self.add_argument('--lr', '--learning-rate', default=0.01,
                          type=float,
                          help='maximum learning rate. When using one_cycle '
                               'as --lr_scheduler lr will first increase to '
                               'the value provided and then slowly decrease.')
        self.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                          help='weight decay')
        self.add_argument('--momentum', default=0.9, type=float, metavar='M',
                          help='momentum')
        self.add_argument('--optimizer', type=str, default='SGD',
                          choices=['SGD', 'Adam'])
        self.add_argument('--class_weighting', type=str,
                          default='median_frequency',
                          choices=['median_frequency', 'logarithmic', 'None'],
                          help='which weighting mode to use for weighting the '
                               'classes of the unbalanced dataset'
                               'for the loss function during training.')
        self.add_argument('--c_for_logarithmic_weighting', type=str,
                          default=1.02,
                          help='the value for restricting the class weights. '
                               'The value is only used when --class_weighting '
                               'is set to logarithmic')
        self.add_argument('--he_init', dest='he_init', default=False,
                          action='store_true',
                          help='Set this if you want to initialize '
                               'convolution layers with He initialization.')
        self.add_argument('--valid_full_res', default=False,
                          action='store_true',
                          help='Whether to validate on the full resolution '
                               '(for cityscapes).')

        # model
        self.add_argument('--activation', type=str, default='relu',
                          choices=['relu', 'swish', 'hswish'],
                          help='Which activation function to use in the model')
        self.add_argument('--encoder', type=str, default='resnet34',
                          choices=['resnet18', 'resnet34', 'resnet50'],
                          help='Wich encoder to use for rgb features.'
                               'if parameter --encoder_depth is None the same '
                               'encoder is used for the depth features.')
        self.add_argument('--encoder_block', type=str,
                          default='NonBottleneck1D',
                          choices=['BasicBlock', 'NonBottleneck1D'],
                          help='The block that is used in the ResNet encoder.'
                               'The NonBottleneck1D achieves better '
                               'results than the BasicBlock.')
        self.add_argument('--nr_decoder_blocks', type=int, default=[3],
                          nargs='+',
                          help='How many decoder blocks are used in each '
                               'decoder module. This variable is only used '
                               'when decoder_block != "None"')
        self.add_argument('--encoder_depth', type=str, default=None,
                          choices=['resnet18', 'resnet34', 'resnet50', 'None'],
                          help='Take a different encoder for the depth '
                               'features than for the rgb features. Parameter '
                               'will only be used when modality is rgbd.')
        self.add_argument('--modality', type=str, default='rgbd',
                          choices=['rgbd', 'rgb', 'depth'],
                          help='If modality is rgb or depth the model '
                               'consists of one encoder and one decoder. If '
                               'modality is rgbd the model consists of two '
                               'decoders for rgb and depth images '
                               'respectively and one decoder for the combined '
                               'features. If multi task is chosen, the model '
                               'consists of one rgb encoder and two decoders '
                               'for the segmentation and the depth prediction'
                               ' respectively.')
        self.add_argument('--encoder_decoder_fusion', type=str,
                          default='add',
                          choices=['add', 'None'],
                          help='How to fuse encoder feature maps into the '
                               'decoder. If None no encoder feature maps are '
                               'fused into the decoder.')
        self.add_argument('--context_module', type=str, default='ppm',
                          choices=['ppm', 'None', 'ppm-1-2-4-8', 'appm',
                                   'appm-1-2-4-8'],
                          help='Which context module to use.')
        self.add_argument('--channels_decoder', type=int, default=128,
                          help='How many feature maps to use in the decoder. '
                               'This is only used when you set:'
                               '--decoder_channels_mode constant')
        self.add_argument('--decoder_channels_mode', default='decreasing',
                          choices=['constant', 'decreasing'],
                          help='constant: the number of channels in the '
                               'decoder stays the same.'
                               'decreasing: the channel number is decreasing '
                               'as the resolution is increasing. Note that '
                               'than the argument --channels_decoder is '
                               'ignored.')
        self.add_argument('--fuse_depth_in_rgb_encoder', default='SE-add',
                          choices=['SE-add', 'add', 'None'],
                          help='Fuses the depth feature maps in the rgb '
                               'encoder maps over several layers in the '
                               'encoder.')
        self.add_argument('--upsampling', default='learned-3x3-zeropad',
                          choices=['nearest', 'bilinear', 'learned-3x3',
                                   'learned-3x3-zeropad'],
                          help='How to usample in the decoder. '
                               'Bilinear upsampling can cause problems'
                               'with conversion to TensorRT. learned-3x3 '
                               'mimics a bilinear interpolation with nearest '
                               'neighbor interpolation and a 3x3 conv '
                               'afterwards')

        # dataset
        self.add_argument('--dataset', default='nyuv2',
                          choices=['sunrgbd',
                                   'nyuv2',
                                   'cityscapes', 'cityscapes-with-depth',
                                   'scenenetrgbd'])
        self.add_argument('--dataset_dir',
                          default=None,
                          help='Path to dataset root.',)
        self.add_argument('--raw_depth', action='store_true', default=False,
                          help='Whether to use the raw depth values instead of'
                               'the refined depth values')
        self.add_argument('--aug_scale_min', default=1.0, type=float,
                          help='the minimum scale for random rescaling the '
                               'training data.')
        self.add_argument('--aug_scale_max', default=1.4, type=float,
                          help='the maximum scale for random rescaling the '
                               'training data.')

        # others
        self.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                          help='number of data loading workers')
        self.add_argument('--debug', default=False, action='store_true',
                          help='Only one batch in training and validation.')
