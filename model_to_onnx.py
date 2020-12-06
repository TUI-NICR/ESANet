# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import os

import numpy as np
import torch

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data

if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (ONNX Export)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--onnx_opset_version', type=int, default=11,
                        help='Different versions lead to different results but'
                             'not all versions are supported for a following'
                             'TensorRT conversion.')
    parser.add_argument('--model_output_name', type=str, default='model',
                        help='Name for the onnx model that will be saved.')
    args = parser.parse_args()
    args.pretrained_on_imagenet = False
    dataset, _ = prepare_data(args, with_input_orig=True)
    model, device = build_model(args, dataset.n_classes_without_void)

    os.makedirs('./onnx_models', exist_ok=True)

    # load weights
    if args.last_ckpt:
        checkpoint = torch.load(args.last_ckpt,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()
    model.to(device)

    rgb = np.random.random(size=(1, 3, args.height, args.width))
    rgb = rgb.astype(np.float32)
    depth = np.random.random(size=(1, 1, args.height, args.width))
    depth = depth.astype(np.float32)

    onnx_file_path = os.path.join('onnx_models',
                                  f'{args.model_output_name}.onnx')
    rgb_torch = torch.from_numpy(rgb)
    depth_torch = torch.from_numpy(depth)

    rgb_torch = rgb_torch.to(device)
    depth_torch = depth_torch.to(device)

    if args.modality == 'rgbd':
        # rgbd
        inp = (rgb_torch, depth_torch)
        input_names = ['rgb', 'depth']
    elif args.modality == 'rgb':
        # rgb
        inp = rgb_torch
        input_names = ['rgb']
    else:
        # depth
        inp = depth_torch
        input_names = ['depth']

    torch.onnx.export(model,
                      inp,
                      onnx_file_path,
                      export_params=True,
                      input_names=input_names,
                      output_names=['output'],
                      do_constant_folding=True,
                      verbose=False,
                      opset_version=args.onnx_opset_version)
