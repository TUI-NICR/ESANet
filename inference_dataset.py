# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]
    n_classes = data_loader.dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    cameras = data_loader.dataset.cameras

    for camera in cameras:
        with data_loader.dataset.filter_camera(camera):
            for sample in data_loader:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)

                label = sample['label_orig']

                _, image_h, image_w = label.shape

                with torch.no_grad():
                    if args.modality == 'rgbd':
                        pred = model(image, depth)
                    elif args.modality == 'rgb':
                        pred = model(image)
                    else:
                        pred = model(depth)

                fig, axs = plt.subplots(args.batch_size, 4,
                                        figsize=(16, args.batch_size * 3))
                [ax.set_axis_off() for ax in axs.ravel()]

                pred = F.interpolate(pred, (image_h, image_w),
                                     mode='bilinear', align_corners=False)
                pred = torch.max(pred, 1)[1] + 1
                pred = pred.cpu().numpy().squeeze().astype(np.uint8)
                for i in range(args.batch_size):
                    pred_colored = data_loader.dataset.color_label(pred[i])
                    label_colored = data_loader.dataset.color_label(label[i])

                    axs[i, 0].imshow(sample['image_orig'][i])
                    axs[i, 1].imshow(sample['depth_orig'][i])
                    axs[i, 2].imshow(label_colored)
                    axs[i, 3].imshow(pred_colored)

                # one batch is enough
                break

            plt.suptitle(f"Dataset: {args.dataset} ({camera}), "
                         f"Model: {args.ckpt_path}")
            # plt.tight_layout()
            plt.show()
