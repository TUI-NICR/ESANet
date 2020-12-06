# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

See: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""
import argparse as ap
import os
from tempfile import gettempdir
import urllib.request

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from nyuv2 import NYUv2Base


# https://github.com/VainF/nyuv2-python-toolkit/blob/master/splits.mat
SPLITS_FILEPATH = os.path.join(os.path.dirname(__file__),
                               'splits.mat')
# https://github.com/VainF/nyuv2-python-toolkit/blob/master/class13Mapping.mat
CLASSES_13_FILEPATH = os.path.join(os.path.dirname(__file__),
                                  'class13Mapping.mat')
# https://github.com/VainF/nyuv2-python-toolkit/blob/master/classMapping40.mat
CLASSES_40_FILEPATH = os.path.join(os.path.dirname(__file__),
                                  'classMapping40.mat')
# see: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
DATASET_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_filepath, display_progressbar=False):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1],
                             disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url,
                                   filename=output_filepath,
                                   reporthook=t.update_to)


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath, 'PNG')


def dimshuffle(input_img, from_axes, to_axes):
    # check axes parameter
    if from_axes.find('0') == -1 or from_axes.find('1') == -1:
        raise ValueError("`from_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if to_axes.find('0') == -1 or to_axes.find('1') == -1:
        raise ValueError("`to_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if len(from_axes) != len(input_img.shape):
        raise ValueError("Number of axis given by `from_axes` does not match "
                         "the number of axis in `input_img`")

    # handle special cases for channel axis
    to_axes_c = to_axes.find('c')
    from_axes_c = from_axes.find('c')
    # remove channel axis (only grayscale image)
    if to_axes_c == -1 and from_axes_c >= 0:
        if input_img.shape[from_axes_c] != 1:
            raise ValueError('Cannot remove channel axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_c)
        from_axes = from_axes.replace('c', '')

    # handle special cases for batch axis
    to_axes_b = to_axes.find('b')
    from_axes_b = from_axes.find('b')
    # remove batch axis
    if to_axes_b == -1 and from_axes_b >= 0:
        if input_img.shape[from_axes_b] != 1:
            raise ValueError('Cannot remove batch axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_b)
        from_axes = from_axes.replace('b', '')

    # add new batch axis (in front)
    if to_axes_b >= 0 and from_axes_b == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'b' + from_axes

    # add new channel axis (in front)
    if to_axes_c >= 0 and from_axes_c == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'c' + from_axes

    return np.transpose(input_img, [from_axes.find(a) for a in to_axes])


if __name__ == '__main__':
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare NYUv2 dataset for segmentation.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    parser.add_argument('--mat_filepath', default=None,
                        help='filepath to NYUv2 mat file')
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)
    if args.mat_filepath is None:
        mat_filepath = os.path.join(gettempdir(), 'nyu_depth_v2_labeled.mat')
    else:
        mat_filepath = os.path.expanduser(args.mat_filepath)

    # download mat file if mat_filepath does not exist
    if not os.path.exists(mat_filepath):
        print(f"Downloading mat file to: `{mat_filepath}`")
        download_file(DATASET_URL, mat_filepath, display_progressbar=True)

    # create output path if not exist
    os.makedirs(output_path, exist_ok=True)

    # load mat file and extract images
    print(f"Loading mat file: `{mat_filepath}`")
    with h5py.File(mat_filepath, 'r') as f:
        rgb_images = np.array(f['images'])
        labels = np.array(f['labels'])
        depth_images = np.array(f['depths'])
        raw_depth_images = np.array(f['rawDepths'])

    # dimshuffle images
    rgb_images = dimshuffle(rgb_images, 'bc10', 'b01c')
    labels = dimshuffle(labels, 'b10', 'b01')
    depth_images = dimshuffle(depth_images, 'b10', 'b01')
    raw_depth_images = dimshuffle(raw_depth_images, 'b10', 'b01')

    # convert depth images (m to mm)
    depth_images = (depth_images * 1e3).astype('uint16')
    raw_depth_images = (raw_depth_images * 1e3).astype('uint16')

    # load split file (note that returned indexes start from 1)
    splits = loadmat(SPLITS_FILEPATH)
    train_idxs, test_idxs = splits['trainNdxs'][:, 0], splits['testNdxs'][:, 0]

    # load classes and class mappings (number of classes are without void)
    classes_40 = loadmat(CLASSES_40_FILEPATH)
    classes_13 = loadmat(CLASSES_13_FILEPATH)['classMapping13'][0][0]
    # class_names = {
    #     894: ['void'] + [c[0] for c in classes_40['allClassName'][0]],
    #     40: ['void'] + [c[0] for c in classes_40['className'][0]],
    #     13: ['void'] + [c[0] for c in classes_13[1][0]]
    # }
    mapping_894_to_40 = np.concatenate([[0], classes_40['mapClass'][0]])
    mapping_40_to_13 = np.concatenate([[0], classes_13[0][0]])

    # get color (1 (void) + n_colors)
    colors = {
        894: np.array(NYUv2Base.CLASS_COLORS_894, dtype='uint8'),
        40: np.array(NYUv2Base.CLASS_COLORS_40, dtype='uint8'),
        13: np.array(NYUv2Base.CLASS_COLORS_13, dtype='uint8')
    }

    # save images
    for idxs, set_ in zip([train_idxs, test_idxs], ['train', 'test']):
        print(f"Processing set: {set_}")
        set_dir = NYUv2Base.SPLIT_DIRS[set_]
        rgb_base_path = os.path.join(output_path, set_dir, NYUv2Base.RGB_DIR)
        depth_base_path = os.path.join(output_path, set_dir,
                                       NYUv2Base.DEPTH_DIR)
        depth_raw_base_path = os.path.join(output_path, set_dir,
                                           NYUv2Base.DEPTH_RAW_DIR)
        labels_894_base_path = os.path.join(output_path, set_dir,
                                            NYUv2Base.LABELS_DIR_FMT.format(894))
        labels_40_base_path = os.path.join(
            output_path, set_dir, NYUv2Base.LABELS_DIR_FMT.format(40))
        labels_13_base_path = os.path.join(
            output_path, set_dir, NYUv2Base.LABELS_DIR_FMT.format(13))
        labels_894_colored_base_path = os.path.join(
            output_path, set_dir, NYUv2Base.LABELS_COLORED_DIR_FMT.format(894))
        labels_40_colored_base_path = os.path.join(
            output_path, set_dir, NYUv2Base.LABELS_COLORED_DIR_FMT.format(40))
        labels_13_colored_base_path = os.path.join(
            output_path, set_dir, NYUv2Base.LABELS_COLORED_DIR_FMT.format(13))

        os.makedirs(rgb_base_path, exist_ok=True)
        os.makedirs(depth_base_path, exist_ok=True)
        os.makedirs(depth_raw_base_path, exist_ok=True)
        os.makedirs(labels_894_base_path, exist_ok=True)
        os.makedirs(labels_13_base_path, exist_ok=True)
        os.makedirs(labels_40_base_path, exist_ok=True)
        os.makedirs(labels_894_colored_base_path, exist_ok=True)
        os.makedirs(labels_13_colored_base_path, exist_ok=True)
        os.makedirs(labels_40_colored_base_path, exist_ok=True)

        for idx in tqdm(idxs):
            # convert index from Matlab to [REST OF WORLD]
            idx_ = idx - 1

            # rgb image
            cv2.imwrite(os.path.join(rgb_base_path, f'{idx:04d}.png'),
                        cv2.cvtColor(rgb_images[idx_], cv2.COLOR_RGB2BGR))

            # depth image
            cv2.imwrite(os.path.join(depth_base_path, f'{idx:04d}.png'),
                        depth_images[idx_])

            # raw depth image
            cv2.imwrite(os.path.join(depth_raw_base_path, f'{idx:04d}.png'),
                        raw_depth_images[idx_])

            # label with 1+894 classes
            label_894 = labels[idx_]
            cv2.imwrite(os.path.join(labels_894_base_path, f'{idx:04d}.png'),
                        label_894)

            # colored label image
            # (normal png16 as this type does not support indexed palettes)
            label_894_colored = colors[894][label_894]
            cv2.imwrite(os.path.join(labels_894_colored_base_path,
                                     f'{idx:04d}.png'),
                        cv2.cvtColor(label_894_colored, cv2.COLOR_RGB2BGR))

            # label with 1+40 classes
            label_40 = mapping_894_to_40[label_894].astype('uint8')
            cv2.imwrite(os.path.join(labels_40_base_path, f'{idx:04d}.png'),
                        label_40)
            # colored label image
            # (indexed png8 with color palette)
            save_indexed_png(os.path.join(labels_40_colored_base_path,
                                          f'{idx:04d}.png'),
                             label_40, colors[40])

            # label with 1+13 classes
            label_13 = mapping_40_to_13[label_40].astype('uint8')
            cv2.imwrite(os.path.join(labels_13_base_path, f'{idx:04d}.png'),
                        label_13)
            # colored label image
            # (indexed png8 with color palette)
            save_indexed_png(os.path.join(labels_13_colored_base_path,
                                          f'{idx:04d}.png'),
                             label_13, colors[13])

    # save meta files
    print("Writing meta files")
    np.savetxt(os.path.join(output_path, 'class_names_1+13.txt'),
               NYUv2Base.CLASS_NAMES_13,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+13.txt'),
               NYUv2Base.CLASS_COLORS_13,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_names_1+40.txt'),
               NYUv2Base.CLASS_NAMES_40,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+40.txt'),
               NYUv2Base.CLASS_COLORS_40,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_names_1+894.txt'),
               NYUv2Base.CLASS_NAMES_894,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+894.txt'),
               NYUv2Base.CLASS_COLORS_894,
               delimiter=',', fmt='%s')

    # splits
    np.savetxt(os.path.join(output_path,
                            NYUv2Base.SPLIT_FILELIST_FILENAMES['train']),
               train_idxs,
               fmt='%04d')
    np.savetxt(os.path.join(output_path,
                            NYUv2Base.SPLIT_FILELIST_FILENAMES['test']),
               test_idxs,
               fmt='%04d')

    # remove downloaded file
    if args.mat_filepath is None:
        print(f"Removing downloaded mat file: `{mat_filepath}`")
        os.remove(mat_filepath)
