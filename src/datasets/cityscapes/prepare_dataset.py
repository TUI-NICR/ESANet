# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de
"""
import argparse as ap
from collections import OrderedDict
import json
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from cityscapes import CityscapesBase


RGB_DIR = 'leftImg8bit'
PARAMETERS_RAW_DIR = 'camera'
DISPARITY_RAW_DIR = 'disparity'
LABEL_DIR = 'gtFine'


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype ='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath, 'PNG')


def get_files_by_extension(path,
                           extension='.png',
                           flat_structure=False,
                           recursive=False,
                           follow_links=True):
    # check input args
    if not os.path.exists(path):
        raise IOError("No such file or directory: '{}'".format(path))

    if flat_structure:
        filelist = []
    else:
        filelist = {}

    # path is a file
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if extension is None or basename.lower().endswith(extension):
            if flat_structure:
                filelist.append(path)
            else:
                filelist[os.path.dirname(path)] = [basename]
        return filelist

    # get filelist
    filter_func = lambda f: extension is None or f.lower().endswith(extension)
    for root, _, filenames in os.walk(path, topdown=True,
                                      followlinks=follow_links):
        filenames = list(filter(filter_func, filenames))
        if filenames:
            if flat_structure:
                filelist.extend((os.path.join(root, f) for f in filenames))
            else:
                filelist[root] = sorted(filenames)
        if not recursive:
            break

    # return
    if flat_structure:
        return sorted(filelist)
    else:
        return OrderedDict(sorted(filelist.items()))


if __name__ == '__main__':
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare Cityscapes dataset for segmentation.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    parser.add_argument('cityscapes_filepath', type=str,
                        help='filepath to downloaded (and uncompressed) '
                             'Cityscapes files')
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)
    cityscapes_filepath = os.path.expanduser(args.cityscapes_filepath)

    # create output path if not exist
    os.makedirs(output_path, exist_ok=True)

    rgb_filepaths = get_files_by_extension(
        os.path.join(args.cityscapes_filepath, RGB_DIR),
        extension='.png',
        flat_structure=True,
        recursive=True
    )

    label_filepaths = get_files_by_extension(
        os.path.join(args.cityscapes_filepath, LABEL_DIR),
        extension='.png',
        flat_structure=True,
        recursive=True
    )
    label_filepaths = [fp for fp in label_filepaths
                       if os.path.basename(fp).find('labelIds') > -1]

    disparity_raw_filepaths = get_files_by_extension(
        os.path.join(args.cityscapes_filepath, DISPARITY_RAW_DIR),
        extension='.png',
        flat_structure=True,
        recursive=True
    )

    parameters_filepaths = get_files_by_extension(
        os.path.join(args.cityscapes_filepath, PARAMETERS_RAW_DIR),
        extension='.json',
        flat_structure=True,
        recursive=True
    )

    # check for consistency
    assert all(len(l) == 5000 for l in [rgb_filepaths, label_filepaths,
                                        disparity_raw_filepaths,
                                        parameters_filepaths])

    def get_basename(fp):
        # e.g. berlin_000000_000019_camera.json -> berlin_000000_000019
        return '_'.join(os.path.basename(fp).split('_')[:3])

    basenames = [get_basename(f) for f in rgb_filepaths]
    for l in [label_filepaths, disparity_raw_filepaths, parameters_filepaths]:
        assert basenames == [get_basename(f) for f in l]

    filelists = {s: {'rgb': [],
                     'depth_raw': [],
                     'disparity_raw': [],
                     'labels_33': [],
                     'labels_19': []}
                 for s in CityscapesBase.SPLITS}

    # copy rgb images
    print("Copying rgb files")
    for rgb_fp in tqdm(rgb_filepaths):
        basename = os.path.basename(rgb_fp)
        city = os.path.basename(os.path.dirname(rgb_fp))
        subset = os.path.basename(os.path.dirname(os.path.dirname(rgb_fp)))
        subset = 'valid' if subset == 'val' else subset

        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.RGB_DIR, city)
        os.makedirs(dest_path, exist_ok=True)

        # print(rgb_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(rgb_fp, os.path.join(dest_path, basename))
        filelists[subset]['rgb'].append(os.path.join(city, basename))

    # copy depth images
    print("Copying disparity files and creating depth files")
    for d_fp, p_fp in tqdm(zip(disparity_raw_filepaths,
                               parameters_filepaths),
                           total=len(disparity_raw_filepaths)):
        basename = os.path.basename(d_fp)
        city = os.path.basename(os.path.dirname(d_fp))
        subset = os.path.basename(os.path.dirname(os.path.dirname(d_fp)))
        subset = 'valid' if subset == 'val' else subset

        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.DISPARITY_RAW_DIR, city)
        os.makedirs(dest_path, exist_ok=True)

        # print(d_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(d_fp, os.path.join(dest_path, basename))
        filelists[subset]['disparity_raw'].append(os.path.join(city, basename))

        # load disparity file and camera parameters
        disp = cv2.imread(d_fp, cv2.IMREAD_UNCHANGED)
        with open(p_fp, 'r') as f:
            camera_parameters = json.load(f)
        baseline = camera_parameters['extrinsic']['baseline']
        fx = camera_parameters['intrinsic']['fx']

        # convert disparity to depth (im m?)
        # see: https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        disp_mask = disp > 0
        depth = disp.astype('float32')
        depth[disp_mask] = (depth[disp_mask] - 1) / 256
        disp_mask = depth > 0    # avoid divide by zero
        depth[disp_mask] = (baseline * fx) / depth[disp_mask]

        # cast to float16
        depth = depth.astype('float16')

        # save depth image
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.DEPTH_RAW_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        depth_basename = basename.replace('.png', '.npy')
        depth_basename = depth_basename.replace('disparity', 'depth')
        np.save(os.path.join(dest_path, depth_basename), depth)
        filelists[subset]['depth_raw'].append(os.path.join(city,
                                                           depth_basename))

    print("Processing label files")
    mapping_1plus33_to_1plus19 = np.array(
        [CityscapesBase.CLASS_MAPPING_REDUCED[i]
         for i in range(1+33)], dtype='uint8'
    )

    for l_fp in tqdm(label_filepaths):
        basename = os.path.basename(l_fp)
        city = os.path.basename(os.path.dirname(l_fp))
        subset = os.path.basename(os.path.dirname(os.path.dirname(l_fp)))
        subset = 'valid' if subset == 'val' else subset

        # load label with 1+33 classes
        label_full = cv2.imread(l_fp, cv2.IMREAD_UNCHANGED)

        # full: 1+33 classes (original label file -> just copy file)
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.LABELS_FULL_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        # print(l_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(l_fp, os.path.join(dest_path, basename))
        filelists[subset]['labels_33'].append(os.path.join(city, basename))

        # full: 1+33 classes colored
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.LABELS_FULL_COLORED_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        save_indexed_png(os.path.join(dest_path, basename), label_full,
                         colormap=CityscapesBase.CLASS_COLORS_FULL)

        # map full to reduced: 1+33 classes -> 1+19 classes
        label_reduced = mapping_1plus33_to_1plus19[label_full]

        # reduced: 1+19 classes
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.LABELS_REDUCED_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        cv2.imwrite(os.path.join(dest_path, basename), label_reduced)
        filelists[subset]['labels_19'].append(os.path.join(city, basename))

        # reduced: 1+19 classes colored
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesBase.LABELS_REDUCED_COLORED_DIR,
                                 city)
        os.makedirs(dest_path, exist_ok=True)
        save_indexed_png(os.path.join(dest_path, basename), label_reduced,
                         colormap=CityscapesBase.CLASS_COLORS_REDUCED)

    # ensure that filelists are valid and faultless
    def get_identifier(filepath):
        return '_'.join(filepath.split('_')[:3])

    n_samples = 0
    for subset in CityscapesBase.SPLITS:
        identifier_lists = []
        for filelist in filelists[subset].values():
            identifier_lists.append([get_identifier(fp) for fp in filelist])

        assert all(l == identifier_lists[0] for l in identifier_lists[1:])
        n_samples += len(identifier_lists[0])

    assert n_samples == 5000

    # save meta files
    print("Writing meta files")
    np.savetxt(os.path.join(output_path, 'class_names_1+33.txt'),
               CityscapesBase.CLASS_NAMES_FULL,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+33.txt'),
               CityscapesBase.CLASS_COLORS_FULL,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_names_1+19.txt'),
               CityscapesBase.CLASS_NAMES_REDUCED,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+19.txt'),
               CityscapesBase.CLASS_COLORS_REDUCED,
               delimiter=',', fmt='%s')

    for subset in CityscapesBase.SPLITS:
        subset_dict = filelists[subset]
        for key, filelist in subset_dict.items():
            np.savetxt(os.path.join(output_path, f'{subset}_{key}.txt'),
                       filelist,
                       delimiter=',', fmt='%s')
