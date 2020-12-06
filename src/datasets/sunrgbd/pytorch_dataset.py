# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import numpy as np
import cv2

from .sunrgbd import SUNRBDBase
from ..dataset_base import DatasetBase


class SUNRGBD(SUNRBDBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 split='train',
                 depth_mode='refined',
                 with_input_orig=False):
        super(SUNRGBD, self).__init__()

        self._n_classes = self.N_CLASSES
        self._cameras = ['realsense', 'kv2', 'kv1', 'xtion']
        assert split in self.SPLITS, \
            f'parameter split must be one of {self.SPLITS}, got {split}'
        self._split = split
        assert depth_mode in ['refined', 'raw']
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            self._data_dir = data_dir
            self.img_dir, self.depth_dir, self.label_dir = \
                self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        self._class_names = self.CLASS_NAMES_ENGLISH
        self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 18320.348967710495, std: 8898.658819551309
        self._depth_mean = 19025.14930492213
        self._depth_std = 9880.916071806689

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def load_image(self, idx):
        if self.camera is None:
            img_dir = self.img_dir[self._split]['list']
        else:
            img_dir = self.img_dir[self._split]['dict'][self.camera]
        fp = os.path.join(self._data_dir, img_dir[idx])
        image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_depth(self, idx):
        if self.camera is None:
            depth_dir = self.depth_dir[self._split]['list']
        else:
            depth_dir = self.depth_dir[self._split]['dict'][self.camera]

        if self._depth_mode == 'raw':
            depth_file = depth_dir[idx].replace('depth_bfx', 'depth')
        else:
            depth_file = depth_dir[idx]

        fp = os.path.join(self._data_dir, depth_file)
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        return depth

    def load_label(self, idx):
        if self.camera is None:
            label_dir = self.label_dir[self._split]['list']
        else:
            label_dir = self.label_dir[self._split]['dict'][self.camera]

        label = np.load(os.path.join(self._data_dir,
                                     label_dir[idx])).astype(np.uint8)

        return label

    def load_file_lists(self):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        img_dir_train_file = _get_filepath('train_rgb.txt')
        depth_dir_train_file = _get_filepath('train_depth.txt')
        label_dir_train_file = _get_filepath('train_label.txt')

        img_dir_test_file = _get_filepath('test_rgb.txt')
        depth_dir_test_file = _get_filepath('test_depth.txt')
        label_dir_test_file = _get_filepath('test_label.txt')

        img_dir = dict()
        depth_dir = dict()
        label_dir = dict()

        for phase in ['train', 'test']:
            img_dir[phase] = dict()
            depth_dir[phase] = dict()
            label_dir[phase] = dict()

        img_dir['train']['list'], img_dir['train']['dict'] = \
            self.list_and_dict_from_file(img_dir_train_file)
        depth_dir['train']['list'], depth_dir['train']['dict'] = \
            self.list_and_dict_from_file(depth_dir_train_file)
        label_dir['train']['list'], label_dir['train']['dict'] = \
            self.list_and_dict_from_file(label_dir_train_file)

        img_dir['test']['list'], img_dir['test']['dict'] = \
            self.list_and_dict_from_file(img_dir_test_file)
        depth_dir['test']['list'], depth_dir['test']['dict'] = \
            self.list_and_dict_from_file(depth_dir_test_file)
        label_dir['test']['list'], label_dir['test']['dict'] = \
            self.list_and_dict_from_file(label_dir_test_file)

        return img_dir, depth_dir, label_dir

    def list_and_dict_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        dictionary = dict()
        for cam in self.cameras:
            dictionary[cam] = [i for i in file_list if cam in i]

        return file_list, dictionary

    def __len__(self):
        if self.camera is None:
            return len(self.img_dir[self._split]['list'])
        return len(self.img_dir[self._split]['dict'][self.camera])
