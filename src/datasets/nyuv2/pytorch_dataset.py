# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import cv2
import numpy as np

from ..dataset_base import DatasetBase
from .nyuv2 import NYUv2Base


class NYUv2(NYUv2Base, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 n_classes=40,
                 split='train',
                 depth_mode='refined',
                 with_input_orig=False):
        super(NYUv2, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        assert depth_mode in ['refined', 'raw']

        self._n_classes = n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['kv1']

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            # load filenames
            fp = os.path.join(self._data_dir,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        # load class names
        self._class_names = getattr(self, f'CLASS_NAMES_{self._n_classes}')

        # load class colors
        self._class_colors = np.array(
            getattr(self, f'CLASS_COLORS_{self._n_classes}'),
            dtype='uint8'
        )

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 2769.0187903686697, std: 1350.4174149841133
        self._depth_mean = 2841.94941272766
        self._depth_std = 1417.2594281672277

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

    def _load(self, directory, filename):
        fp = os.path.join(self._data_dir,
                          self.split,
                          directory,
                          f'{filename}.png')
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im

    def load_image(self, idx):
        return self._load(self.RGB_DIR, self._filenames[idx])

    def load_depth(self, idx):
        if self._depth_mode == 'raw':
            return self._load(self.DEPTH_RAW_DIR, self._filenames[idx])
        else:
            return self._load(self.DEPTH_DIR, self._filenames[idx])

    def load_label(self, idx):
        return self._load(self.LABELS_DIR_FMT.format(self._n_classes),
                          self._filenames[idx])

    def __len__(self):
        return len(self._filenames)
