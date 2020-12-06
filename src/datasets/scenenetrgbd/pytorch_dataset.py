# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import numpy as np
import cv2

from ..dataset_base import DatasetBase
from .scenenetrgbd import SceneNetRGBDBase


class SceneNetRGBD(SceneNetRGBDBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 n_classes=13,
                 split='train',
                 depth_mode='refined',
                 with_input_orig=False):
        super(SceneNetRGBD, self).__init__()
        assert split in self.SPLITS
        assert n_classes == self.N_CLASSES
        assert depth_mode == 'refined'

        self._n_classes = n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['camera1']    # just a dummy camera name

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            # load file lists
            def _loadtxt(fn):
                return np.loadtxt(os.path.join(self._data_dir, fn), dtype=str)

            self._files = {
                'rgb': _loadtxt(f'{self._split}_rgb.txt'),
                'depth': _loadtxt(f'{self._split}_depth.txt'),
                'label': _loadtxt(f'{self._split}_labels_{self._n_classes}.txt')
            }
            assert all(len(l) == len(self._files['rgb'])
                       for l in self._files.values())
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        # class names, class colors, and label directory
        self._class_names = self.CLASS_NAMES
        self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')

        self._depth_mean = 4006.9281155769777
        self._depth_std = 2459.7763971709933

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
                          filename)
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im

    def load_image(self, idx):
        return self._load(self.RGB_DIR, self._files['rgb'][idx])

    def load_depth(self, idx):
        return self._load(self.DEPTH_DIR, self._files['depth'][idx])

    def load_label(self, idx):
        return self._load(self.LABELS_13_DIR, self._files['label'][idx])

    def __len__(self):
        return len(self._files['rgb'])
