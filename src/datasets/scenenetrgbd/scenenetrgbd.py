# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""


class SceneNetRGBDBase:
    SPLITS = ['train', 'valid']

    # number of classes without void (NYUv2 classes)
    N_CLASSES = 13

    CLASS_NAMES = ['void',
                   'bed',
                   'books',
                   'ceiling',
                   'chair',
                   'floor',
                   'furniture',
                   'objects',
                   'picture',
                   'sofa',
                   'table',
                   'tv',
                   'wall',
                   'window']

    CLASS_COLORS = [[0, 0, 0],
                    [0, 0, 255],
                    [232, 88, 47],
                    [0, 217, 0],
                    [148, 0, 240],
                    [222, 241, 23],
                    [255, 205, 205],
                    [0, 223, 228],
                    [106, 135, 204],
                    [116, 28, 41],
                    [240, 35, 235],
                    [0, 166, 156],
                    [249, 139, 0],
                    [225, 228, 194]]

    DEPTH_DIR = 'depth'
    RGB_DIR = 'rgb'
    LABELS_13_DIR = 'labels_13'
    LABELS_13_COLORED_DIR = 'labels_13_colored'
