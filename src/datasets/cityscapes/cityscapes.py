# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from cityscapesscripts.helpers.labels import labels


class CityscapesBase:
    SPLITS = ['train', 'valid', 'test']

    # number of classes without void/unlabeled and license plate (class 34)
    N_CLASSES = [19, 33]

    # 1+33 classes (0: unlabeled)
    CLASS_NAMES_FULL = [label.name for label in labels]
    CLASS_COLORS_FULL = [label.color for label in labels]

    # 1+19 classes (0: void)
    CLASS_NAMES_REDUCED = ['void'] + [label.name for label in labels
                                      if not label.ignoreInEval]
    CLASS_COLORS_REDUCED = [(0, 0, 0)] + [label.color for label in labels
                                          if not label.ignoreInEval]
    # forward mapping (0: unlabeled) + 33 classes -> (0: void) + 19 classes
    CLASS_MAPPING_REDUCED = {
        c: labels[c].trainId+1 if not labels[c].ignoreInEval else 0
        for c in range(1+33)
    }

    # DEPTH_DIR = 'depth'    # refined depth does not exist
    DEPTH_RAW_DIR = 'depth_raw'
    DISPARITY_RAW_DIR = 'disparity_raw'
    RGB_DIR = 'rgb'

    LABELS_FULL_DIR = 'labels_33'
    LABELS_FULL_COLORED_DIR = 'labels_33_colored'

    LABELS_REDUCED_DIR = 'labels_19'
    LABELS_REDUCED_COLORED_DIR = 'labels_19_colored'
