# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import csv


class CSVLogger(object):
    def __init__(self, keys, path, append=False):
        super(CSVLogger, self).__init__()
        self._keys = keys
        self._path = path
        if append is False or not os.path.exists(self._path):
            with open(self._path, 'w') as f:
                w = csv.DictWriter(f, self._keys)
                w.writeheader()

    def write_logs(self, logs):
        with open(self._path, 'a') as f:
            w = csv.DictWriter(f, self._keys)
            w.writerow(logs)
