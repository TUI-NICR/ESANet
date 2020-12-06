# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import copy
import os
import pickle
from torch.utils.data import DataLoader

from src import preprocessing
from src.datasets import Cityscapes
from src.datasets import NYUv2
from src.datasets import SceneNetRGBD
from src.datasets import SUNRGBD


def prepare_data(args, ckpt_dir=None, with_input_orig=False, split=None):
    train_preprocessor_kwargs = {}
    if args.dataset == 'sunrgbd':
        Dataset = SUNRGBD
        dataset_kwargs = {}
        valid_set = 'test'
    elif args.dataset == 'nyuv2':
        Dataset = NYUv2
        dataset_kwargs = {'n_classes': 40}
        valid_set = 'test'
    elif args.dataset == 'cityscapes':
        Dataset = Cityscapes
        dataset_kwargs = {
            'n_classes': 19,
            'disparity_instead_of_depth': True
        }
        valid_set = 'valid'
    elif args.dataset == 'cityscapes-with-depth':
        Dataset = Cityscapes
        dataset_kwargs = {
            'n_classes': 19,
            'disparity_instead_of_depth': False
        }
        valid_set = 'valid'
    elif args.dataset == 'scenenetrgbd':
        Dataset = SceneNetRGBD
        dataset_kwargs = {'n_classes': 13}
        valid_set = 'valid'
        if args.width == 640 and args.height == 480:
            # for SceneNetRGBD, we additionally scale up the images by factor
            # of 2
            train_preprocessor_kwargs['train_random_rescale'] = (1.0*2, 1.4*2)
    else:
        raise ValueError(f"Unknown dataset: `{args.dataset}`")
    if args.aug_scale_min != 1 or args.aug_scale_max != 1.4:
        train_preprocessor_kwargs['train_random_rescale'] = (
            args.aug_scale_min, args.aug_scale_max)

    if split in ['valid', 'test']:
        valid_set = split

    if args.raw_depth:
        # We can not expect the model to predict depth values that are just
        # interpolated and not really there. It is better to let the model only
        # predict the measured depth values and ignore the rest.
        depth_mode = 'raw'
    else:
        depth_mode = 'refined'

    # train data
    train_data = Dataset(
        data_dir=args.dataset_dir,
        split='train',
        depth_mode=depth_mode,
        with_input_orig=with_input_orig,
        **dataset_kwargs
    )

    train_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        depth_mean=train_data.depth_mean,
        depth_std=train_data.depth_std,
        depth_mode=depth_mode,
        phase='train',
        **train_preprocessor_kwargs
    )

    train_data.preprocessor = train_preprocessor

    if ckpt_dir is not None:
        pickle_file_path = os.path.join(ckpt_dir, 'depth_mean_std.pickle')
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                depth_stats = pickle.load(f)
            print(f'Loaded depth mean and std from {pickle_file_path}')
            print(depth_stats)
        else:
            # dump depth stats
            depth_stats = {'mean': train_data.depth_mean,
                           'std': train_data.depth_std}
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(depth_stats, f)
    else:
        depth_stats = {'mean': train_data.depth_mean,
                       'std': train_data.depth_std}

    # valid data
    valid_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        depth_mean=depth_stats['mean'],
        depth_std=depth_stats['std'],
        depth_mode=depth_mode,
        phase='test'
    )

    if args.valid_full_res:
        valid_preprocessor_full_res = preprocessing.get_preprocessor(
            depth_mean=depth_stats['mean'],
            depth_std=depth_stats['std'],
            depth_mode=depth_mode,
            phase='test'
        )

    valid_data = Dataset(
        data_dir=args.dataset_dir,
        split=valid_set,
        depth_mode=depth_mode,
        with_input_orig=with_input_orig,
        **dataset_kwargs
    )

    valid_data.preprocessor = valid_preprocessor

    if args.dataset_dir is None:
        # no path to the actual data was passed -> we cannot create dataloader,
        # return the valid dataset and preprocessor object for inference only
        if args.valid_full_res:
            return valid_data, valid_preprocessor_full_res
        else:
            return valid_data, valid_preprocessor

    # create the data loaders
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              drop_last=True,
                              shuffle=True)

    # for validation we can use higher batch size as activations do not
    # need to be saved for the backwards pass
    batch_size_valid = args.batch_size_valid or args.batch_size
    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size_valid,
                              num_workers=args.workers,
                              shuffle=False)

    if args.valid_full_res:
        valid_loader_full_res = copy.deepcopy(valid_loader)
        valid_loader_full_res.dataset.preprocessor = valid_preprocessor_full_res
        return train_loader, valid_loader, valid_loader_full_res

    return train_loader, valid_loader
