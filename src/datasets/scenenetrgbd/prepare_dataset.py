# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

See:
- https://robotvault.bitbucket.io/scenenet-rgbd.html
- https://github.com/jmccormac/pySceneNetRGBD
"""
import argparse as ap
import os
import shutil
import traceback

import numpy as np
from PIL import Image
from termcolor import cprint
from tqdm import tqdm
import cv2

from scenenetrgbd import SceneNetRGBDBase

try:
    import scenenet_pb2 as sn
except ImportError:
    print("Please run `protoc --python_out=./ scenenet.proto` first! "
          "(see README.md)")
    raise


# dict to convert WordNetIDs to NYUv2 classes
# taken from https://github.com/jmccormac/pySceneNetRGBD/blob/master/convert_instance2class.py#L39
WNID_TO_NYU = {'04593077': 4, '03262932': 4, '02933112': 6, '03207941': 7,
               '03063968': 10, '04398044': 7, '04515003': 7, '00017222': 7,
               '02964075': 10, '03246933': 10, '03904060': 10, '03018349': 6,
               '03786621': 4, '04225987': 7, '04284002': 7, '03211117': 11,
               '02920259': 1, '03782190': 11, '03761084': 7, '03710193': 7,
               '03367059': 7, '02747177': 7, '03063599': 7, '04599124': 7,
               '20000036': 10, '03085219': 7, '04255586': 7, '03165096': 1,
               '03938244': 1, '14845743': 7, '03609235': 7, '03238586': 10,
               '03797390': 7, '04152829': 11, '04553920': 7, '04608329': 10,
               '20000016': 4, '02883344': 7, '04590933': 4, '04466871': 7,
               '03168217': 4, '03490884': 7, '04569063': 7, '03071021': 7,
               '03221720': 12, '03309808': 7, '04380533': 7, '02839910': 7,
               '03179701': 10, '02823510': 7, '03376595': 4, '03891251': 4,
               '03438257': 7, '02686379': 7, '03488438': 7, '04118021': 5,
               '03513137': 7, '04315948': 7, '03092883': 10, '15101854': 6,
               '03982430': 10, '02920083': 1, '02990373': 3, '03346455': 12,
               '03452594': 7, '03612814': 7, '06415419': 7, '03025755': 7,
               '02777927': 12, '04546855': 12, '20000040': 10, '20000041': 10,
               '04533802': 7, '04459362': 7, '04177755': 9, '03206908': 7,
               '20000021': 4, '03624134': 7, '04186051': 7, '04152593': 11,
               '03643737': 7, '02676566': 7, '02789487': 6, '03237340': 6,
               '04502670': 7, '04208936': 7, '20000024': 4, '04401088': 7,
               '04372370': 12, '20000025': 4, '03956922': 7, '04379243': 10,
               '04447028': 7, '03147509': 7, '03640988': 7, '03916031': 7,
               '03906997': 7, '04190052': 6, '02828884': 4, '03962852': 1,
               '03665366': 7, '02881193': 7, '03920867': 4, '03773035': 12,
               '03046257': 12, '04516116': 7, '00266645': 7, '03665924': 7,
               '03261776': 7, '03991062': 7, '03908831': 7, '03759954': 7,
               '04164868': 7, '04004475': 7, '03642806': 7, '04589593': 13,
               '04522168': 7, '04446276': 7, '08647616': 4, '02808440': 7,
               '08266235': 10, '03467517': 7, '04256520': 9, '04337974': 7,
               '03990474': 7, '03116530': 6, '03649674': 4, '04349401': 7,
               '01091234': 7, '15075141': 7, '20000028': 9, '02960903': 7,
               '04254009': 7, '20000018': 4, '20000020': 4, '03676759': 11,
               '20000022': 4, '20000023': 4, '02946921': 7, '03957315': 7,
               '20000026': 4, '20000027': 4, '04381587': 10, '04101232': 7,
               '03691459': 7, '03273913': 7, '02843684': 7, '04183516': 7,
               '04587648': 13, '02815950': 3, '03653583': 6, '03525454': 7,
               '03405725': 6, '03636248': 7, '03211616': 11, '04177820': 4,
               '04099969': 4, '03928116': 7, '04586225': 7, '02738535': 4,
               '20000039': 10, '20000038': 10, '04476259': 7, '04009801': 11,
               '03909406': 12, '03002711': 7, '03085602': 11, '03233905': 6,
               '20000037': 10, '02801938': 7, '03899768': 7, '04343346': 7,
               '03603722': 7, '03593526': 7, '02954340': 7, '02694662': 7,
               '04209613': 7, '02951358': 7, '03115762': 9, '04038727': 6,
               '03005285': 7, '04559451': 7, '03775636': 7, '03620967': 10,
               '02773838': 7, '20000008': 6, '04526964': 7, '06508816': 7,
               '20000009': 6, '03379051': 7, '04062428': 7, '04074963': 7,
               '04047401': 7, '03881893': 13, '03959485': 7, '03391301': 7,
               '03151077': 12, '04590263': 13, '20000006': 1, '03148324': 6,
               '20000004': 1, '04453156': 7, '02840245': 2, '04591713': 7,
               '03050864': 7, '03727837': 5, '06277280': 11, '03365592': 5,
               '03876519': 8, '03179910': 7, '06709442': 7, '03482252': 7,
               '04223580': 7, '02880940': 7, '04554684': 7, '20000030': 9,
               '03085013': 7, '03169390': 7, '04192858': 7, '20000029': 9,
               '04331277': 4, '03452741': 7, '03485997': 7, '20000007': 1,
               '02942699': 7, '03231368': 10, '03337140': 7, '03001627': 4,
               '20000011': 6, '20000010': 6, '20000013': 6, '04603729': 10,
               '20000015': 4, '04548280': 12, '06410904': 2, '04398951': 10,
               '03693474': 9, '04330267': 7, '03015149': 9, '04460038': 7,
               '03128519': 7, '04306847': 7, '03677231': 7, '02871439': 6,
               '04550184': 6, '14974264': 7, '04344873': 9, '03636649': 7,
               '20000012': 6, '02876657': 7, '03325088': 7, '04253437': 7,
               '02992529': 7, '03222722': 12, '04373704': 4, '02851099': 13,
               '04061681': 10, '04529681': 7}

RGB_DIR = 'photo'
DEPTH_DIR = 'depth'
INSTANCE_DIR = 'instance'

PROTOBUF_FILENAMES = {
    'train': [f'scenenet_rgbd_train_{i}.pb' for i in range(17)],
    'valid': ['scenenet_rgbd_val.pb']
}

N_VIEWS_DEFAULT = 300


def _mapping_for_instances(instances):
    mapping = np.zeros(len(instances), dtype='uint8')
    for inst in instances:
        if inst.instance_type == sn.Instance.BACKGROUND:
            # map background to void class -> 0
            continue
        mapping[inst.instance_id] = WNID_TO_NYU[inst.semantic_wordnet_id]
    return mapping


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype ='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath, 'PNG')


if __name__ == '__main__':
    # use fixed seed
    np.random.seed(42)

    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare SceneNetRGBD dataset for segmentation.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    parser.add_argument('scenenetrgbd_filepath', type=str,
                        help='filepath to downloaded (and uncompressed) '
                             'SceneNetRGBD files')
    parser.add_argument('--n_random_views_to_include_train', type=int,
                        choices=list(range(1, N_VIEWS_DEFAULT+1)),
                        default=N_VIEWS_DEFAULT,
                        help=f'Number of views to randomly pick from each '
                             f'trajectory to build the training set. In '
                             f'SceneNetRGBD each trajectory '
                             f'is comprised of {N_VIEWS_DEFAULT} views. '
                             f'Use this parameter to subsample the train set.')
    parser.add_argument('--n_random_views_to_include_valid', type=int,
                        choices=list(range(1, N_VIEWS_DEFAULT+1)),
                        default=N_VIEWS_DEFAULT,
                        help=f'Number of views to randomly pick from each '
                             f'trajectory to build the validation set. In '
                             f'SceneNetRGBD each trajectory '
                             f'is comprised of {N_VIEWS_DEFAULT} views. '
                             f'Use this parameter to subsample the validation '
                             f'set.')
    parser.add_argument('--force_at_least_n_classes_in_view', type=int,
                        default=-1,
                        help=f'Minimum number of classes to be present in a '
                             f'view that is picked randomly. Note, missing '
                             f'views are counted and supplementary taken from '
                             f'subsequent trajectories')
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)

    # process files
    # see: https://github.com/jmccormac/pySceneNetRGBD/blob/master/convert_instance2class.py

    filelists = {s: {'rgb': [],
                     'depth': [],
                     'labels_13': []}
                 for s in SceneNetRGBDBase.SPLITS}

    for split in SceneNetRGBDBase.SPLITS:
        split_output_path = os.path.join(output_path, split)

        if split == 'train':
            n_random_views_to_include = args.n_random_views_to_include_train
        else:
            n_random_views_to_include = args.n_random_views_to_include_valid
        cprint(f"Processing {split} set (subsampling from {N_VIEWS_DEFAULT} "
               f"views to {n_random_views_to_include} view(s) with at least "
               f"{args.force_at_least_n_classes_in_view} classes in each "
               f"view)",
               color='green')

        def _source_path(render_path, type_dir):
            split_ = 'val' if split == 'valid' else split
            return os.path.join(args.scenenetrgbd_filepath,
                                split_,
                                render_path,
                                type_dir)

        def _output_path(render_path, type_dir):
            path = os.path.join(args.output_path,
                                split,
                                type_dir,
                                render_path)
            os.makedirs(path, exist_ok=True)
            return path

        # load protobuf files
        print('Loading protobuf files')
        trajectory_list = []
        for fn in PROTOBUF_FILENAMES[split]:
            trajectories = sn.Trajectories()

            with open(os.path.join(args.scenenetrgbd_filepath, fn), 'rb') as f:
                trajectories.ParseFromString(f.read())

            trajectory_list.append(trajectories)

        # progress bar
        n_views = n_random_views_to_include
        n_trajectories = sum([len(traj.trajectories)
                              for traj in trajectory_list])

        pbar = tqdm(total=n_trajectories*n_views,
                    disable=False)

        print(f'Preparing {split} set')

        # count number of view that are missing due to sampling
        n_views_missing = 0

        for trajs in trajectory_list:
            # process each set of trajectories

            for traj in trajs.trajectories:
                # process each trajectory
                files_traj = {'rgb': [],
                              'depth': [],
                              'labels_13': []}
                traj_failed = False

                # process views in trajectory
                mapping = _mapping_for_instances(traj.instances)

                n_views_picked = 0
                n_views_to_pick = n_random_views_to_include + n_views_missing
                for i in np.random.permutation(N_VIEWS_DEFAULT):
                    # get view
                    view = traj.views[i]

                    # load label
                    s_fp = os.path.join(_source_path(traj.render_path,
                                                     INSTANCE_DIR),
                                        f'{view.frame_num}.png')
                    instance = cv2.imread(s_fp, cv2.IMREAD_UNCHANGED)

                    # map instances to classes
                    try:
                        label_13 = mapping[instance]
                    except Exception:
                        # for some strange reason, for some view, we get an
                        # index error (instance id is greater than the
                        # instances given by traj.instances)
                        cprint(f"View `{s_fp}` skipped (mapping failed)",
                               color='red')
                        print(traceback.format_exc())

                        # stop if an error occurred
                        traj_failed = True
                        break

                        # just skip these strange views
                        # continue

                        # print("Use fallback: map unknown instances to void")
                        # label_13 = np.zeros_like(instance, dtype='uint8')
                        # for inst, c in enumerate(mapping):
                        #    label_13[instance == inst] = c

                    # check number of classes in view
                    if args.force_at_least_n_classes_in_view != -1:
                        n = len(np.unique(label_13))
                        if n < args.force_at_least_n_classes_in_view:
                            # process next view
                            cprint(f"View `{s_fp}` skipped ({n} < "
                                   f"{args.force_at_least_n_classes_in_view} "
                                   f"classes in view)",
                                   color='yellow')
                            continue

                    # view is fine, pick it
                    n_views_picked += 1

                    # copy rgb file
                    s_fp = os.path.join(_source_path(traj.render_path,
                                                     RGB_DIR),
                                        f'{view.frame_num}.jpg')
                    d_fp = os.path.join(_output_path(traj.render_path,
                                                     SceneNetRGBDBase.RGB_DIR),
                                        f'{view.frame_num}.jpg')
                    shutil.copy(s_fp, d_fp)
                    files_traj['rgb'].append(
                        os.path.join(traj.render_path, f'{view.frame_num}.jpg')
                    )

                    # copy depth file
                    s_fp = os.path.join(_source_path(traj.render_path,
                                                     DEPTH_DIR),
                                        f'{view.frame_num}.png')
                    d_fp = os.path.join(_output_path(traj.render_path,
                                                     SceneNetRGBDBase.DEPTH_DIR),
                                        f'{view.frame_num}.png')
                    shutil.copy(s_fp, d_fp)
                    files_traj['depth'].append(
                        os.path.join(traj.render_path, f'{view.frame_num}.png')
                    )

                    # save label
                    d_fp = os.path.join(_output_path(traj.render_path,
                                                     SceneNetRGBDBase.LABELS_13_DIR),
                                        f'{view.frame_num}.png')
                    cv2.imwrite(d_fp, label_13)
                    files_traj['labels_13'].append(
                        os.path.join(traj.render_path, f'{view.frame_num}.png')
                    )

                    # save colored label
                    save_indexed_png(
                        os.path.join(_output_path(traj.render_path,
                                                  SceneNetRGBDBase.LABELS_13_COLORED_DIR),
                                     f'{view.frame_num}.png'),
                         label_13,
                         colormap=SceneNetRGBDBase.CLASS_COLORS
                    )

                    if n_views_picked == n_views_to_pick:
                        # enough views processed
                        break

                if traj_failed:
                    # an error occurred while picking views from this
                    # trajectory -> discard all files from this trajectory
                    n_views_picked = 0

                    cprint(f"Trajectory `{traj.render_path}` skipped (at least"
                           f" one mapping failed)",
                           color='red')

                    # clean already copied files
                    to_delete = [
                        _output_path(traj.render_path,
                                     SceneNetRGBDBase.RGB_DIR),
                        _output_path(traj.render_path,
                                     SceneNetRGBDBase.DEPTH_DIR),
                        _output_path(traj.render_path,
                                     SceneNetRGBDBase.LABELS_13_DIR),
                        _output_path(traj.render_path,
                                     SceneNetRGBDBase.LABELS_13_COLORED_DIR)
                    ]
                    for path in to_delete:
                        if os.path.exists(path):
                            shutil.rmtree(path)
                            cprint(f"Removing `{path}`", color='yellow')
                else:
                    # views are fine, extend filelists
                    filelists[split]['rgb'].extend(files_traj['rgb'])
                    filelists[split]['depth'].extend(files_traj['depth'])
                    filelists[split]['labels_13'].extend(files_traj['labels_13'])

                if n_views_picked < n_views_to_pick:
                    cprint(f"Not enough views picked from trajectory "
                           f"`{traj.render_path}` "
                           f"({n_views_picked}/{n_views_to_pick}).",
                           color='magenta')

                if n_views_picked > n_random_views_to_include:
                    cprint(f"Additionally picked "
                           f"{n_views_picked-n_random_views_to_include} "
                           f"view(s) from trajectory `{traj.render_path}`.",
                           color='magenta')

                n_views_missing += n_random_views_to_include - n_views_picked
                if n_views_missing > 0:
                    cprint(f"In total {n_views_missing} views are missing.",
                           color='magenta')

                # update progress bar
                pbar.update(n_views_picked)

        # destroy progress bar
        pbar.close()

        if n_views_missing > 0:
            cprint(f"{n_views_missing} views are missing.", color='yellow')

    # ensure that filelists are valid and faultless
    def get_identifier(filepath):
        identifier = os.path.splitext(filepath)[0]
        to_replace = os.path.dirname(os.path.dirname(os.path.dirname(identifier)))
        return identifier.replace(to_replace, '')

    n_samples = 0
    for subset in SceneNetRGBDBase.SPLITS:
        identifier_lists = []
        for filelist in filelists[subset].values():
            identifier_lists.append([get_identifier(fp) for fp in filelist])

        assert all(l == identifier_lists[0] for l in identifier_lists[1:])

    # save meta files
    print("Writing meta files")
    np.savetxt(os.path.join(output_path, 'class_names_1+13.txt'),
               SceneNetRGBDBase.CLASS_NAMES,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+13.txt'),
               SceneNetRGBDBase.CLASS_COLORS,
               delimiter=',', fmt='%s')

    for subset in SceneNetRGBDBase.SPLITS:
        subset_dict = filelists[subset]
        for key, filelist in subset_dict.items():
            np.savetxt(os.path.join(output_path, f'{subset}_{key}.txt'),
                       filelist,
                       delimiter=',', fmt='%s')
