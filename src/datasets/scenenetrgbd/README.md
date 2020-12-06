# SceneNet RGB-D dataset

SceneNet RGB-D expands the previous work of SceneNet to enable large scale photorealistic rendering of indoor scene trajectories. It provides pixel-perfect ground truth for scene understanding problems such as semantic segmentation, instance segmentation, and object detection, and also for geometric computer vision problems such as optical flow, depth estimation, camera pose estimation, and 3D reconstruction. Random sampling permits virtually unlimited scene configurations, and here we provide a set of 5M rendered RGB-D images from over 15K trajectories in synthetic layouts with random but physically simulated object poses. Each layout also has random lighting, camera trajectories, and textures. The scale of this dataset is well suited for pre-training data-driven computer vision techniques from scratch with RGB-D inputs, which previously has been limited by relatively small labelled datasets in NYUv2 and SUN RGB-D. It also provides a basis for investigating 3D scene labelling tasks by providing perfect camera poses and depth data as proxy for a SLAM system.

For more details, see: [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html) and [pySceneNetRGBD](https://github.com/jmccormac/pySceneNetRGBD).

## Prepare dataset

1. Install requirements:
    ```bash
    pip install -r ./requirements.txt [--user]
    ```

2. Download and untar dataset files:  
    ```bash
    # see: https://robotvault.bitbucket.io/scenenet-rgbd.html
    
    SCENENETRGBD_DOWNLOAD_DIR="/path/where/to/store/scenenetrgbd_dowloads"
   
    # train
    wget https://www.doc.ic.ac.uk/~bjm113/scenenet_data/SceneNet-train.tar.gz -P ${SCENENETRGBD_DOWNLOAD_DIR}   # -> 263GB
    wget https://www.doc.ic.ac.uk/~bjm113/scenenet_data/train_protobufs.tar.gz -P ${SCENENETRGBD_DOWNLOAD_DIR}    # -> 323MB
   
    # valid
    wget http://www.doc.ic.ac.uk/~bjm113/scenenet_data/SceneNetRGBD-val.tar.gz -P ${SCENENETRGBD_DOWNLOAD_DIR}   # -> 15GB
    wget http://www.doc.ic.ac.uk/~bjm113/scenenet_data/scenenet_rgbd_val.pb -P ${SCENENETRGBD_DOWNLOAD_DIR}   # -> 31MB
   
    # untar files
    find ${SCENENETRGBD_DOWNLOAD_DIR} -name '*.tar.gz' -exec tar xfvz {} \;
   
    # move train protobuf files
    mv ${SCENENETRGBD_DOWNLOAD_DIR}/train_protobufs/* ${SCENENETRGBD_DOWNLOAD_DIR}
    rm -rf ${SCENENETRGBD_DOWNLOAD_DIR}/train_protobufs
    ```

3. Build protobuf python source file:
    ```
    protoc --python_out=./ scenenet.proto
    ```

3. Convert dataset:
    ```bash
    # cd to this directory
   
    # full dataset:
    # - train: 16x1000 + 1x865 trajectories with 300 views per trajectory -> 5,059,500 samples
    # - valid: 1x1000 trajectories with 300 views per trajectory -> 300,000 samples
    python prepare_dataset.py ../../datasets/scenenetrgbd ${SCENENETRGBD_DOWNLOAD_DIR}
   
    # subsampled dataset 
    # -> randomly pick 3 views from each trajectory for training
    # -> randomly pick 6 views from each trajectory for validation
    # -> pick only views with >= 4 different classes
    # - train: 16x1000 + 1x865 trajectories with 3 views per trajectory -> 50,595 samples
    # - valid: 1x1000 trajectories with 6 views per trajectory -> 6,000 samples
    python prepare_dataset.py ../../../datasets/scenenetrgbd ${SCENENETRGBD_DOWNLOAD_DIR} --n_random_views_to_include_train 3 --n_random_views_to_include_valid 6 --force_at_least_n_classes_in_view 4
    ```

## Use dataset
```python
# see ../../src/prepare_data.py
```