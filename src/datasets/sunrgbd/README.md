# SUNRGB-D Dataset

The SUNRGB-D dataset is comprised of images of four different cameras, namely 
Intel Realsense, Asus Xtion, and Microsoft Kinect 1 and 2.
It contains all images from NYUv2, manually selected images from Berkeley 
B3DO and SUN3D as well as newly shot images.

It contains 10.335 densely labeled pairs of aligned RGB and depth images.

For more details, see: [SUNRGB-D dataset](https://rgbd.cs.princeton.edu/)

## Prepare dataset

1. Install requirements:
```bash
python -m pip install -r ./requirements.txt
```

2. Convert dataset:
```bash
# cd to this directory and run:
python prepare_dataset.py ../../../datasets/sunrgbd

# if you already downloaded the files manually, use:
python prepare_dataset.py ../../../datasets/sunrgbd \
    --toolbox_filepath /path/to/SUNRGBDtoolbox.zip \
    --data_filepath /path/to/SUNRGBD_data.zip
```

## Use dataset
```python
# see ../../src/prepare_data.py
```
