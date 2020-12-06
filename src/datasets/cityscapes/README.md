# Cityscapes dataset

The Cityscapes dataset contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames. 
The dataset is thus an order of magnitude larger than similar previous attempts. Details on [annotated classes](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions) and [examples of our annotations](https://www.cityscapes-dataset.com/examples/#dense-pixel-annotations) are available at this webpage.

For more details, see: [Cityscapes Dataset](https://www.cityscapes-dataset.com/) and [Cityscapes Dataset at GitHub](https://github.com/mcordts/cityscapesScripts).

## Prepare dataset

1. Install requirements:
    ```bash
    pip install -r ./requirements.txt [--user]
    ```

2. Download and unzip dataset files:  
    Use `csDownload` or download the files mentioned below manually from: [Cityscapes Dataset Downloads](https://www.cityscapes-dataset.com/downloads/)
    ```bash
    CITYSCAPES_DOWNLOAD_DIR="/path/where/to/store/cityscapes_downloads"
   
    # using cityscapesScripts
    # use "csDownload -l" to list available packages
    
    # labels
    csDownload gtFine_trainvaltest.zip -d $CITYSCAPES_DOWNLOAD_DIR    # -> 241MB
    # rgb images
    csDownload leftImg8bit_trainvaltest.zip -d $CITYSCAPES_DOWNLOAD_DIR     # -> 11GB
    # disparity images (only upon request)
    csDownload disparity_trainvaltest.zip -d $CITYSCAPES_DOWNLOAD_DIR     # -> 3.5GB  
    # intrinsic and extrinsic camera parameter to calculate depth
    csDownload camera_trainvaltest.zip -d $CITYSCAPES_DOWNLOAD_DIR    # -> 2MB
   
    # unzip files
    find $CITYSCAPES_DOWNLOAD_DIR -name '*.zip' -exec unzip -o {} -d $CITYSCAPES_DOWNLOAD_DIR \;
    ```

3. Convert dataset:
    ```bash
    # cd to this directory
    python prepare_dataset.py ../../../datasets/cityscapes $CITYSCAPES_DOWNLOAD_DIR
    ```

## Use dataset
```python
# see ../../src/prepare_data.py
```