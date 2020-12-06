## External models

To reproduce the timings of other models we compared in our paper to:
1. Clone the external repository.
2. Apply our changes to the external repository (patch) to enable ONNX export 
   and TensorRT engine creation.
3. Extract ONNX model(s).
4. Determine the inference time on NVIDIA Jetson AGX Xavier using:
    ```bash
    cd /path/to/this/repository
    
    # note that height H and width W might be different for external models
    python3 inference_time_whole_model.py \
        --dataset nyuv2 \
        --model onnx \
        --model_onnx_filepath /path/to/the/extracted/onnx/model \
        --height H \
        --width W \
        --no_time_onnxruntime \
        --trt_floatx 16
    ```

#### FuseNet
```bash
# clone repository
git clone https://github.com/MehmetAygun/fusenet-pytorch
cd fusenet-pytorch
git checkout 1a316437899a0c402b0986e126b719da4fadd891

# install requirements

# apply patch
patch -s -p0 < ../fusenet.patch

# extract onnx model(s)
python ./to_onnx.py    # this will fail due to unsuported max_unpool2d
```

#### RedNet
```bash
# clone repository
git clone https://github.com/JindongJiang/RedNet
cd RedNet
git checkout 1835eb525195f751ca586f0eca0a3c5659373dcc

# install requirements

# apply patch
patch -s -p0 < ../rednet.patch

# extract onnx model(s)
python ./to_onnx.py
```

### SSMA (PyTorch version):
```bash
# clone repository
git clone https://github.com/metahexane/ssma_pytorch
cd ssma_pytorch
git checkout e8bf6328739191cac9072d5409cc6b7feb512025

# install requirements

# apply patch
patch -s -p0 < ../ssma.patch

# extract onnx model(s)
python ./src/to_onnx.py
```

### RDFNet (PyTorch version):
```bash
# clone repository
git clone https://github.com/charlesCXK/PyTorch_Semantic_Segmentation
cd PyTorch_Semantic_Segmentation
git checkout da459730efe495153edaf6cb5cf70545daa3e3e4

# install requirements

# apply patch
patch -s -p0 < ../rdfnet.patch

# extract onnx model(s)
python ./RDFNet_PyTorch/to_onnx.py
```

### ACNet:
```bash
# clone repository
git clone https://github.com/anheidelonghu/ACNet
cd ACNet
git checkout 5b9b48678cc948afa9149dc6a5e47b42734698bc

# install requirements

# apply patch
patch -s -p0 < ../acnet.patch

# extract onnx model(s)
python ./to_onnx.py
```

### SA-Gate:
```bash
# clone repository
git clone https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch
cd RGBD_Semantic_Segmentation_PyTorch
git checkout 32b3f86822d278103a13ea6f93f9668d3b631398

# install requirements

# apply patch
patch -s -p0 < ../sa_gate.patch

# extract onnx model(s)
python ./model/SA-Gate.nyu/to_onnx.py
```

### ERFNet:
```bash
# clone repository
git clone https://github.com/Eromera/erfnet_pytorch
cd erfnet_pytorch
git checkout d4a46faf9e465286c89ebd9c44bc929b2d213fb3

# install requirements

# apply patch
patch -s -p0 < ../erfnet.patch

# extract onnx model(s)
python ./train/to_onnx.py
```

### LEDNet:
```bash
# clone repository
git clone https://github.com/xiaoyufenfei/LEDNet
cd LEDNet
git checkout 5d900d9cfabb3091c952b79be34246aea0608e42

# install requirements

# apply patch
patch -s -p0 < ../lednet.patch

# extract onnx model(s)
python ./train/to_onnx.py
```

### ESPNetv2:
```bash
# clone repository
git clone https://github.com/sacmehta/ESPNetv2
cd ESPNetv2
git checkout 11f93af5f9c535704a129838cfe78a2323dc800b

# install requirements

# apply patch
patch -s -p0 < ../espnetv2.patch

# extract onnx model(s)
python ./segmentation/to_onnx.py
```

### SwiftNet:
```bash
# clone repository
git clone https://github.com/orsic/swiftnet
cd swiftnet
git checkout 2b88990e1ab674e8ef7cb533a1d8d49ef34ac93d

# install requirements

# apply patch
patch -s -p0 < ../swiftnet.patch

# extract onnx model(s)
python ./to_onnx.py
```

### LDFNet:
```bash
# clone repository
git clone https://github.com/shangweihung/LDFNet
cd LDFNet
git checkout e918e9358e2b85f5f985303f75709243704fa1de

# install requirements

# apply patch
patch -s -p0 < ../ldfnet.patch

# extract onnx model(s)
python ./model/to_onnx.py
```

### BiSeNet, PSPNet, DeepLabv3 (taken from pytorchcv):
```bash
# clone repository
git clone https://github.com/osmr/imgclsmob
cd imgclsmob
git checkout 92f642d0d5f4300ca7a88302c05039e13cb5132c

# apply patch
patch -s -p0 < ../pytorchcv.patch

# install package
pip3 install pytorch

# extract onnx model(s)
cd ..
python from_pytorch_cv.py
```
