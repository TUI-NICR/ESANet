import os
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

out_dir = './onnx_models'
os.makedirs(out_dir, exist_ok=True)


def export(model, rgb, onnx_name, opset=10):
    onnx_file_path = os.path.join(out_dir, onnx_name)

    torch.onnx.export(model,
                      rgb,
                      onnx_file_path,
                      export_params=True,
                      input_names=['rgb'],
                      output_names=['output'],
                      do_constant_folding=True,
                      verbose=False,
                      opset_version=opset
                      )
    print(f'exported {onnx_name}')


if __name__ == '__main__':

    #%% DeepLabv3 with output stride 8
    H = 1024
    W = 2048

    rgb = torch.rand(size=(1, 3, H, W), dtype=torch.float32)
    deeplabv3 = ptcv_get_model('deeplabv3_resnetd101b_cityscapes',
                               in_size=(H, W),
                               aux=False)
    deeplabv3.eval()
    export(deeplabv3, rgb, 'deeplabv3.onnx')

    #%% PSPNet with output stride 8
    H = 1008  # height and width need to be dividable by 16 and 6
    W = 2016

    rgb = torch.rand(size=(1, 3, H, W), dtype=torch.float32)
    pspnet = ptcv_get_model('pspnet_resnetd101b_cityscapes',
                            in_size=(H, W),
                            aux=False)

    pspnet.eval()
    export(pspnet, rgb, 'pspnet.onnx')

    #%% BiSENet
    H = 1024
    W = 2048

    rgb = torch.rand(size=(1, 3, H, W), dtype=torch.float32)
    bisenet = ptcv_get_model('bisenet_resnet18_celebamaskhq',
                             in_size=(H, W),
                             aux=False)
    bisenet.eval()
    export(bisenet, rgb, 'bisenet.onnx')
