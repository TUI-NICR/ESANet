# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import os
import argparse
import subprocess
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import mock    # pip install mock
import numpy as np
import torch

from src.args import ArgumentParserRGBDSegmentation

from src.models.model_utils import SqueezeAndExcitationTensorRT
from src.datasets.sunrgbd.sunrgbd import SUNRBDBase
from src.prepare_data import prepare_data

with mock.patch('src.models.model_utils.SqueezeAndExcitation',
                SqueezeAndExcitationTensorRT):
    from src.build_model import build_model


def _parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--model', type=str, default='own',
                        choices=['own', 'onnx'],
                        help='The model for which the inference time will be'
                             'measured.')
    parser.add_argument('--model_onnx_filepath', type=str, default=None,
                        help="Path to ONNX model file when --model is 'onnx'")

    # runs
    parser.add_argument('--n_runs', type=int, default=100,
                        help='For how many runs the inference time will be '
                             'measured')
    parser.add_argument('--n_runs_warmup', type=int, default=10,
                        help='How many forward paths trough the model before'
                             'the inference time measurements starts. This is '
                             'necessary as the first runs are slower.')
    # timings
    parser.add_argument('--no_time_pytorch', dest='time_pytorch',
                        action='store_false', default=True,
                        help='Set this if you do not want to measure the'
                             'pytorch times.')
    parser.add_argument('--no_time_tensorrt', dest='time_tensorrt',
                        action='store_false', default=True,
                        help='Set this if you do not want to measure the '
                             'tensorrt times.')
    parser.add_argument('--no_time_onnxruntime', dest='time_onnxruntime',
                        action='store_false', default=True,
                        help='Set this if you do not want to measure the '
                             'onnxruntime times.')
    # plots / export
    parser.add_argument('--plot_timing', default=False, action='store_true',
                        help='Wether to plot the inference time for each'
                             'forward pass')
    parser.add_argument('--plot_outputs', default=False, action='store_true',
                        help='Wether to plot the colored segmentation output'
                             'of the model')
    parser.add_argument('--export_outputs', default=False, action='store_true',
                        help='Whether to export the colored segmentation output'
                             'of the model to png')

    # tensorrt
    parser.add_argument('--trt_workspace', type=int, default=2 << 30,
                        help='default is 2GB')
    parser.add_argument('--trt_floatx', type=int, default=32, choices=[16, 32],
                        help='Whether to measure tensorrt timings with float16'
                             'or float32.')
    parser.add_argument('--trt_batchsize', type=int, default=1)
    parser.add_argument('--trt_onnx_opset_version', type=int, default=10,
                        help='different versions lead to different results but'
                             'not all versions are supported for the following'
                             'tensorrt conversion.')
    parser.add_argument('--trt_dont_force_rebuild', dest='trt_force_rebuild',
                        default=True, action='store_false',
                        help='Possibly already existing trt engine file will '
                             'be reused when providing this argument.')
    parser.add_argument('--onnxruntime_onnx_opset_version', type=int,
                        default=11,
                        help='opset 10 leads to different results compared to'
                             'PyTorch')
    # see: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md
    parser.add_argument('--onnxruntime_trt_max_partition_iterations', type=str,
                        default='15',
                        help='maximum number of iterations allowed in model '
                             'partitioning for TensorRT')

    args = parser.parse_args()
    args.pretrained_on_imagenet = False
    return args


def color_label_from_numpy_array(label):
    cmap = np.asarray(SUNRBDBase.CLASS_COLORS, dtype='uint8')
    return cmap[label]


def get_engine(onnx_filepath,
               engine_filepath,
               trt_floatx=16,
               trt_batchsize=1,
               trt_workspace=2 << 30,
               force_rebuild=True):
    # note that we use onnx2trt from TensorRT Open Source Software Components
    # to convert ONNX files to TensorRT engines
    if not os.path.exists(engine_filepath) or force_rebuild:
        print("Building engine using onnx2trt")
        if trt_floatx == 32:
            print("... this may take a while")
        else:
            print("... this may take -> AGES <-")
        cmd = f'onnx2trt {onnx_filepath}'
        cmd += f' -d {trt_floatx}'    # 16: float16, 32: float32
        cmd += f' -b {trt_batchsize}'    # batchsize
        # cmd += ' -v'    # verbose
        # cmd += ' -l'    # list layers
        cmd += f' -w {trt_workspace}'   # workspace size mb
        cmd += f' -o {engine_filepath}'

        try:
            print(cmd)
            out = subprocess.check_output(cmd,
                                          shell=True,
                                          stderr=subprocess.STDOUT,
                                          universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print("onnx2trt failed:", e.returncode, e.output)
            raise
        print(out)

    print(f"Loading engine: {engine_filepath}")
    with open(engine_filepath, "rb") as f, \
            trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def alloc_buf(engine):
    # input bindings
    in_cpu = []
    in_gpu = []
    for i in range(engine.num_bindings-1):
        shape = trt.volume(engine.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(i))

        in_cpu.append(cuda.pagelocked_empty(shape, dtype))
        in_gpu.append(cuda.mem_alloc(in_cpu[-1].nbytes))

    # output binding
    shape = trt.volume(engine.get_binding_shape(engine.num_bindings-1))
    dtype = trt.nptype(engine.get_binding_dtype(engine.num_bindings-1))
    out_cpu = cuda.pagelocked_empty(shape, dtype)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)

    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def time_inference_pytorch(model,
                           inputs,
                           device,
                           n_runs_warmup=5):
    timings = []
    with torch.no_grad():
        outs = []
        for i in range(len(inputs[0])):
            # use PyTorch to time events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # copy to gpu
            inputs_gpu = [inp[i].to(device) for inp in inputs]

            # model forward pass
            out_pytorch = model(*inputs_gpu)

            # compute argmax and copy back to cpu
            # do not compute argmax for a fair comparison
            # out_pytorch = torch.argmax(out_pytorch, axis=1).squeeze()
            out_pytorch = out_pytorch.cpu()

            end.record()
            torch.cuda.synchronize()

            if i >= n_runs_warmup:
                timings.append(start.elapsed_time(end) / 1e3)

            outs.append(out_pytorch)

    return np.array(timings), outs


def time_inference_tensorrt(onnx_filepath,
                            inputs,
                            trt_floatx=16,
                            trt_batchsize=1,
                            trt_workspace=2 << 30,
                            n_runs_warmup=5,
                            force_tensorrt_engine_rebuild=True):
    # create engine
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'

    engine = get_engine(onnx_filepath, trt_filepath,
                        trt_floatx=trt_floatx,
                        trt_batchsize=trt_batchsize,
                        trt_workspace=trt_workspace,
                        force_rebuild=force_tensorrt_engine_rebuild)
    context = engine.create_execution_context()

    # allocate memory on gpu
    in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)

    timings = []
    pointers = [int(in_) for in_ in in_gpu] + [int(out_gpu)]
    outs = []
    for i in range(len(inputs[0])):
        start_time = time.time()

        # copy to gpu (do not use for loop)
        cuda.memcpy_htod(in_gpu[0], inputs[0][i].numpy())
        if len(inputs) == 2:
            cuda.memcpy_htod(in_gpu[1], inputs[1][i].numpy())

        # model forward pass
        context.execute(1, pointers)

        # copy back to cpu
        cuda.memcpy_dtoh(out_cpu, out_gpu)

        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)

        outs.append(out_cpu.copy())

    return np.array(timings), outs


def time_inference_onnxruntime(onnx_filepath,
                               inputs,
                               n_runs_warmup=5,
                               profile_execution=False):

    # sess = rt.InferenceSession(onnx_filepath)

    opt = onnxruntime.SessionOptions()
    # see: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md
    # ORT_DISABLE_ALL / ORT_ENABLE_BASIC / ORT_ENABLE_EXTENDED / ORT_ENABLE_ALL
    opt.graph_optimization_level = \
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL    # default as well
    opt.intra_op_num_threads = 1    # only useful for cpu provider

    # enable logs
    opt.log_severity_level = 0   # -1

    # see: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Perf_Tuning.md#profiling-and-performance-report
    # load resulting json file using chrome://tracing/ subsequently
    opt.enable_profiling = profile_execution
    sess = onnxruntime.InferenceSession(onnx_filepath, opt)

    # set execution providers (NOTE, the order matters)
    sess.set_providers(['TensorrtExecutionProvider',
                        'CUDAExecutionProvider',
                        'CPUExecutionProvider'])

    timings = []
    outs = []
    for i in range(len(inputs[0])):
        start_time = time.time()

        sess_inputs = {sess.get_inputs()[j].name: inputs[j][i].numpy()
                       for j in range(len(sess.get_inputs()))}
        out = sess.run(None, sess_inputs)[0]    # None -> single output

        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)

        outs.append(out.copy())

    return np.array(timings), outs


if __name__ == '__main__':
    args = _parse_args()
    print(f"args: {vars(args)}")

    print('PyTorch version:', torch.__version__)

    if args.time_tensorrt:
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda

        print('TensorRT version:', trt.__version__)

    if args.time_onnxruntime:
        import onnxruntime

        onnxruntime_profile_execution = True

        # see: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md
        os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = str(2 << 30)
        os.environ['ORT_TENSORRT_MIN_SUBGRAPH_SIZE'] = '1'  # 5
        # note, 1 does not raise an error if not available but enabled
        os.environ['ORT_TENSORRT_FP16_ENABLE'] = '0'   # 1
        os.environ['ORT_TENSORRT_MAX_PARTITION_ITERATIONS'] = \
            args.onnxruntime_trt_max_partition_iterations

        print('ONNXRuntime version:', onnxruntime.__version__)
        print('ONNXRuntime available providers:',
              onnxruntime.get_available_providers())

    gpu_devices = torch.cuda.device_count()

    # prepare inputs ----------------------------------------------------------
    label_downsampling_rates = []
    results_dir = os.path.join(os.path.dirname(__file__),
                               f'inference_results_{args.upsampling}',
                               args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    args.batch_size = 1
    args.batch_size_valid = 1

    rgb_images = []
    depth_images = []
    if args.dataset_dir is not None:
        # get samples from dataset
        _, valid_loader, *additional = prepare_data(args)
        if args.valid_full_res:
            # use full res valid loader
            valid_loader = additional[0]
        dataset = valid_loader.dataset

        for i, sample in enumerate(valid_loader):
            if i == (args.n_runs + args.n_runs_warmup):
                break
            rgb_images.append(sample['image'])
            depth_images.append(sample['depth'])
    else:
        # get random samples
        dataset, preprocessor = prepare_data(args)

        for _ in range(args.n_runs + args.n_runs_warmup):
            img_rgb = np.random.randint(0, 255,
                                        size=(args.height, args.width, 3),
                                        dtype='uint8')
            img_depth = np.random.randint(0, 40000,
                                          size=(args.height, args.width),
                                          dtype='uint16')
            # preprocess
            sample = preprocessor({'image': img_rgb, 'depth': img_depth})
            rgb_images.append(sample['image'][None])
            depth_images.append(sample['depth'][None])

    n_classes_without_void = dataset.n_classes_without_void

    if args.modality == 'rgbd':
        inputs = (rgb_images, depth_images)
    elif args.modality == 'rgb':
        inputs = (rgb_images,)
    elif args.modality == 'depth':
        inputs = (depth_images,)
    else:
        raise NotImplementedError()

    # create model ------------------------------------------------------------
    if args.model is 'onnx' and args.time_pytorch:
        warnings.warn("PyTorch inference timing disabled since onnx model is "
                      "given")
        args.time_pytorch = False

    if args.model == 'own':
        model, device = build_model(args, n_classes_without_void)

        # load weights
        if args.last_ckpt:
            checkpoint = torch.load(args.last_ckpt,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'], strict=True)

        model.eval()
        model.to(device)
    else:
        # onnx model given
        assert args.model_onnx_filepath is not None

    # time inference using PyTorch --------------------------------------------
    if args.time_pytorch:
        timings_pytorch, outs_pytorch = time_inference_pytorch(
            model,
            inputs,
            device,
            n_runs_warmup=args.n_runs_warmup
        )
        print(f'fps pytorch: {np.mean(1/timings_pytorch):0.4f} ± '
              f'{np.std(1/timings_pytorch):0.4f}')

    # time inference using TensorRT -------------------------------------------
    if args.time_tensorrt:
        if args.model_onnx_filepath is None:
            dummy_inputs = [inp[0].to(device) for inp in inputs]

            input_names = [f'input_{i}' for i in range(len(dummy_inputs))]
            output_names = ['output']
            onnx_filepath = './model_tensorrt.onnx'

            torch.onnx.export(model,
                              tuple(dummy_inputs),
                              onnx_filepath,
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              do_constant_folding=True,
                              verbose=False,
                              opset_version=args.trt_onnx_opset_version)
            print(f"ONNX file written to '{onnx_filepath}'.")
        else:
            onnx_filepath = args.model_onnx_filepath

        timings_tensorrt, outs_tensorrt = time_inference_tensorrt(
            onnx_filepath,
            inputs,
            trt_floatx=args.trt_floatx,
            trt_batchsize=args.trt_batchsize,
            trt_workspace=args.trt_workspace,
            n_runs_warmup=args.n_runs_warmup,
            force_tensorrt_engine_rebuild=args.trt_force_rebuild,
        )

        print(f'fps tensorrt: {np.mean(1/timings_tensorrt):0.4f} ± '
              f'{np.std(1/timings_tensorrt):0.4f}')

    # time inference using ONNXRuntime ----------------------------------------
    if args.time_onnxruntime:
        if args.model_onnx_filepath is None:
            dummy_inputs = [inp[0].to(device) for inp in inputs]

            input_names = [f'input_{i}' for i in range(len(dummy_inputs))]
            output_names = ['output']
            onnx_filepath = './model_onnxruntime.onnx'

            torch.onnx.export(
                model,
                tuple(dummy_inputs),
                onnx_filepath,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=True,
                verbose=False,
                opset_version=args.onnxruntime_onnx_opset_version
            )
            print(f"ONNX file written to '{onnx_filepath}'.\n")
            input("Press [ENTER] to continue interfence timing in the same "
                  "run or [CTRL+C] to stop here and rerun the script with "
                  "--model_onnx_filepath to lower memory consumption.")
        else:
            onnx_filepath = args.model_onnx_filepath

        timings_onnxruntime, outs_onnxruntime = time_inference_onnxruntime(
            onnx_filepath,
            inputs,
            n_runs_warmup=args.n_runs_warmup,
            profile_execution=onnxruntime_profile_execution
        )

        print(f'fps onnxruntime: {np.mean(1/timings_onnxruntime):0.4f} ± '
              f'{np.std(1/timings_onnxruntime):0.4f}')

    # plot/export results -----------------------------------------------------
    if args.plot_timing:
        plt.figure()
        if 'timings_pytorch' in locals():
            plt.plot(1 / timings_pytorch, label='pytorch')
        if 'timings_tensorrt' in locals():
            plt.plot(1 / timings_tensorrt, label='tensorrt')
        if 'timings_onnxruntime' in locals():
            plt.plot(1 / timings_onnxruntime, label='onnxruntime')
        plt.xlabel("run")
        plt.ylabel("fps")
        plt.legend()
        plt.title("Inference time")
        plt.show()

    if args.plot_outputs or args.export_outputs:
        if 'timings_pytorch' in locals():
            for i, out_pytorch in enumerate(outs_pytorch):
                argmax_pytorch = np.argmax(out_pytorch.numpy()[0],
                                           axis=0).astype(np.uint8) + 1
                colored = dataset.color_label(argmax_pytorch)

                if args.export_outputs:
                    save_path = os.path.join(results_dir,
                                             f'{i:04d}_jetson_pytorch.png')
                    save_path_colored = os.path.join(
                        results_dir, f'{i:04d}_jetson_pytorch_colored.png')

                    cv2.imwrite(save_path, argmax_pytorch)
                    cv2.imwrite(save_path_colored,
                                cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                if args.plot_outputs:
                    plt.figure()
                    plt.imshow(colored)
                    plt.title("Pytorch")
                    plt.show()

        if 'timings_tensorrt' in locals():
            for i, out_tensorrt in enumerate(outs_tensorrt):
                out = out_tensorrt.reshape(-1, args.height, args.width)

                argmax_tensorrt = np.argmax(out, axis=0).astype(np.uint8) + 1
                colored = dataset.color_label(argmax_tensorrt)

                if args.export_outputs:
                    save_path = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_tensorrt_float{args.trt_floatx}.png'
                    )
                    save_path_colored = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_tensorrt_float{args.trt_floatx}'
                        f'_colored.png'
                    )

                    cv2.imwrite(save_path, argmax_tensorrt)
                    cv2.imwrite(save_path_colored,
                                cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                if args.plot_outputs:
                    plt.figure()
                    plt.imshow(colored)
                    plt.title("TensorRT")
                    plt.show()

        if 'timings_onnxruntime' in locals():
            if os.environ['ORT_TENSORRT_FP16_ENABLE'] == '1':
                floatx = '16'
            else:
                floatx = '32'
            for i, out_onnxruntime in enumerate(outs_onnxruntime):
                out = out_onnxruntime.reshape(-1, args.height, args.width)

                argmax_onnxruntime = np.argmax(out,
                                               axis=0).astype(np.uint8) + 1
                colored = dataset.color_label(argmax_onnxruntime)

                if args.export_outputs:
                    save_path = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_onnxruntime_float{floatx}.png')
                    save_path_colored = os.path.join(
                        results_dir,
                        f'{i:04d}_jetson_onnxruntime_float{floatx}'
                        f'_colored.png')

                    cv2.imwrite(save_path, argmax_onnxruntime)
                    cv2.imwrite(save_path_colored,
                                cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                if args.plot_outputs:
                    plt.figure()
                    plt.imshow(colored)
                    plt.title("ONNXRuntime")
                    plt.show()
