#!/bin/bash
set -o xtrace

DEFAULT_PARAMS='--nr_decoder_blocks 3 --upsampling learned-3x3'

TIME_PYTORCH='--n_runs_warmup 50 --n_runs 50 --no_time_onnxruntime --no_time_tensorrt --debug'
TIME_TRT32='--n_runs_warmup 50 --n_runs 50 --no_time_onnxruntime --no_time_pytorch --debug'
TIME_TRT16='--n_runs_warmup 50 --n_runs 50 --no_time_onnxruntime --no_time_pytorch --trt_floatx 16 --debug'

SED_PYTORCH="sed -n 's/.*fps pytorch: \([0-9.]*\) ± \([0-9.]*\).*$/\1 \2,/p'"
SED_TRT32="sed -n 's/.*fps tensorrt: \([0-9.]*\) ± \([0-9.]*\).*$/\1 \2,/p'"
SED_TRT16="sed -n 's/.*fps tensorrt: \([0-9.]*\) ± \([0-9.]*\).*$/\1 \2/p'"

# ------------------------------------------------------------------------------
# SUNRGBD
# ------------------------------------------------------------------------------
DATASET="--dataset sunrgbd --dataset_dir ./datasets/sunrgbd"
RESULTS_FILE='./sunrgbd_timings.csv'

# rgbd -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --encoder resnet18"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --encoder resnet18 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34 (=baseline)
PARAMS="${DEFAULT_PARAMS} --encoder resnet34"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --encoder resnet34 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --encoder resnet50"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# ------------------------------------------------------------------------------
# NYUv2
# ------------------------------------------------------------------------------
DATASET='--dataset nyuv2 --dataset_dir ./datasets/nyuv2'
RESULTS_FILE='./nyuv2_timings.csv'

# rgbd -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --encoder resnet18"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --encoder resnet18 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34 (=baseline)
PARAMS="${DEFAULT_PARAMS} --encoder resnet34"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --encoder resnet34 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --encoder resnet50"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# rgb -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet18"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet18 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet34"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet34 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet50"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# depth ------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --modality depth --encoder resnet18"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --modality depth --encoder resnet18 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --modality depth --encoder resnet34"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --modality depth --encoder resnet34 --encoder_block BasicBlock"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --modality depth --encoder resnet50"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# decoder blocks ---------------------------------------------------------------
# 4
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 4"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# 3 6 4
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 3 6 4"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# 3 (=baseline)
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 3"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# 2
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 2"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# 1
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 1"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# 0
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 0"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# 0 and just 128 channels
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 0 --decoder_channels_mode constant"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# upsampling -------------------------------------------------------------------
# learned-3x3 (=baseline)
PARAMS="--upsampling learned-3x3 --nr_decoder_blocks 3"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# nearest
PARAMS="--upsampling nearest --nr_decoder_blocks 3"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# bilinear
PARAMS="--upsampling bilinear --nr_decoder_blocks 3"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# context module, se, skip ---------------------------------------------------
# no skip
PARAMS="${DEFAULT_PARAMS} --encoder_decoder_fusion None"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# no se
PARAMS="${DEFAULT_PARAMS} --fuse_depth_in_rgb_encoder add"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# no context module
PARAMS="${DEFAULT_PARAMS} --context_module None"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# no se, no skip
PARAMS="${DEFAULT_PARAMS} --fuse_depth_in_rgb_encoder add --encoder_decoder_fusion None"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# no context module, no skip
PARAMS="${DEFAULT_PARAMS} --context_module None --encoder_decoder_fusion None"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# no context module, no se
PARAMS="${DEFAULT_PARAMS} --context_module None --fuse_depth_in_rgb_encoder add"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# no context module, no se, no skip
PARAMS="${DEFAULT_PARAMS} --context_module None --fuse_depth_in_rgb_encoder add --encoder_decoder_fusion None"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE


# ------------------------------------------------------------------------------
# Cityscapes (512x1024)
# ------------------------------------------------------------------------------
DATASET="--dataset cityscapes --dataset_dir ./datasets/cityscapes"
RESULTS_FILE='./cityscapes_timings.csv'

# rgbd -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --encoder resnet18 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --encoder resnet18 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34 (=baseline)
PARAMS="${DEFAULT_PARAMS} --encoder resnet34 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --encoder resnet34 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --encoder resnet50 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# rgb -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet18 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet18 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34 (=baseline)
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet34 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet34 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --modality rgb --encoder resnet50 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# ------------------------------------------------------------------------------
# Cityscapes (1024x2048)
# ------------------------------------------------------------------------------
DATASET="--dataset cityscapes --dataset_dir ./datasets/cityscapes"
RESULTS_FILE='./cityscapes_timings.csv'

# rgbd -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --valid_full_res --encoder resnet18 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --valid_full_res --encoder resnet18 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34 (=baseline)
PARAMS="${DEFAULT_PARAMS} --valid_full_res --encoder resnet34 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --valid_full_res --encoder resnet34 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --valid_full_res --encoder resnet50 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# rgb -------------------------------------------------------------------------
# resnet 18
PARAMS="${DEFAULT_PARAMS} --valid_full_res --modality rgb --encoder resnet18 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 18 bb
PARAMS="${DEFAULT_PARAMS} --valid_full_res --modality rgb --encoder resnet18 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34 (=baseline)
PARAMS="${DEFAULT_PARAMS} --valid_full_res --modality rgb --encoder resnet34 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 34
PARAMS="${DEFAULT_PARAMS} --valid_full_res --modality rgb --encoder resnet34 --encoder_block BasicBlock --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE

# resnet 50
PARAMS="${DEFAULT_PARAMS} --valid_full_res --modality rgb --encoder resnet50 --height 512 --width 1024 --context_module appm-1-2-4-8 --raw_depth"
echo -n "${PARAMS}," >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_PYTORCH $PARAMS | eval $SED_PYTORCH | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT32 $PARAMS | eval $SED_TRT32 | xargs echo -n >> $RESULTS_FILE
python3 inference_time_whole_model.py ${DATASET} $TIME_TRT16 $PARAMS | eval $SED_TRT16 >> $RESULTS_FILE
