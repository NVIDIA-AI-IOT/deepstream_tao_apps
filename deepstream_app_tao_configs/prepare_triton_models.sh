#!/bin/bash
################################################################################
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin

mkdir -p ./triton/vehiclemakenet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
-O ./triton/vehiclemakenet/resnet18_pruned.onnx && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
-O ./triton/vehiclemakenet/labels.txt

mkdir -p ./triton/vehiclemakenet/1
trtexec --onnx=./triton/vehiclemakenet/resnet18_pruned.onnx --fp16 \
        --saveEngine=./triton/vehiclemakenet/1/resnet18_pruned.onnx_b4_gpu0_fp16.engine --minShapes="input_1:0":1x3x224x224 \
        --optShapes="input_1:0":4x3x224x224 --maxShapes="input_1:0":4x3x224x224
cp triton/vehiclemakenet_config.pbtxt ./triton/vehiclemakenet/config.pbtxt

mkdir -p ./triton/vehicletypenet/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
-O ./triton/vehicletypenet/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
-O ./triton/vehicletypenet/labels.txt


mkdir -p ./triton/vehicletypenet/1
trtexec --onnx=./triton/vehicletypenet/resnet18_pruned.onnx --fp16 \
        --saveEngine=./triton/vehicletypenet/1/resnet18_pruned.onnx_b4_gpu0_fp16.engine --minShapes="input_1:0":1x3x224x224 \
        --optShapes="input_1:0":4x3x224x224 --maxShapes="input_1:0":4x3x224x224
cp triton/vehicletypenet_config.pbtxt ./triton/vehicletypenet/config.pbtxt

mkdir -p ./triton/peopleNet/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=labels.txt' \
-O ./triton/peopleNet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx' \
-O ./triton/peopleNet/resnet34_peoplenet_int8.onnx


mkdir -p ./triton/peopleNet/1
trtexec --onnx=./triton/peopleNet/resnet34_peoplenet_int8.onnx --fp16 --saveEngine=./triton/peopleNet/1/resnet34_peoplenet_int8.onnx_b1_gpu0_fp16.engine --minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960&
cp triton/peopleNet_config.pbtxt ./triton/peopleNet/config.pbtxt

mkdir -p ./triton/trafficcamnet/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=labels.txt' \
-O ./triton/trafficcamnet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned.onnx' \
-O ./triton/trafficcamnet/resnet18_trafficcamnet_pruned.onnx


mkdir -p ./triton/trafficcamnet/1
trtexec --onnx=./triton/trafficcamnet/resnet18_trafficcamnet_pruned.onnx --fp16 \
 --saveEngine=./triton/trafficcamnet/1/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine --minShapes="input_1:0":1x3x544x960 \
 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960
cp triton/trafficcamnet_config.pbtxt ./triton/trafficcamnet/config.pbtxt

mkdir -p ./triton/peoplenet_transformer_v2/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O ./triton/peoplenet_transformer_v2/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx' -O ./triton/peoplenet_transformer_v2/dino_fan_small_astro_delta.onnx

mkdir -p ./triton/peoplenet_transformer_v2/1
trtexec --onnx=./triton/peoplenet_transformer_v2/dino_fan_small_astro_delta.onnx --fp16 \
 --saveEngine=./triton/peoplenet_transformer_v2/1/dino_fan_small_astro_delta.onnx_b1_gpu0_fp16.engine \
 --minShapes="inputs":1x3x544x960 --optShapes="inputs":1x3x544x960 --maxShapes="inputs":1x3x544x960&
cp triton/peoplenet_transformer_v2_config.pbtxt ./triton/peoplenet_transformer_v2/config.pbtxt

