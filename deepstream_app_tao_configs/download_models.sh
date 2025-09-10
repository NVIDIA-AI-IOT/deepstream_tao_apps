#!/bin/sh
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

# Check following part for how to download the TAO models:
# https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deepstream_tao_integration.html

echo "==================================================================="
echo "begin download models for vehiclemakenet / vehicletypenet"
echo " / trafficcamnet"
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/vehiclemakenet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
-O ../../models/tao_pretrained_models/vehiclemakenet/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned_int8.txt' \
-O ../../models/tao_pretrained_models/vehiclemakenet/resnet18_pruned_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehiclemakenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/vehiclemakenet/labels.txt
mkdir -p ../../models/tao_pretrained_models/vehicletypenet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned.onnx' \
-O ../../models/tao_pretrained_models/vehicletypenet/resnet18_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/vehicletypenet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/vehicletypenet/pruned_onnx_v1.1.0/files?redirect=true&path=resnet18_pruned_int8.txt' \
-O ../../models/tao_pretrained_models/vehicletypenet/resnet18_pruned_int8.txt
mkdir -p ../../models/tao_pretrained_models/trafficcamnet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/trafficcamnet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned.onnx' \
-O ../../models/tao_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned_int8.txt' \
-O ../../models/tao_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned_int8.txt

echo "==================================================================="
echo "begin download models for peopleNet "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peopleNet/
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=labels.txt' \
-O ../../models/tao_pretrained_models/peopleNet/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx' \
-O ../../models/tao_pretrained_models/peopleNet/resnet34_peoplenet_int8.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.txt' \
-O ../../models/tao_pretrained_models/peopleNet/resnet34_peoplenet_int8.txt

echo "==================================================================="
echo "begin download models for peoplenet_transformer_v2 "
echo "==================================================================="
mkdir -p ../../models/tao_pretrained_models/peoplenet_transformer_v2/
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O ../../models/tao_pretrained_models/peoplenet_transformer_v2/labels.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer_v2/deployable_v1.0/files?redirect=true&path=dino_fan_small_astro_delta.onnx' -O ../../models/tao_pretrained_models/peoplenet_transformer_v2/dino_fan_small_astro_delta.onnx

echo "==================================================================="
echo "Download models successfully "
echo "==================================================================="
