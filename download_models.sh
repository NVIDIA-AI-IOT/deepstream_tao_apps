#!/bin/sh
################################################################################
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

# Check following part for how to download the TLT 3.0 models:

# For Faster-RCNN / YoloV3 / YoloV4 /SSD / DSSD / RetinaNet/ EfficientDet0/ UNET/:
# wget https://nvidia.box.com/shared/static/w0xxle5b3mjiv20wrq5q37v8u7b3u5tn -O models.zip

# For peopleSemSegNet:
# wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/zip \
# -O peoplesemsegnet_deployable_v1.0.zip

set -e

echo "==================================================================="
echo "begin download models for Faster-RCNN / YoloV3 / YoloV4 /SSD / DSSD / RetinaNet/ UNET/"
echo "==================================================================="
wget https://nvidia.box.com/shared/static/w0xxle5b3mjiv20wrq5q37v8u7b3u5tn -O models.zip
unzip -o models.zip
rm models.zip

echo "==================================================================="
echo "begin download models for Mask2Former "
echo "==================================================================="
mkdir -p models/mask2former
cd ./models/mask2former
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/mask2former/mask2former_swint_deployable_v1.0/files?redirect=true&path=mask2former_swint.onnx' \
-O mask2former.onnx

echo "==================================================================="
echo "begin download models for peopleSemSegNet "
echo "==================================================================="
cd -
mkdir -p models/peopleSemSegNet_vanilla
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_vanilla_unet_onnx_v2.0.2/zip \
-O deployable_vanilla_unet_onnx_v2.0.2.zip
unzip -o deployable_vanilla_unet_onnx_v2.0.2.zip -d models/peopleSemSegNet_vanilla
rm deployable_vanilla_unet_onnx_v2.0.2.zip

mkdir -p models/peopleSemSegNet_shuffle
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_onnx_v1.0.1/zip \
-O peoplesemsegnet_deployable_shuffleseg_unet_onnx_v1.0.1.zip
unzip -o peoplesemsegnet_deployable_shuffleseg_unet_onnx_v1.0.1.zip -d models/peopleSemSegNet_shuffle
rm peoplesemsegnet_deployable_shuffleseg_unet_onnx_v1.0.1.zip

echo "==================================================================="
echo "begin downloading CitySemSegFormer model "
echo "==================================================================="
mkdir -p ./models/citysemsegformer
cd ./models/citysemsegformer
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/citysemsegformer/deployable_onnx_v1.0/files?redirect=true&path=citysemsegformer.onnx' -O citysemsegformer.onnx && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/citysemsegformer/deployable_onnx_v1.0/files?redirect=true&path=labels.txt' -O labels.txt

echo "==================================================================="
echo "begin downloading PeopleNet Transformer model "
echo "==================================================================="
cd -
mkdir -p ./models/peoplenet_transformer
cd ./models/peoplenet_transformer
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer/deployable_v1.1/files?redirect=true&path=resnet50_peoplenet_transformer_op17.onnx' \
 -O resnet50_peoplenet_transformer_op17.onnx
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O labels.txt

echo "==================================================================="
echo "begin downloading Re-Identification model "
echo "==================================================================="
cd -
mkdir -p ./models/reidentificationnet
cd ./models/reidentificationnet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/deployable_v1.2/files?redirect=true&path=resnet50_market1501_aicity156.onnx' -O resnet50_market1501_aicity156.onnx

echo "========================================================================"
echo "begin downloading Retail Object Detection DINO vdeployable_binary model "
echo "========================================================================"
cd -
mkdir -p ./models/retail_object_detection_binary_dino
cd ./models/retail_object_detection_binary_dino
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_binary_v2.0/files/class_map.txt' -O class_map.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_retail_object_detection_binary_v2.2.2.3/files?redirect=true&path=retail_object_detection_binary_v2.2.2.3.onnx'  -O retail_object_detection_dino_binary.onnx

echo "==================================================================="
echo "begin downloading Retail Object Recognition model "
echo "==================================================================="
cd -
mkdir -p ./models/retail_object_recognition
cd ./models/retail_object_recognition
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_recognition/deployable_v2.0/files?redirect=true&path=FANDualHead_Base_NVIN_op16.onnx' -O retail_object_recognition.onnx
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_recognition/versions/deployable_v2.0/files/recognitionv2_name_list.txt -O retail_object_recognition_labels.txt

echo "==================================================================="
echo "begin downloading PeopleNet model "
echo "==================================================================="
cd -
mkdir -p ./models/peoplenet
cd ./models/peoplenet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O resnet34_peoplenet_int8.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.txt' -O resnet34_peoplenet_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=labels.txt' -O labels.txt

echo "==================================================================="
echo "begin downloading BodyPose3DNet model "
echo "==================================================================="
cd -
mkdir -p ./models/bodypose3dnet
cd ./models/bodypose3dnet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/bodypose3dnet/deployable_accuracy_onnx_1.0/files?redirect=true&path=bodypose3dnet_accuracy.onnx' -O bodypose3dnet_accuracy.onnx

echo "==================================================================="
echo "begin downloading poseclassificationnet model "
echo "==================================================================="
cd -
mkdir -p ./models/poseclassificationnet
cd ./models/poseclassificationnet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/poseclassificationnet/deployable_onnx_v1.0/files?redirect=true&path=st-gcn_3dbp_nvidia.onnx' -O st-gcn_3dbp_nvidia.onnx

echo "==================================================================="
echo "begin downloading nvocdr model "
echo "==================================================================="
cd -
mkdir -p ./models/nvocdr
cd ./models/nvocdr
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_onnx_v2.4/files?redirect=true&path=ocdnet_fan_tiny_2x_icdar_pruned.onnx' -O ocdnet.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_onnx_v2.4/files?redirect=true&path=ocdnet_fan_tiny_2x_icdar_pruned.cal' -O ocdnet.cal
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.1.1/files?redirect=true&path=ocrnet-vit-pcb.onnx' -O ocrnet.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=character_list' -O character_list

echo "==================================================================="
echo "begin downloading CAR Plate models "
echo "==================================================================="
cd -
mkdir -p ./models/trafficcamnet
cd ./models/trafficcamnet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned.onnx' -O resnet18_trafficcamnet_pruned.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned_int8.txt' -O resnet18_trafficcamnet_pruned_int8.txt

cd -
mkdir -p ./models/LPD_us
cd ./models/LPD_us
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.3.1/files?redirect=true&path=LPDNet_usa_pruned_tao5.onnx' -O LPDNet_usa_pruned_tao5.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.3.1/files?redirect=true&path=usa_cal_10.1.0.bin' -O usa_cal_10.1.0.bin
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/usa_lpd_label.txt
cd -
mkdir -p ./models/LPD_ch
cd ./models/LPD_ch
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.3.1/files?redirect=true&path=LPDNet_CCPD_pruned_tao5.onnx' -O LPDNet_CCPD_pruned_tao5.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.3.1/files?redirect=true&path=ccpd_cal_10.1.0.bin' -O ccpd_cal_10.1.0.bin
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/ccpd_label.txt
cd -

mkdir -p ./models/LPR_us
cd ./models/LPR_us
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lprnet/deployable_onnx_v1.1/files?redirect=true&path=us_lprnet_baseline18_deployable.onnx' -O us_lprnet_baseline18_deployable.onnx
touch labels_us.txt
cd -
mkdir -p ./models/LPR_ch
cd ./models/LPR_ch
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lprnet/deployable_onnx_v1.1/files?redirect=true&path=ch_lprnet_baseline18_deployable.onnx' -O ch_lprnet_baseline18_deployable.onnx
touch labels_ch.txt

echo "==================================================================="
echo "begin downloading tracker model "
echo "==================================================================="
mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker
cd /opt/nvidia/deepstream/deepstream/samples/models/Tracker
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt
# cp ./models/reidentificationnet/resnet50_market1501_aicity156.onnx /opt/nvidia/deepstream/deepstream/samples/models/Tracker

echo "==================================================================="
echo "Download models successfully "
echo "==================================================================="
