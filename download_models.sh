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
# wget https://nvidia.box.com/shared/static/hzrhk33vijf31w9nxb9c93gctu1w0spd -O models.zip

# For peopleSemSegNet:
# wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0/zip \
# -O peoplesemsegnet_deployable_v1.0.zip

# For peopleSegNet V2:
# wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0/zip \
# -O peoplesegnet_deployable_v2.0.zip

# For old peopleSegNet:
# wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v1.0/zip \
# -O peoplesegnet_deployable_v1.0.zip

echo "==================================================================="
echo "begin download models for Faster-RCNN / YoloV3 / YoloV4 /SSD / DSSD / RetinaNet/ UNET/"
echo "==================================================================="
wget https://nvidia.box.com/shared/static/hzrhk33vijf31w9nxb9c93gctu1w0spd -O models.zip
unzip -o models.zip
rm models.zip

echo "==================================================================="
echo "begin download models for peopleSegNet "
echo "==================================================================="
mkdir -p models/peopleSegNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.2/zip \
-O peoplesegnet_deployable_v2.0.2.zip
unzip -o peoplesegnet_deployable_v2.0.2.zip -d models/peopleSegNet
rm peoplesegnet_deployable_v2.0.2.zip

echo "==================================================================="
echo "begin download models for peopleSemSegNet "
echo "==================================================================="
mkdir -p models/peopleSemSegNet_vanilla
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_quantized_vanilla_unet_onnx_v2.0/zip \
-O deployable_quantized_vanilla_unet_onnx_v2.0.zip
unzip -o deployable_quantized_vanilla_unet_onnx_v2.0.zip -d models/peopleSemSegNet_vanilla
rm deployable_quantized_vanilla_unet_onnx_v2.0.zip


mkdir -p models/peopleSemSegNet_shuffle
cd ./models/peopleSemSegNet_shuffle
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0/files?redirect=true&path=labels.txt' -O labels.txt && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0/files?redirect=true&path=peoplesemsegnet_shuffleseg.onnx' -O peoplesemsegnet_shuffleseg.onnx && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0/files?redirect=true&path=peoplesemsegnet_shuffleseg_cache.txt' -O peoplesemsegnet_shuffleseg_cache.txt


echo "==================================================================="
echo "begin downloading facial landmarks model "
echo "==================================================================="
cd -
mkdir -p ./models/faciallandmark
cd ./models/faciallandmark
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/model.etlt -O faciallandmark.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/int8_calibration.txt -O fpenet_cal.txt

echo "==================================================================="
echo "begin downloading facenet model "
echo "==================================================================="
cd -
mkdir -p ./models/facenet
cd ./models/facenet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/model.etlt -O facenet.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/int8_calibration.txt -O facenet_cal.txt

echo "==================================================================="
echo "begin downloading emotionNet model "
echo "==================================================================="
cd -
mkdir -p ./models/emotion
cd ./models/emotion
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/emotionnet/versions/deployable_v1.0/files/model.etlt -O emotion.etlt

echo "==================================================================="
echo "begin downloading GazeNet model "
echo "==================================================================="
cd -
mkdir -p ./models/gazenet
cd ./models/gazenet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/gazenet/versions/deployable_v1.0/files/model.etlt -O gazenet_facegrid.etlt

echo "==================================================================="
echo "begin downloading HeartRateNet model "
echo "==================================================================="
cd -
mkdir -p ./models/heartrate
cd ./models/heartrate
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/heartratenet/versions/deployable_v2.0/files/model.etlt -O heartrate.etlt

echo "==================================================================="
echo "begin downloading GestureNet model "
echo "==================================================================="
cd -
mkdir -p ./models/gesture
cd ./models/gesture
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/gesturenet/versions/deployable_v2.0.2/files/model.etlt -O gesture.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/gesturenet/versions/deployable_v2.0.2/files/int8_calibration.txt -O int8_calibration.txt

echo "==================================================================="
echo "begin downloading BodyPose2d model "
echo "==================================================================="
cd -
mkdir -p ./models/bodypose2d
cd ./models/bodypose2d
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/bodyposenet/versions/deployable_v1.0.1/zip -O bodyposenet_deployable_v1.0.1.zip
unzip -o bodyposenet_deployable_v1.0.1.zip
rm bodyposenet_deployable_v1.0.1.zip

echo "==================================================================="
echo "begin downloading CitySemSegFormer model "
echo "==================================================================="
cd -
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

echo "==================================================================="
echo "begin downloading Retail Object Detection vdeployable_100 model "
echo "==================================================================="
cd -
mkdir -p ./models/retail_object_detection_100
cd ./models/retail_object_detection_100
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_100_onnx_v1.0/files?redirect=true&path=retail_detector_100.onnx' -O retail_detector_100.onnx && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_100_onnx_v1.0/files?redirect=true&path=retail_detector_100_int8.txt' -O retail_detector_100_int8.txt && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_100_onnx_v1.0/files?redirect=true&path=retail_detector_100_labels.txt' -O class_map.txt
echo "==================================================================================="
echo "begin downloading Retail Object Detection EfficientdetDet vdeployable_binary model "
echo "==================================================================================="
cd -
mkdir -p ./models/retail_object_detection_binary_effdet
cd ./models/retail_object_detection_binary_effdet && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_binary_onnx_v1.0/files?redirect=true&path=retail_detector_binary.onnx' -O retail_detector_binary.onnx  && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_binary_onnx_v1.0/files?redirect=true&path=retail_detector_binary_int8.txt' -O retail_detector_binary_int8.txt  && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_binary_onnx_v1.0/files?redirect=true&path=retail_detector_binary_labels.txt' -O class_map.txt

echo "========================================================================"
echo "begin downloading Retail Object Detection DINO vdeployable_binary model "
echo "========================================================================"
cd -
mkdir -p ./models/retail_object_detection_binary_dino
cd ./models/retail_object_detection_binary_dino
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_binary_v2.0/files/class_map.txt' -O class_map.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_detection/deployable_binary_v2.1.2/files?redirect=true&path=retail_object_detection_binary_v2.1.2.opset17.onnx' -O retail_object_detection_dino_binary.onnx

echo "========================================================================"
echo "begin downloading Retail Object Detection DINO vdeployable_meta model "
echo "========================================================================"
cd -
mkdir -p ./models/retail_object_detection_meta
cd ./models/retail_object_detection_meta
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_meta_v2.0/files/class_map.txt' -O class_map.txt
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_meta_v2.0/files/retail_object_detection_dino_meta.onnx' -O retail_object_detection_dino_meta.onnx

echo "==================================================================="
echo "begin downloading Retail Object Recognition model "
echo "==================================================================="
cd -
mkdir -p ./models/retail_object_recognition
cd ./models/retail_object_recognition
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/retail_object_recognition/deployable_onnx_v1.0/files?redirect=true&path=retail_object_recognition.onnx' -O retail_object_recognition.onnx
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_recognition/versions/deployable_v2.0/files/recognitionv2_name_list.txt -O retail_object_recognition_labels.txt

echo "==================================================================="
echo "begin downloading PeopleNet model "
echo "==================================================================="
cd -
mkdir -p ./models/peoplenet
cd ./models/peoplenet
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O resnet34_peoplenet_int8.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.txt' -O resnet34_peoplenet_int8.txt
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=labels.txt' -O labels.txt

echo "==================================================================="
echo "begin downloading BodyPose3DNet model "
echo "==================================================================="
cd -
mkdir -p ./models/bodypose3dnet
cd ./models/bodypose3dnet
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/bodypose3dnet/versions/deployable_accuracy_v1.0/files/bodypose3dnet_accuracy.etlt

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
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_v2.3/files?redirect=true&path=ocdnet_fan_tiny_2x_icdar_pruned.onnx' -O ocdnet.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=ocrnet-vit.onnx' -O ocrnet.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=character_list' -O character_list

echo "==================================================================="
echo "begin downloading tracker model "
echo "==================================================================="
cd -
mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker
cd /opt/nvidia/deepstream/deepstream/samples/models/Tracker
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt

echo "==================================================================="
echo "Download models successfully "
echo "==================================================================="
