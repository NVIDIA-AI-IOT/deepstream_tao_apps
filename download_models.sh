#!/bin/sh
################################################################################
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
# wget https://nvidia.box.com/shared/static/taqr2y52go17x1ymaekmg6dh8z6d43wr -O models.zip

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
wget https://nvidia.box.com/shared/static/taqr2y52go17x1ymaekmg6dh8z6d43wr -O models.zip
unzip models.zip
rm models.zip

echo "==================================================================="
echo "begin download models for peopleSegNet "
echo "==================================================================="
mkdir -p models/peopleSegNet/V2
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.2/zip \
-O peoplesegnet_deployable_v2.0.2.zip
unzip peoplesegnet_deployable_v2.0.2.zip -d models/peopleSegNet/V2
rm peoplesegnet_deployable_v2.0.2.zip

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v1.0/zip \
-O peoplesegnet_deployable_v1.0.zip
unzip peoplesegnet_deployable_v1.0.zip -d models/peopleSegNet/
rm peoplesegnet_deployable_v1.0.zip

echo "==================================================================="
echo "begin download models for peopleSemSegNet "
echo "==================================================================="
mkdir -p models/peopleSemSegNet/vanilla
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_quantized_vanilla_unet_v2.0/zip \
-O peoplesemsegnet_deployable_quantized_vanilla_unet_v2.0.zip
unzip peoplesemsegnet_deployable_quantized_vanilla_unet_v2.0.zip -d models/peopleSemSegNet/vanilla
rm peoplesemsegnet_deployable_quantized_vanilla_unet_v2.0.zip
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_vanilla_unet_v2.0.1/files/peoplesemsegnet_vanilla_unet_dynamic_etlt_fp32.etlt \
-O models/peopleSemSegNet/vanilla/peoplesemsegnet_vanilla_unet_dynamic_etlt_fp32.etlt

mkdir -p models/peopleSemSegNet/shuffle
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/zip \
-O deployable_shuffleseg_unet_v1.0.zip
unzip deployable_shuffleseg_unet_v1.0.zip -d models/peopleSemSegNet/shuffle
rm deployable_shuffleseg_unet_v1.0.zip

echo "==================================================================="
echo "begin downloading facial landmarks model "
echo "==================================================================="
mkdir -p ./models/faciallandmark
cd ./models/faciallandmark
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/model.etlt -O faciallandmarks.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/int8_calibration.txt -O faciallandmarks_int8_calibration.txt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/model.etlt -O facenet.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/int8_calibration.txt -O facenet_int8_calibration.txt

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
unzip bodyposenet_deployable_v1.0.1.zip
rm bodyposenet_deployable_v1.0.1.zip

echo "==================================================================="
echo "begin downloading CitySemSegFormer model "
echo "==================================================================="
cd -
mkdir -p ./models/citysemsegformer_vdeployable_v1.0
cd ./models/citysemsegformer_vdeployable_v1.0
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/citysemsegformer/versions/deployable_v1.0/files/citysemsegformer.etlt -O citysemsegformer.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/citysemsegformer/versions/deployable_v1.0/files/labels.txt -O labels.txt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/citysemsegformer/versions/deployable_v1.0/files/nvinfer_config.txt -O nvinfer_config.txt

echo "==================================================================="
echo "begin downloading PeopleNet Transformer model "
echo "==================================================================="
cd -
mkdir -p ./models/peoplenet_transformer_vdeployable_v1.0
cd ./models/peoplenet_transformer_vdeployable_v1.0
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/resnet50_peoplenet_transformer.etlt -O resnet50_peoplenet_transformer.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O labels.txt

echo "==================================================================="
echo "begin downloading Re-Identification model "
echo "==================================================================="
cd -
mkdir -p ./models/reidentificationnet_vdeployable_v1.0
cd ./models/reidentificationnet_vdeployable_v1.0
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt -O resnet50_market1501.etlt

echo "==================================================================="
echo "begin downloading Retail Object Detection vdeployable_100 model "
echo "==================================================================="
cd -
mkdir -p ./models/retail_object_detection_vdeployable_100_v1.0
cd ./models/retail_object_detection_vdeployable_100_v1.0
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_100_v1.0/files/retail_detector_100.etlt  -O retail_detector_100.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_100_v1.0/files/retail_detector_100_int8.txt -O retail_detector_100_int8.txt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_100_v1.0/files/retail_detector_100_labels.txt -O retail_detector_100_labels.txt

echo "==================================================================="
echo "begin downloading Retail Object Detection vdeployable_binary model "
echo "==================================================================="
cd -
mkdir -p ./models/retail_object_detection_vdeployable_binary_v1.0
cd ./models/retail_object_detection_vdeployable_binary_v1.0
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_binary_v1.0/files/retail_detector_binary.etlt -O retail_detector_binary.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_binary_v1.0/files/retail_detector_binary_int8.txt -O retail_detector_binary_int8.txt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_detection/versions/deployable_binary_v1.0/files/retail_detector_binary_labels.txt -O retail_detector_binary_labels.txt

echo "==================================================================="
echo "begin downloading Retail Object Recognition model "
echo "==================================================================="
cd -
mkdir -p ./models/retail_object_recognition_vdeployable_v1.0
cd ./models/retail_object_recognition_vdeployable_v1.0
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/retail_object_recognition/versions/deployable_v1.0/files/retail_object_recognition.etlt -O retail_object_recognition.etlt

echo "==================================================================="
echo "Download models successfully "
echo "==================================================================="
