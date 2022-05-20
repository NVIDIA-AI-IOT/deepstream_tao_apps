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

# For Faster-RCNN / YoloV3 / YoloV4 /SSD / DSSD / RetinaNet/ UNET/:
# wget https://nvidia.box.com/shared/static/em2dh1h4isjhfu7qf0hh6ggzbusdg129 -O models.zip

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
wget https://nvidia.box.com/shared/static/em2dh1h4isjhfu7qf0hh6ggzbusdg129 -O models.zip
unzip models.zip
rm models.zip

echo "==================================================================="
echo "begin download models for peopleSegNet "
echo "==================================================================="
mkdir -p models/peopleSegNet/V2
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v2.0.1/zip -O peoplesegnet_deployable_v2.0.1.zip \
-O peoplesegnet_deployable_v2.0.1.zip
unzip peoplesegnet_deployable_v2.0.1.zip -d models/peopleSegNet/V2
rm peoplesegnet_deployable_v2.0.1.zip

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesegnet/versions/deployable_v1.0/zip \
-O peoplesegnet_deployable_v1.0.zip
unzip peoplesegnet_deployable_v1.0.zip -d models/peopleSegNet/
rm peoplesegnet_deployable_v1.0.zip

echo "==================================================================="
echo "begin download models for peopleSemSegNet "
echo "==================================================================="
mkdir -p models/peopleSemSegNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_v1.0.1/zip -O peoplesemsegnet_deployable_v1.0.1.zip \
-O peoplesemsegnet_deployable_v1.0.1.zip
unzip peoplesemsegnet_deployable_v1.0.1.zip -d models/peopleSemSegNet/
rm peoplesemsegnet_deployable_v1.0.1.zip

echo "==================================================================="
echo "begin downloading facial landmarks model "
echo "==================================================================="
mkdir -p ./models/faciallandmark
cd ./models/faciallandmark
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/model.etlt -O faciallandmarks.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/int8_calibration.txt -O fpenet_cal.txt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0/files/model.etlt -O facenet.etlt
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0/files/int8_calibration.txt -O int8_calibration.txt

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
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/gesturenet/versions/deployable_v2.0.1/files/model.etlt -O gesture.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/gesturenet/versions/deployable_v2.0.1/files/int8_calibration.txt -O int8_calibration.txt

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
echo "Download models successfully "
echo "==================================================================="
