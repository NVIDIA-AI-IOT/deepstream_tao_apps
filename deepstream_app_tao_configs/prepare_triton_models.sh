#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin


mkdir -p ./triton/dashcamnet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_onnx_v1.0.5/zip \
-O dashcamnet_pruned_onnx_v1.0.5.zip && unzip -o dashcamnet_pruned_onnx_v1.0.5.zip -d ./triton/dashcamnet
rm dashcamnet_pruned_onnx_v1.0.5.zip

mkdir -p ./triton/dashcamnet/1
trtexec --onnx=./triton/dashcamnet/resnet18_dashcamnet_pruned.onnx --int8 --calib=./triton/dashcamnet/resnet18_dashcamnet_pruned.txt \
 --saveEngine=./triton/dashcamnet/1/resnet18_dashcamnet_pruned.onnx_b1_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 \
 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960
cp triton/dashcamnet_config.pbtxt ./triton/dashcamnet/config.pbtxt

mkdir -p ./triton/vehiclemakenet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehiclemakenet/versions/pruned_onnx_v1.1.0/zip \
-O vehiclemakenet_pruned_onnx_v1.1.0.zip
unzip -o vehiclemakenet_pruned_onnx_v1.1.0.zip -d ./triton/vehiclemakenet/
rm vehiclemakenet_pruned_onnx_v1.1.0.zip

mkdir -p ./triton/vehiclemakenet/1
trtexec --onnx=./triton/vehiclemakenet/resnet18_pruned.onnx --int8 --calib=./triton/vehiclemakenet/resnet18_pruned_int8.txt \
        --saveEngine=./triton/vehiclemakenet/1/resnet18_pruned.onnx_b4_gpu0_int8.engine --minShapes="input_1:0":1x3x224x224 \
        --optShapes="input_1:0":4x3x224x224 --maxShapes="input_1:0":4x3x224x224
cp triton/vehiclemakenet_config.pbtxt ./triton/vehiclemakenet/config.pbtxt

mkdir -p ./triton/vehicletypenet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/vehicletypenet/versions/pruned_onnx_v1.1.0/zip \
-O vehicletypenet_pruned_onnx_v1.1.0.zip
unzip -o vehicletypenet_pruned_onnx_v1.1.0.zip -d ./triton/vehicletypenet
rm -r vehicletypenet_pruned_onnx_v1.1.0.zip

mkdir -p ./triton/vehicletypenet/1
trtexec --onnx=./triton/vehicletypenet/resnet18_pruned.onnx --int8 --calib=./triton/vehicletypenet/resnet18_pruned_int8.txt \
        --saveEngine=./triton/vehicletypenet/1/resnet18_pruned.onnx_b4_gpu0_int8.engine --minShapes="input_1:0":1x3x224x224 \
        --optShapes="input_1:0":4x3x224x224 --maxShapes="input_1:0":4x3x224x224
cp triton/vehicletypenet_config.pbtxt ./triton/vehicletypenet/config.pbtxt

mkdir -p ./triton/peopleNet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.4/zip \
-O peoplenet_pruned_quantized_decrypted_v2.3.4.zip
unzip -o peoplenet_pruned_quantized_decrypted_v2.3.4 -d ./triton/peopleNet/
rm peoplenet_pruned_quantized_decrypted_v2.3.4.zip

mkdir -p ./triton/peopleNet/1
trtexec --onnx=./triton/peopleNet/resnet34_peoplenet_int8.onnx --int8 --calib=./triton/peopleNet/resnet34_peoplenet_int8.txt --saveEngine=./triton/peopleNet/1/resnet34_peoplenet_int8.onnx_b1_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960&
cp triton/peopleNet_config.pbtxt ./triton/peopleNet/config.pbtxt

mkdir -p ./triton/trafficcamnet/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_onnx_v1.0.4/zip -O trafficcamnet_pruned_onnx_v1.0.4.zip && \
unzip -o trafficcamnet_pruned_onnx_v1.0.4.zip -d ./triton/trafficcamnet/ && \
rm trafficcamnet_pruned_onnx_v1.0.4.zip

mkdir -p ./triton/trafficcamnet/1
trtexec --onnx=./triton/trafficcamnet/resnet18_trafficcamnet_pruned.onnx --int8 --calib=./triton/trafficcamnet/resnet18_trafficcamnet_pruned_int8.txt \
 --saveEngine=./triton/trafficcamnet/1/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 \
 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960
cp triton/trafficcamnet_config.pbtxt ./triton/trafficcamnet/config.pbtxt

#mkdir -p ./triton/multi_task/1
#trtexec --onnx=./triton/multi_task/multitask_cls_resnet10_epoch_010.onnx --fp16 --saveEngine=./triton/multi_task/1/multitask_cls_resnet10_epoch_010.onnx_b1_gpu0_fp16.engine --minShapes=input_1:1x3x80x60 --optShapes=input_1:1x3x80x60 --maxShapes=input_1:1x3x80x60
#cp triton/multi_task_config.pbtxt ./triton/multi_task/config.pbtxt