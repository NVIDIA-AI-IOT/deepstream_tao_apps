#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin

#detection
#efficientdet
echo "Building Model efficientdet..."
mkdir  -p  models/efficientdet/1
if [ ! ${IS_JETSON_PLATFORM} ]; then
  trtexec --onnx=./models/efficientdet/d0_avlp_544_960.onnx --int8 --calib=./models/efficientdet/d0_avlp_544_960_x86.cal --saveEngine=./models/efficientdet/1/d0_avlp_544_960.onnx_b1_gpu0_int8.engine&
else
  trtexec --onnx=./models/efficientdet/d0_avlp_544_960.onnx --int8 --calib=./models/efficientdet/d0_avlp_544_960_jetson.cal --saveEngine=./models/efficientdet/1/d0_avlp_544_960.onnx_b1_gpu0_int8.engine&
fi

#yolov5
#echo "Building Model yolov5..."
#mkdir  -p  models/yolov5/1
#/usr/src/tensorrt/bin/trtexec --fp16  --onnx=./models/yolov5/yolov5s.onnx  \
# --saveEngine=./models/yolov5/1/yolov5s.onnx_b4_gpu0_fp16.engine --minShapes=images:1x3x672x672 \
# --optShapes=images:4x3x672x672 --maxShapes=images:4x3x672x672 --shapes=images:4x3x672x672

#for classifcaiton
#multi_task
#mkdir  -p  models/multi_task/1
#trtexec --onnx=./models/multi_task/multi_task.onnx  --saveEngine=./models/multi_task/1/multi_task.onnx_b4_gpu0_fp16.engine \
# --minShapes=input_1:1x3x80x60 --optShapes=input_1:4x3x80x60 --maxShapes=input_1:4x3x80x60&

#for instance segmentation
#for mask2former
echo "Building Model mask2former..."
mkdir  -p  models/mask2former/1
trtexec --onnx=./models/mask2former/mask2former.onnx --fp16 --saveEngine=./models/mask2former/1/mask2former.onnx_b1_gpu0_fp16.engine \
 --minShapes=inputs:1x3x800x800 --optShapes=inputs:1x3x800x800 --maxShapes=inputs:1x3x800x800&

#for segmentation
#peopleSemSegNet
echo "Building Model peopleSemSegNet..."
mkdir  -p  models/peopleSemSegNet_vanilla/1
trtexec --onnx=./models/peopleSemSegNet_vanilla/peoplesemsegnet_vanilla_unet_dynamic_etlt_fp32.onnx --int8 \
 --calib=./models/peopleSemSegNet_vanilla/peoplesemsegnet_vanilla_unet_int8.txt --saveEngine=./models/peopleSemSegNet_vanilla/1/peoplesemsegnet_vanilla_unet_dynamic_etlt_fp32.onnx_b4_gpu0_int8.engine \
 --minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":4x3x544x960 --maxShapes="input_1:0":4x3x544x960&

mkdir  -p  models/peopleSemSegNet_shuffle/1
trtexec --onnx=./models/peopleSemSegNet_shuffle/peoplesemsegnet_shuffleseg.onnx --int8 \
 --calib=./models/peopleSemSegNet_shuffle/peoplesemsegnet_shuffleseg_int8.txt --saveEngine=./models/peopleSemSegNet_shuffle/1/peoplesemsegnet_shuffleseg.onnx_b4_gpu0_int8.engine \
 --minShapes="input_2:0":1x3x544x960 --optShapes="input_2:0":4x3x544x960 --maxShapes="input_2:0":4x3x544x960&

#unet
echo "Building Model unet..."
mkdir  -p  models/unet/1
if [ ! ${IS_JETSON_PLATFORM} ]; then
  trtexec --onnx=./models/unet/unet_resnet18.onnx --int8 --calib=./models/unet/unet_cal_x86.bin --saveEngine=./models/unet/1/unet_resnet18.onnx_b4_gpu0_int8.engine --minShapes="input_1:0":1x3x320x320 --optShapes="input_1:0":2x3x320x320 --maxShapes="input_1:0":4x3x320x320&
else
  trtexec --onnx=./models/unet/unet_resnet18.onnx --int8 --calib=./models/unet/unet_cal_jetson.bin --saveEngine=./models/unet/1/unet_resnet18.onnx_b4_gpu0_int8.engine --minShapes="input_1:0":1x3x320x320 --optShapes="input_1:0":2x3x320x320 --maxShapes="input_1:0":4x3x320x320&
fi
#citysemsegformer
echo "Building Model citysemsegformer..."
mkdir  -p  models/citysemsegformer/1  && \
trtexec --onnx=./models/citysemsegformer/citysemsegformer.onnx --fp16 \
 --saveEngine=./models/citysemsegformer/1/citysemsegformer.onnx_b1_gpu0_fp16.engine \
 --minShapes="input":1x3x1024x1820 --optShapes="input":1x3x1024x1820 --maxShapes="input":1x3x1024x1820&

#peoplenet_transformer
echo "Building Model peoplenet_transformer"
mkdir -p models/peoplenet_transformer/1
trtexec --onnx=./models/peoplenet_transformer/resnet50_peoplenet_transformer_op17.onnx --fp16 \
 --saveEngine=./models/peoplenet_transformer/1/resnet50_peoplenet_transformer_op17.onnx_b1_gpu0_fp16.engine \
 --minShapes="inputs":1x3x544x960 --optShapes="inputs":1x3x544x960 --maxShapes="inputs":1x3x544x960&

#reidentificationnet
echo "Building Model reidentificationnet"
mkdir -p models/reidentificationnet/1
trtexec --minShapes=input:1x3x256x128 --optShapes=input:8x3x256x128 --maxShapes=input:16x3x256x128 \
 --fp16 --saveEngine=models/reidentificationnet/1/resnet50_market1501_aicity156.onnx_b16_gpu0_fp16.engine \
 --onnx=models/reidentificationnet/resnet50_market1501_aicity156.onnx &

#retail_object_detection_binary_dino
echo "Building Model retail_object_detection_binary_dino"
mkdir -p models/retail_object_detection_binary_dino/1
trtexec --onnx=models/retail_object_detection_binary_dino/retail_object_detection_dino_binary.onnx \
 --saveEngine=models/retail_object_detection_binary_dino/1/retail_object_detection_dino_binary.onnx_b1_gpu0_fp32.engine \
 --minShapes=inputs:1x3x540x960 --optShapes=inputs:1x3x540x960 --maxShapes=inputs:1x3x540x960 \
 --sparsity=enable&

#retail_object_recognition
echo "Building Model retail_object_recognition"
mkdir -p models/retail_object_recognition/1
trtexec --onnx=models/retail_object_recognition/retail_object_recognition.onnx \
 --saveEngine=models/retail_object_recognition/1/retail_object_recognition.onnx_b16_gpu0_fp16.engine \
 --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 \
 --fp16 --sparsity=enable&

#peoplenet
echo "Building Model peoplenet..."
mkdir  -p  models/peoplenet/1
trtexec --onnx=./models/peoplenet/resnet34_peoplenet_int8.onnx --int8 \
--calib=./models/peoplenet/resnet34_peoplenet_int8.txt --saveEngine=./models/peoplenet/1/resnet34_peoplenet_int8.onnx_b2_gpu0_int8.engine \
--minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":2x3x544x960 --maxShapes="input_1:0":2x3x544x960&

#poseclassificationnet
echo "Building Model poseclassificationnet..."
mkdir  -p  models/poseclassificationnet/1
trtexec --onnx=./models/poseclassificationnet/st-gcn_3dbp_nvidia.onnx --fp16 \
--saveEngine=./models/poseclassificationnet/1/st-gcn_3dbp_nvidia.onnx_b4_gpu0_fp16.engine \
--minShapes="input":1x3x300x34x1 --optShapes="input":4x3x300x34x1 --maxShapes="input":4x3x300x34x1&

#bodypose3dnet
echo "Building Model bodypose3dnet..."
mkdir -p models/bodypose3dnet/1
trtexec --onnx=models/bodypose3dnet/bodypose3dnet_accuracy.onnx --fp16 \
	--saveEngine=models/bodypose3dnet/1/bodypose3dnet_accuracy.onnx_b8_gpu0_fp16.engine \
	--minShapes="input0":1x3x256x192,k_inv:1x3x3,t_form_inv:1x3x3,scale_normalized_mean_limb_lengths:1x36,mean_limb_lengths:1x36 \
	--optShapes="input0":8x3x256x192,k_inv:8x3x3,t_form_inv:8x3x3,scale_normalized_mean_limb_lengths:8x36,mean_limb_lengths:8x36 \
	--maxShapes="input0":8x3x256x192,k_inv:8x3x3,t_form_inv:8x3x3,scale_normalized_mean_limb_lengths:8x36,mean_limb_lengths:8x36

#LPD/LPR
echo "Building Models for LPD/LPR..."
mkdir -p models/trafficcamnet/1
trtexec --onnx=models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx --int8 --calib=models/trafficcamnet/resnet18_trafficcamnet_pruned_int8.txt \
 --saveEngine=models/trafficcamnet/1/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_int8.engine --minShapes="input_1:0":1x3x544x960 \
 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960

mkdir -p models/LPD_us/1
trtexec --onnx=models/LPD_us/LPDNet_usa_pruned_tao5.onnx --int8 --calib=models/LPD_us/usa_cal_10.1.0.bin \
 --saveEngine=models/LPD_us/1/LPDNet_usa_pruned_tao5.onnx_b16_gpu0_int8.engine --minShapes="input_1:0":1x3x480x640 \
 --optShapes="input_1:0":16x3x480x640 --maxShapes="input_1:0":16x3x480x640

mkdir -p models/LPD_ch/1
trtexec --onnx=models/LPD_ch/LPDNet_CCPD_pruned_tao5.onnx --fp16 \
 --saveEngine=models/LPD_ch/1/LPDNet_CCPD_pruned_tao5.onnx_b16_gpu0_fp16.engine --minShapes="input_1:0":1x3x1168x720 \
 --optShapes="input_1:0":16x3x1168x720 --maxShapes="input_1:0":16x3x1168x720

mkdir -p models/LPR_us/1
 trtexec --onnx=models/LPR_us/us_lprnet_baseline18_deployable.onnx --fp16 \
 --saveEngine=models/LPR_us/1/us_lprnet_baseline18_deployable.onnx_b16_gpu0_fp16.engine --minShapes="image_input":1x3x48x96 \
 --optShapes="image_input":8x3x48x96 --maxShapes="image_input":16x3x48x96

mkdir -p models/LPR_ch/1
 trtexec --onnx=models/LPR_ch/ch_lprnet_baseline18_deployable.onnx --fp16 \
 --saveEngine=models/LPR_ch/1/ch_lprnet_baseline18_deployable.onnx_b16_gpu0_fp16.engine --minShapes="image_input":1x3x48x96 \
 --optShapes="image_input":8x3x48x96 --maxShapes="image_input":16x3x48x96
