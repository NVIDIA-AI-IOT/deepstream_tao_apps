#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin


if [ ! ${IS_JETSON_PLATFORM} ]; then
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_8.6.3.1_x86/files?redirect=true&path=tao-converter' -O tao-converter
else
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_jp6.0_aarch64/files?redirect=true&path=tao-converter' -O tao-converter
fi
chmod 755 tao-converter

#detection
#dssd
echo "Building Model dssd..."
mkdir  -p  models/dssd/1
trtexec --onnx=./models/dssd/dssd_resnet18_epoch_118.onnx --int8 --calib=./models/dssd/dssd_cal.bin --saveEngine=./models/dssd/1/dssd_resnet18_epoch_118.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&

#efficientdet
echo "Building Model efficientdet..."
mkdir  -p  models/efficientdet/1
trtexec --onnx=./models/efficientdet/d0_avlp_544_960.onnx --int8 --calib=./models/efficientdet/d0_avlp_544_960.cal --saveEngine=./models/efficientdet/1/d0_avlp_544_960.onnx_b1_gpu0_int8.engine&


#frcnn
echo "Building Model frcnn..."
mkdir  -p  models/frcnn/1
trtexec --onnx=./models/frcnn/frcnn_kitti_resnet18.epoch24_trt8.onnx --int8 --calib=./models/frcnn/cal_frcnn_20230707_cal.bin --saveEngine=./models/frcnn/1/frcnn_kitti_resnet18.epoch24_trt8.onnx_b4_gpu0_int8.engine --minShapes=input_image:1x3x544x960 --optShapes=input_image:2x3x544x960 --maxShapes=input_image:4x3x544x960&

 #retinanet
echo "Building Model retinanet..."
mkdir  -p  models/retinanet/1
trtexec --onnx=./models/retinanet/retinanet_resnet18_epoch_080_its.onnx --int8 --calib=./models/retinanet/retinanet_resnet18_epoch_080_its_tao5.cal --saveEngine=./models/retinanet/1/retinanet_resnet18_epoch_080_its.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&

#ssd
echo "Building Model ssd..."
mkdir  -p  models/ssd/1
trtexec --onnx=./models/ssd/ssd_resnet18_epoch_074.onnx --int8 --calib=./models/ssd/ssd_cal.bin --saveEngine=./models/ssd/1/ssd_resnet18_epoch_074.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&

#yolov3
echo "Building Model yolov3..."
mkdir  -p  models/yolov3/1
trtexec --onnx=./models/yolov3/yolov3_resnet18_398.onnx --int8 --calib=./models/yolov3/cal.bin.trt8517 --saveEngine=./models/yolov3/1/yolov3_resnet18_398.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960 --layerPrecisions=cls/Sigmoid:fp32,cls/Sigmoid_1:fp32,box/Sigmoid_1:fp32,box/Sigmoid:fp32,cls/Reshape_reshape:fp32,box/Reshape_reshape:fp32,Transpose2:fp32,sm_reshape:fp32,encoded_sm:fp32,conv_big_object:fp32,cls/mul:fp32,box/concat_concat:fp32,box/add_1:fp32,box/mul_4:fp32,box/add:fp32,box/mul_6:fp32,box/sub_1:fp32,box/add_2:fp32,box/add_3:fp32,yolo_conv1_6:fp32,yolo_conv1_6_lrelu:fp32,yolo_conv2:fp32,Resize1:fp32,yolo_conv1_5_lrelu:fp32,encoded_bg:fp32,yolo_conv4_lrelu:fp32,yolo_conv4:fp32&

#yolov4
echo "Building Model yolov4..."
mkdir  -p  models/yolov4/1
trtexec --onnx=./models/yolov4/yolov4_resnet18_epoch_080.onnx --int8 --calib=./models/yolov4/cal_trt861.bin --saveEngine=./models/yolov4/1/yolov4_resnet18_epoch_080.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&

#yolov4-tiny
echo "Building Model yolov4-tiny..."
mkdir  -p  models/yolov4-tiny/1
trtexec --onnx=./models/yolov4-tiny/yolov4_cspdarknet_tiny_397.onnx --int8 --calib=./models/yolov4-tiny/cal.bin.trt8517 --saveEngine=./models/yolov4-tiny/1/yolov4_cspdarknet_tiny_397.onnx_b4_gpu0_int8.engine --minShapes=Input:1x3x544x960 --optShapes=Input:2x3x544x960 --maxShapes=Input:4x3x544x960&

#yolov5
#echo "Building Model yolov5..."
#mkdir  -p  models/yolov5/1
#/usr/src/tensorrt/bin/trtexec --fp16  --onnx=./models/yolov5/yolov5s.onnx  \
# --saveEngine=./models/yolov5/1/yolov5s.onnx_b4_gpu0_fp16.engine --minShapes=images:1x3x672x672 \
# --optShapes=images:4x3x672x672 --maxShapes=images:4x3x672x672 --shapes=images:4x3x672x672 --workspace=10000

#for classifcaiton
#multi_task
#mkdir  -p  models/multi_task/1
#./tao-converter -k nvidia_tlt -t fp16  -b 4 -d 3,80,60 -e models/multi_task/1/abc.etlt_b4_gpu0_fp16.engine \
# models/multi_task/abc.etlt

#for instance segmentation
#for peopleSegNet
echo "Building Model peopleSegNet..."
mkdir  -p  models/peopleSegNet/1
./tao-converter -k nvidia_tlt -t int8 -c models/peopleSegNet/peoplesegnet_resnet50_int8.txt -b 4 -d 3,576,960 \
 -e models/peopleSegNet/1/peoplesegnet_resnet50.etlt_b4_gpu0_int8.engine  models/peopleSegNet/peoplesegnet_resnet50.etlt&
#for segmentation
#peopleSemSegNet
echo "Building Model peopleSemSegNet..."
mkdir  -p  models/peopleSemSegNet_vanilla/1
./tao-converter -k tlt_encode -t int8 -c models/peopleSemSegNet_vanilla/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8.cache -p input_1:0,1x3x544x960,4x3x544x960,4x3x544x960  \
  -e models/peopleSemSegNet_vanilla/1/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16.etlt_b4_gpu0_int8.engine \
  models/peopleSemSegNet_vanilla/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16.etlt&

mkdir  -p  models/peopleSemSegNet_shuffle/1
./tao-converter -k tlt_encode -t int8 -c models/peopleSemSegNet_shuffle/peoplesemsegnet_shuffleseg_cache.txt -p input_2:0,1x3x544x960,4x3x544x960,4x3x544x960  \
  -e models/peopleSemSegNet_shuffle/1/peoplesemsegnet_shuffleseg_etlt.etlt_b4_gpu0_int8.engine \
  models/peopleSemSegNet_shuffle/peoplesemsegnet_shuffleseg_etlt.etlt&

#unet
echo "Building Model unet..."
mkdir  -p  models/unet/1
trtexec --onnx=./models/unet/unet_resnet18.onnx --int8 --calib=./models/unet/unet_cal.bin --saveEngine=./models/unet/1/unet_resnet18.onnx_b4_gpu0_int8.engine --minShapes="input_1:0":1x3x320x320 --optShapes="input_1:0":2x3x320x320 --maxShapes="input_1:0":4x3x320x320&

#citysemsegformer
echo "Building Model citysemsegformer..."
mkdir  -p  models/citysemsegformer/1
./tao-converter -k tlt_encode -t fp16   -p input,1x3x1024x1820,1x3x1024x1820,1x3x1024x1820 \
 -e models/citysemsegformer/1/citysemsegformer.etlt_b1_gpu0_fp16.engine \
 models/citysemsegformer/citysemsegformer.etlt&

#bodypose2d
echo "Building Model bodypose2d..."
mkdir  -p  models/bodypose2d/1
./tao-converter -k nvidia_tlt -t fp16 -p  input_1:0,1x288x384x3,32x288x384x3,32x288x384x3 \
 -e models/bodypose2d/1/model.etlt_b32_gpu0_fp16.engine models/bodypose2d/model.etlt&

#gesture
echo "Building Model gesture..."
mkdir  -p  models/gesture/1
./tao-converter -k nvidia_tlt -t int8 -c models/gesture/int8_calibration.txt -p  \
 input_1,1x3x160x160,8x3x160x160,8x3x160x160 -e models/gesture/1/gesture.etlt_b8_gpu0_int8.engine models/gesture/gesture.etlt&

#facenet
echo "Building Model facenet..."
mkdir  -p  models/facenet/1
./tao-converter -k nvidia_tlt -t int8 -c models/facenet/facenet_cal.txt -b 16 -d 3,416,736   \
 -e models/facenet/1/facenet.etlt_b16_gpu0_int8.engine models/facenet/facenet.etlt&

#gesture
echo "Building Model faciallandmark..."
mkdir  -p  models/faciallandmark/1
./tao-converter -k nvidia_tlt -t int8 -c models/faciallandmark/fpenet_cal.txt -b 4  -p  \
 input_face_images,1x1x80x80,2x1x80x80,4x1x80x80 -e models/faciallandmark/1/faciallandmark.etlt_b4_gpu0_int8.engine models/faciallandmark/faciallandmark.etlt&

#peoplenet_transformer
echo "Building Model peoplenet_transformer"
mkdir -p models/peoplenet_transformer/1
trtexec --onnx=./models/peoplenet_transformer/resnet50_peoplenet_transformer_op17.onnx --fp16 \
 --saveEngine=./models/peoplenet_transformer/1/resnet50_peoplenet_transformer_op17.onnx_b1_gpu0_fp16.engine \
 --minShapes="inputs":1x3x544x960 --optShapes="inputs":1x3x544x960 --maxShapes="inputs":1x3x544x960&

#retail_object_recognize
echo "Building Model retail_object_recognize"
mkdir -p models/retail_object_recognition/1
./tao-converter -k nvidia_tlt -t fp16 -p inputs,1x3x224x224,8x3x224x224,16x3x224x224 -e models/retail_object_recognition/1/retail_object_recognition.etlt_b16_gpu0_fp16.engine models/retail_object_recognition/retail_object_recognition.etlt&

#retail_object_detection_100
echo "Building Model retail_object_detection_100"
mkdir -p models/retail_object_detection_100/1
./tao-converter -k nvidia_tlt -t fp16 -p input,1x416x416x3,1x416x416x3,1x416x416x3 -e models/retail_object_detection_100/1/retail_detector_100.etlt_b1_gpu0_fp16.engine models/retail_object_detection_100/retail_detector_100.etlt&

#reidentificationnet
echo "Building Model reidentificationnet"
mkdir -p models/reidentificationnet/1
trtexec --minShapes=input:1x3x256x128 --optShapes=input:8x3x256x128 --maxShapes=input:16x3x256x128 \
 --fp16 --saveEngine=models/reidentificationnet/1/resnet50_market1501_aicity156.onnx_b16_gpu0_fp16.engine \
 --onnx=models/reidentificationnet/resnet50_market1501_aicity156.onnx --workspace=100000&

#retail_object_detection_binary_effdet
echo "Building Model retail_object_detection_binary_effdet"
mkdir -p models/retail_object_detection_binary_effdet/1
trtexec --onnx=models/retail_object_detection_binary_effdet/efficientdet-d5_090.onnx \
 --saveEngine=models/retail_object_detection_binary_effdet/1/efficientdet-d5_090.onnx_b1_gpu0_fp16.engine \
 --minShapes=input:1x544x960x3 --optShapes=input:1x544x960x3 --maxShapes=input:1x544x960x3 --workspace=102400 \
 --fp16 --sparsity=enable

#retail_object_detection_binary_dino
echo "Building Model retail_object_detection_binary_dino"
mkdir -p models/retail_object_detection_binary_dino/1
trtexec --onnx=models/retail_object_detection_binary_dino/retail_object_detection_dino_binary.onnx \
 --saveEngine=models/retail_object_detection_binary_dino/1/retail_object_detection_dino_binary.onnx_b1_gpu0_fp16.engine \
 --minShapes=inputs:1x3x544x960 --optShapes=inputs:1x3x544x960 --maxShapes=inputs:1x3x544x960 --workspace=102400 \
 --fp16 --sparsity=enable&

#retail_object_detection_meta
echo "Building Model retail_object_detection_meta"
mkdir -p models/retail_object_detection_meta/1
trtexec --onnx=models/retail_object_detection_meta/retail_object_detection_dino_meta.onnx \
 --saveEngine=models/retail_object_detection_meta/1/retail_object_detection_dino_meta.onnx_b1_gpu0_fp16.engine \
 --minShapes=inputs:1x3x544x960 --optShapes=inputs:1x3x544x960 --maxShapes=inputs:1x3x544x960 --workspace=102400 \
 --fp16 --sparsity=enable&

#retail_object_recognition
echo "Building Model retail_object_recognition"
mkdir -p models/retail_object_recognition/1
mkdir -p models/retail_object_recognition/1
trtexec --onnx=models/retail_object_recognition/retail_object_recognition.onnx \
 --saveEngine=models/retail_object_recognition/1/retail_object_recognition.onnx_b16_gpu0_fp16.engine \
 --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --workspace=102400  \
 --fp16 --sparsity=enable&

#peoplenet
echo "Building Model peoplenet..."
mkdir  -p  models/peoplenet/1
trtexec --onnx=./models/peoplenet/resnet34_peoplenet_int8.onnx --int8 \
--calib=./models/peoplenet/resnet34_peoplenet_int8.txt --saveEngine=./models/peoplenet/1/resnet34_peoplenet_int8.onnx_b1_gpu0_int8.engine \
--minShapes="input_1:0":1x3x544x960 --optShapes="input_1:0":1x3x544x960 --maxShapes="input_1:0":1x3x544x960&

#poseclassificationnet
echo "Building Model poseclassificationnet..."
mkdir  -p  models/poseclassificationnet/1
./tao-converter -k nvidia_tao -t fp16 -p  input,1x3x300x34x1,1x3x300x34x1,1x3x300x34x1 \
 -e models/poseclassificationnet/1/st-gcn_3dbp_nvidia.etlt_b1_gpu0_fp16.engine models/poseclassificationnet/st-gcn_3dbp_nvidia.etlt&

#bodypose3dnet
echo "Building Model bodypose3dnet..."
mkdir -p models/bodypose3dnet/1
./tao-converter -k tlt_encode -t fp16 -d 3,256,192 -b 8 \
-p input0,1x3x256x192,8x3x256x192,8x3x256x192 \
-p k_inv,1x3x3,8x3x3,8x3x3 \
-p t_form_inv,1x3x3,8x3x3,8x3x3 \
-p scale_normalized_mean_limb_lengths,1x36,8x36,8x36 \
-p mean_limb_lengths,1x36,8x36,8x36 \
-e models/bodypose3dnet/1/bodypose3dnet_accuracy.etlt_b8_gpu0_fp16.engine \
models/bodypose3dnet/bodypose3dnet_accuracy.etlt&
