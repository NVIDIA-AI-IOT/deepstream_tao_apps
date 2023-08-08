#!/bin/bash

IS_JETSON_PLATFORM=`uname -i | grep aarch64`

export PATH=$PATH:/usr/src/tensorrt/bin


if [ ! ${IS_JETSON_PLATFORM} ]; then
     wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v4.0.0_trt8.5.2.2_x86/files/tao-converter' -O tao-converter
else
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_aarch64/files/tao-converter' -O tao-converter
fi
chmod 755 tao-converter

#detection
#dssd
echo "Building Model dssd..."
mkdir  -p  models/dssd/1
./tao-converter -k nvidia_tlt -t int8 -c models/dssd/dssd_cal.bin \
 -b 4 -d 3,544,960 -e models/dssd/1/dssd.etlt_b4_gpu0_int8.engine models/dssd/dssd.etlt

#efficientdet
echo "Building Model efficientdet..."
mkdir  -p  models/efficientdet/1
./tao-converter -k nvidia_tlt -t int8 -c models/efficientdet/d0_avlp_544_960.cal -p image_arrays:0,1x3x544x960,1x3x544x960,1x3x544x960 \
 -e models/efficientdet/1/d0_avlp_544_960.etlt_b1_gpu0_int8.engine models/efficientdet/d0_avlp_544_960.etlt

#frcnn
echo "Building Model frcnn..."
mkdir  -p  models/frcnn/1
./tao-converter -k nvidia_tlt -t int8 -c models/frcnn/cal_8517.bin -b 4 -d 3,544,960 \
 -e models/frcnn/1/frcnn_kitti_resnet18.epoch24_trt8.etlt_b4_gpu0_int8.engine models/frcnn/frcnn_kitti_resnet18.epoch24_trt8.etlt

 #retinanet
echo "Building Model retinanet..."
mkdir  -p  models/retinanet/1
./tao-converter -k nvidia_tlt -t int8 -c models/retinanet/retinanet_resnet18_epoch_080_its_trt8.cal -b 4 -d 3,544,960 \
 -e models/retinanet/1/retinanet_resnet18_epoch_080_its_trt8.etlt_b4_gpu0_int8.engine \
 models/retinanet/retinanet_resnet18_epoch_080_its_trt8.etlt

#ssd
echo "Building Model ssd..."
mkdir  -p  models/ssd/1
./tao-converter -k nvidia_tlt -t int8 -c models/ssd/ssd_cal.bin -b 4 -d 3,544,960 \
 -e models/ssd/1/ssd.etlt_b4_gpu0_int8.engine \
 models/ssd/ssd.etlt

#yolov3
echo "Building Model yolov3..."
mkdir  -p  models/yolov3/1
./tao-converter -k nvidia_tlt -t int8 -c models/yolov3/cal.bin.trt8517 -p Input,1x3x544x960,4x3x544x960,4x3x544x960 \
 -e models/yolov3/1/yolov3_resnet18_398.etlt_b4_gpu0_int8.engine \
 models/yolov3/yolov3_resnet18_398.etlt

#yolov4
echo "Building Model yolov4..."
mkdir  -p  models/yolov4/1
./tao-converter -k nvidia_tlt -t int8 -c models/yolov4/cal.bin.trt8517 -p Input,1x3x544x960,4x3x544x960,4x3x544x960 \
 -e models/yolov4/1/yolov4_resnet18_395.etlt_b4_gpu0_int8.engine \
 models/yolov4/yolov4_resnet18_395.etlt

#yolov4-tiny
echo "Building Model yolov4-tiny..."
mkdir  -p  models/yolov4-tiny/1
./tao-converter -k nvidia_tlt -t int8 -c models/yolov4-tiny/cal.bin.trt8517 -p Input,1x3x544x960,4x3x544x960,4x3x544x960 \
 -e models/yolov4-tiny/1/yolov4_cspdarknet_tiny_397.etlt_b4_gpu0_int8.engine \
 models/yolov4-tiny/yolov4_cspdarknet_tiny_397.etlt

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
 -e models/peopleSegNet/1/peoplesegnet_resnet50.etlt_b4_gpu0_int8.engine  models/peopleSegNet/peoplesegnet_resnet50.etlt
#for segmentation
#peopleSemSegNet
echo "Building Model peopleSemSegNet..."
mkdir  -p  models/peopleSemSegNet_vanilla/1
./tao-converter -k tlt_encode -t int8 -c models/peopleSemSegNet_vanilla/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8.cache -p input_1:0,1x3x544x960,4x3x544x960,4x3x544x960  \
  -e models/peopleSemSegNet_vanilla/1/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16.etlt_b4_gpu0_int8.engine \
  models/peopleSemSegNet_vanilla/peoplesemsegnet_vanilla_unet_dynamic_etlt_int8_fp16.etlt

mkdir  -p  models/peopleSemSegNet_shuffle/1
./tao-converter -k tlt_encode -t int8 -c models/peopleSemSegNet_shuffle/peoplesemsegnet_shuffleseg_cache.txt -p input_2:0,1x3x544x960,4x3x544x960,4x3x544x960  \
  -e models/peopleSemSegNet_shuffle/1/peoplesemsegnet_shuffleseg_etlt.etlt_b4_gpu0_int8.engine \
  models/peopleSemSegNet_shuffle/peoplesemsegnet_shuffleseg_etlt.etlt

#unet
echo "Building Model unet..."
mkdir  -p  models/unet/1
./tao-converter -k tlt_encode -t fp16 -c models/unet/unet_cal.bin -p input_1:0,1x3x320x320,4x3x320x320,4x3x320x320 \
 -e models/unet/1/unet_resnet18.etlt_b4_gpu0_fp16.engine models/unet/unet_resnet18.etlt

#citysemsegformer
echo "Building Model citysemsegformer..."
mkdir  -p  models/citysemsegformer/1
./tao-converter -k tlt_encode -t fp16   -p input,1x3x1024x1820,1x3x1024x1820,1x3x1024x1820 \
 -e models/citysemsegformer/1/citysemsegformer.etlt_b1_gpu0_fp16.engine \
 models/citysemsegformer/citysemsegformer.etlt

#bodypose2d
echo "Building Model bodypose2d..."
mkdir  -p  models/bodypose2d/1
./tao-converter -k nvidia_tlt -t fp16 -p  input_1:0,1x288x384x3,32x288x384x3,32x288x384x3 \
 -e models/bodypose2d/1/model.etlt_b32_gpu0_fp16.engine models/bodypose2d/model.etlt

#gesture
echo "Building Model gesture..."
mkdir  -p  models/gesture/1
./tao-converter -k nvidia_tlt -t int8 -c models/gesture/int8_calibration.txt -p  \
 input_1,1x3x160x160,8x3x160x160,8x3x160x160 -e models/gesture/1/gesture.etlt_b8_gpu0_int8.engine models/gesture/gesture.etlt

#facenet
echo "Building Model facenet..."
mkdir  -p  models/facenet/1
./tao-converter -k nvidia_tlt -t int8 -c models/facenet/facenet_cal.txt -b 16 -d 3,416,736   \
 -e models/facenet/1/facenet.etlt_b16_gpu0_int8.engine models/facenet/facenet.etlt

#gesture
echo "Building Model faciallandmark..."
mkdir  -p  models/faciallandmark/1
./tao-converter -k nvidia_tlt -t int8 -c models/faciallandmark/fpenet_cal.txt -b 4  -p  \
 input_face_images,1x1x80x80,2x1x80x80,4x1x80x80 -e models/faciallandmark/1/faciallandmark.etlt_b4_gpu0_int8.engine models/faciallandmark/faciallandmark.etlt

#peoplenet_transformer
echo "Building Model peoplenet_transformer"
mkdir -p models/peoplenet_transformer/1
./tao-converter -k nvidia_tao -t fp16 -p inputs,1x3x544x960,1x3x544x960,1x3x544x960 -e models/peoplenet_transformer/1/resnet50_peoplenet_transformer.etlt_b1_gpu0_fp16.engine models/peoplenet_transformer/resnet50_peoplenet_transformer.etlt

#retail_object_recognize
echo "Building Model retail_object_recognize"
mkdir -p models/retail_object_recognition/1
./tao-converter -k nvidia_tlt -t fp16 -p inputs,1x3x224x224,8x3x224x224,16x3x224x224 -e models/retail_object_recognition/1/retail_object_recognition.etlt_b16_gpu0_fp16.engine models/retail_object_recognition/retail_object_recognition.etlt

#retail_object_detection_100
echo "Building Model retail_object_detection_100"
mkdir -p models/retail_object_detection_100/1
./tao-converter -k nvidia_tlt -t fp32 -p input,1x416x416x3,1x416x416x3,1x416x416x3 -e models/retail_object_detection_100/1/retail_detector_100.etlt_b1_gpu0_fp32.engine models/retail_object_detection_100/retail_detector_100.etlt

#reidentificationnet
echo "Building Model reidentificationnet"
mkdir -p models/reidentificationnet/1
./tao-converter -k nvidia_tao -t fp16 -p input,1x3x256x128,8x3x256x128,16x3x256x128 -e models/reidentificationnet/1/resnet50_market1501_aicity156.etlt_b16_gpu0_fp16.engine models/reidentificationnet/resnet50_market1501_aicity156.etlt

#retail_object_detection_binary
echo "Building Model retail_object_detection_binary"
mkdir -p models/retail_object_detection_binary/1
./tao-converter -k nvidia_tlt -t fp16 -p input,1x416x416x3,1x416x416x3,1x416x416x3 -e models/retail_object_detection_binary/1/retail_detector_binary.etlt_b1_gpu0_fp16.engine models/retail_object_detection_binary/retail_detector_binary.etlt

#peoplenet
echo "Building Model peoplenet..."
mkdir  -p  models/peoplenet/1
./tao-converter -k tlt_encode -t int8 -c models/peoplenet/resnet34_peoplenet_int8.txt \
 -b 1 -d 3,544,960  -e models/peoplenet/1/resnet34_peoplenet_int8.etlt_b1_gpu0_int8.engine \
 models/peoplenet/resnet34_peoplenet_int8.etlt

#poseclassificationnet
echo "Building Model poseclassificationnet..."
mkdir  -p  models/poseclassificationnet/1
./tao-converter -k nvidia_tao -t fp32 -p  input,1x3x300x34x1,1x3x300x34x1,1x3x300x34x1 \
 -e models/poseclassificationnet/1/st-gcn_3dbp_nvidia.etlt_b1_gpu0_fp32.engine models/poseclassificationnet/st-gcn_3dbp_nvidia.etlt

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
models/bodypose3dnet/bodypose3dnet_accuracy.etlt
