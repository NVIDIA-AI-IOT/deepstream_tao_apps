## Description
The emotion deepstream sample application identify human emotion based on the facial landmarks. Current sample application can identify five emotions as neutral, happy, surprise, squint, disgust and scream.

The TAO pretrained models used in the sample application are:

* [Facial Landmarks Estimation](https://ngc.nvidia.com/catalog/models/nvidia:tao:fpenet)
* [FaceNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:facenet)
* [EmotionNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:emotionnet)

## Prerequisition

* DeepStream SDK 6.2 GA and above

## Application Pipeline
The application pipeline graph

![emotion application pipeline](emotion_pipeline.png)

## Build And Run
The application can be build and run seperately.

And then back to the tao applications project directory
```
cd apps/tao_others/deepstream-emotion-app/emotion_impl
make
cd ../
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
./deepstream-emotion-app 1 ../../../configs/nvinfer/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./landmarks
```
