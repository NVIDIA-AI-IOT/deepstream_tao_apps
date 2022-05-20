## Description
This sample is to show the following TAO3.0 models runing with DeepStream

* 2D Bodypose
* Facial Landmarks Estimation
* EmotionNet
* Gaze Estimation
* GestureNet
* HeartRateNet

## Prerequisition

* [DeepStream SDK 6.0 GA and above](https://developer.nvidia.com/deepstream-sdk-6.0-members-page)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation

* [TAO3.0 models](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html)

  Nvidia has provides all the trainable models in NGC.
  The models used in the sample application are pre-trained models provided by TAO3.0:

| Model name | NGC link  | Version |
|------------|-----------|---------|
| FaceDetect |[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:facenet)|pruned_quantized_v2.0.1|
| Facial Landmarks Estimation|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:fpenet)|deployable_v2.0|
| EmotionNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:emotionnet)|deployable_v1.0|
| Gaze Estimation|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:gazenet)|deployable_v1.0|
| GestureNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:gesturenet)|deployable_v2.0.1|
| HeartRateNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:heartratenet)|deployable_v1.0|
| BodyPoseNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:bodyposenet)|deployable_v1.0.1|

  The [Bodypose2D backbone](https://ngc.nvidia.com/catalog/models/nvidia:tao:bodyposenet) can be trained and deployed with TAO3.0 tools.
  
  There is blog for introducing training and optimization for bodypose 2D estimation model:
  
 https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1

* Download x86 or Jetson tao-converter which is compatible to your platform from the following links.

| Platform   |  Compute                       |        Link                                              |
|------------|--------------------------------|----------------------------------------------------------|
|x86 + GPU   |CUDA 10.2/cuDNN 8.0/TensorRT 7.1|[link](https://developer.nvidia.com/cuda102-trt71)|
|x86 + GPU   |CUDA 10.2/cuDNN 8.0/TensorRT 7.2|[link](https://developer.nvidia.com/cuda102-cudnn80-trt72-0)|
|x86 + GPU   |CUDA 11.0/cuDNN 8.0/TensorRT 7.1|[link](https://developer.nvidia.com/cuda110-cudnn80-trt71-0)|
|x86 + GPU   |CUDA 11.0/cuDNN 8.0/TensorRT 7.2|[link](https://developer.nvidia.com/cuda110-rt72)|
|x86 + GPU   |CUDA 11.1/cuDNN 8.0/TensorRT 7.2.2|[link](https://developer.nvidia.com/cuda111-cudnn80-trt72-0)|
|x86 + GPU   |CUDA 11.3/cuDNN 8.1/TensorRT 8.0|[link](https://developer.nvidia.com/tao-converter-80)|
|Jetson      |JetPack 4.4                     |[link](https://developer.nvidia.com/cuda102-trt71-jp44-0)   |
|Jetson      |JetPack 4.5                     |[link](https://developer.nvidia.com/cuda110-cudnn80-trt72-0)   |
|Jetson      |JetPack 4.6                     |[link](https://developer.nvidia.com/jp46-20210820t231431z-001zip)   |
|Clara AGX   |CUDA 11.1/CuDNN8.0.5/TensorRT7.2.2|[link](https://developer.nvidia.com/tao-converter)   |

## Download

1. Download Project with HTTPS
```
    git clone -b release/tao3.0_ds6.1ga https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```
2. Prepare Models and TensorRT engine

There are pre-trained TAO models available in [NGC](https://ngc.nvidia.com/catalog/models) for faciallandmarks, emotion, gesture, gaze and heart rate.

Please run the following script to download pre-trained models and generate GazeNet and Gesture engines with tao-converter tool: 

```
    cd deepstream_tao_apps
    chmod 755 download_models.sh
    export TAO_CONVERTER=the file path of tao-converter
    export MODEL_PRECISION=fp16
    ./download_models.sh
```

## Build and Run
Go into the application folder
```
    cd apps/tao_others
```

For dGPU
```
    export CUDA_VER=11.6
    make
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
```
For Jetson

```
    cd /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate
    make
    cp libnvdsgst_videotemplate.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
    rm -rf ~/.cache/gstreamer-1.0/
```
And then back to the sample application folder and build the applications.

```
    cd -
    export CUDA_VER=11.4
    make
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
```

Start to run the 2D bodypose application
```
    cd deepstream-bodypose2d-app
    ./deepstream-bodypose2d-app [1:file sink|2:fakesink|3:display sink|4:rtsp output]  \
     <bodypose2d model config file> <udp port> <rtsp port> <input uri> \
     ... <input uri> <out filename>
```
Start to run the facial landmark application
```
    cd deepstream-faciallandmark-app
    ./deepstream-faciallandmark-app [1:file sink|2:fakesink|3:display sink]  \
    <faciallandmark model config file> <input uri> ... <input uri> <out filename>
```
Start to run the emotions application
```
    cd deepstream-emotion-app
    ./deepstream-emotion-app [1:file sink|2:fakesink|3:display sink]  \
     <faciallandmark model config> <input uri> ... <input uri> <out filename>
```

Start to run the gazenet application
```
    cd deepstream-gaze-app
    ./deepstream-gaze-app [1:file sink|2:fakesink|3:display sink]  \
    <faciallandmark model config> <input uri> ... <input uri> <out filename>
```

Start to run the gesture application
```
    cd deepstream-gesture-app
    ./deepstream-gesture-app [1:file sink|2:fakesink|3:display sink]  \
     [1:right hand|2:left hand|3:both hands] <bodypose2d model config file> \
     <input uri> ... <input uri> <out filename>
```

Start to run the heartrate application
```
    cd deepstream-heartrate-app
    ./deepstream-heartrate-app [1:file sink|2:fakesink|3:display sink]  \
     <input uri> ... <input uri> <out filename>
```

A sample of 2D bodypose:

`./deepstream-bodypose2d-app 1 ../../../configs/bodypose2d_tao/sample_bodypose2d_model_config.txt 0 0 file:///usr/data/bodypose2d_test.png ./body2dout`

A sample of facial landmark:

`./deepstream-faciallandmark-app 1 ../../../configs/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./landmarkout`

A sample of emotions:

`./deepstream-emotion-app 1 ../../../configs/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./emotion`

A sample of gazenet:

`./deepstream-gaze-app 1 ../../../configs/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./gazenet`

A sample of gesture:

`./deepstream-gesture-app 1 ../../../configs/bodypose2d_tao/sample_bodypose2d_model_config.txt file:///usr/data/bodypose2d_test.png ./gesture`

A sample of heartrate:

`./deepstream-heartrate-app 1 file:///usr/data/test_video.mp4 ./heartrate`

## Known Issue

The GazeNet and HeartRateNet models are all multiple input layers models. DeepStream can generate engine from such models but the implementation of buffer allocation has some problems. So if running the GazeNet and HeartRateNet sample applications without engine, they will fail with core dump for the first time running. The engine will be generated after the first time running. When running the applications again, they will work.

Another workaround is to generate the engines outside the applications. The 'download_models.sh' script will download the GazeNet and HeartRateNet models. Please refer to the TAO tao-converter tool document: https://docs.nvidia.com/tao/tao-user-guide/text/tensorrt.html

Current bodypose2d model does not support INT8 with TensorRT 8.x, only FP16 and FP32 works.
