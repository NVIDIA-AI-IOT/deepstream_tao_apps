## Description
This sample is to show the following TLT3.0 models runing with DeepStream

* 2D Bodypose
* Facial Landmarks Estimation
* EmotionNet

## Prerequisition

* [DeepStream SDK 6.0 and above](https://developer.nvidia.com/deepstream-sdk-6.0-members-page)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation

* [TLT3.0 models](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/overview.html)

  Nvidia has provides all the trainable models in NGC.
  The models used in the sample application are pre-trained models provided by TLT3.0:

| Model name | NGC link  | Version |
|------------|-----------|---------|
| FaceDetect |[link](https://ngc.nvidia.com/catalog/models/nvidia:tlt_facenet)|deployable_v1.0|
| Facial Landmarks Estimation|[link](https://ngc.nvidia.com/catalog/models/nvidia:tlt_fpenet)|deployable_v1.0|
| EmotionNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tlt_emotionnet)|deployable_v1.0|

  The [Bodypose2D backbone](https://ngc.nvidia.com/catalog/models/nvidia:tlt_bodyposenet) can be trained and deployed with TLT3.0 tools.
  
  There is blog for introducing training and optimization for bodypose 2D estimation model:
  
  https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tlt-part-1/

## Download

1. Download Project with HTTPS
```
    git clone -b release/tlt3.0 https://github.com/NVIDIA-AI-IOT/deepstream_tlt_apps.git
```
2. Prepare Models and TensorRT engine

There are pre-trained TLT models available in [NGC](https://ngc.nvidia.com/catalog/models) for faciallandmarks, emotion, gesture, gaze and heart rate.

Please run the following script to download pre-trained models: 

```
    cd deepstream_tlt_apps
    chmod 755 download_models.sh
    ./download_models.sh
```
To run the bodypose2D sample application, the bodypose2D trainable model should be trained with TLT3.0 tool. 
The trained bodypose2D model can be put in ./models/bodypose2d directory

3. For Jetson platform, gst-nvvideotemplate is not supported by deepstream 6.0 EA package. The gst-nvvideotemplate source code can be available with the servers and workstations package. Please copy the source code to the coresponding folder on Jetson device.

/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate

And then start to build the plugin on Jetson:
```
    cd /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate
    make
    cp libnvdsgst_videotemplate.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
    rm -rf ~/.cache/gstreamer-1.0/
```

## Build and Run
Go into the application folder
```
    cd apps/tlt_others
```

For dGPU
```
    export CUDA_VER=11.1
    make
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
```
For Jetson

```
    export CUDA_VER=10.2
    make
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
```

Start to run the 2D bodypose application
```
    cd deepstream-bodypose2d-app
    ./deepstream-bodypose2d-app [1:file sink|2:fakesink|3:display sink]  \
     <bydypose2d model config file> <input uri> ... <input uri> <out filename>
```
Start to run the facial landmark application
```
    cd deepstream-faciallandmark-app
    ./deepstream-faciallandmark-app [1:file sink|2:fakesink|3:display sink]  \
        <input uri> ... <input uri> <out filename>
```

Start to run the emotionNet application
```
    cd deepstream-emotion-app
    ./deepstream-emotion-app [1:file sink|2:fakesink|3:display sink]  \
     <input uri> ... <input uri> <out filename>
```

A sample of 2D bodypose:

`./deepstream-bodypose2d-app 1 ../../../configs/bodypose2d_tlt/sample_bodypose2d_model_config.txt file:///usr/data/bodypose2d_test.png ./body2dout`

A sample of facial landmark:

`./deepstream-faciallandmark-app 1 file:///usr/data/faciallandmarks_test.jpg ./landmarkout`

A sample of emotions:

`./deepstream-emotion-app 1 file:///usr/data/faciallandmarks_test.jpg ./emotion`
