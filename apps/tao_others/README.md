## Description
This sample is to show the following TAO models runing with DeepStream

* 2D Bodypose
* Facial Landmarks Estimation
* EmotionNet
* Gaze Estimation
* GestureNet
* HeartRateNet
* Retail Object Detection 100 classes
* Retail Object Detection Binary
* Retail Object Recognition
* PeopleNet Transformer
* ReIdentificationNet
* PoseClassificationNet

## Prerequisition

* [DeepStream SDK 6.2 GA or above](https://developer.nvidia.com/deepstream-sdk)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation

* [TAO models](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html)

  Nvidia has provides all the trainable models in NGC.
  The models used in the sample application are pre-trained models provided by TAO:

| Model name | NGC link  | Version |
|------------|-----------|---------|
| FaceDetect |[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:facenet)|pruned_quantized_v2.0.1|
| Facial Landmarks Estimation|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:fpenet)|deployable_v3.0|
| EmotionNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:emotionnet)|deployable_v1.0|
| Gaze Estimation|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:gazenet)|deployable_v1.0|
| GestureNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:gesturenet)|deployable_v2.0.2|
| HeartRateNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:heartratenet)|deployable_v1.0|
| BodyPoseNet|[link](https://ngc.nvidia.com/catalog/models/nvidia:tao:bodyposenet)|deployable_v1.0.1|
| Retail Object Detection 100 Classes|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_detection)|deployable_100_v1.0|
| Retail Object Detection Binary | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_detection)|deployable_binary_v1.0|
|PeopleNet Transformer | [link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer)|deployable_v1.0|
|Retail Object Recognition|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition)|deployable_v1.0|
|ReIdentificationNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet)|deployable_v1.1|
|PoseClassificationNet|[link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/poseclassificationnet)|deployable_v1.0|

  The [Bodypose2D backbone](https://ngc.nvidia.com/catalog/models/nvidia:tao:bodyposenet) can be trained and deployed with TAO tools.

  There is blog for introducing training and optimization for bodypose 2D estimation model:

 https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tao-toolkit-part-1

## Download

1. Download Project with HTTPS
```
    sudo apt update
    sudo apt install git-lfs
    git lfs install --skip-repo
    git clone -b master https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```
2. Prepare Models and TensorRT engine

There are pre-trained TAO models available in [NGC](https://ngc.nvidia.com/catalog/models) for faciallandmarks, emotion, gesture, gaze and heart rate.

Please run the following script to download pre-trained models and generate GazeNet and Gesture engines with tao-converter tool:

```
    cd deepstream_tao_apps
    chmod 755 download_models.sh
    ./download_models.sh
```

## Triton Server Settings
 
 The DeepStream sample applications can work as Triton client with the [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), one of the following two methods can be used to set up the Triton Inference Server before starting a gst-nvinferserver inferncing DeepStream application.

 - Native Triton Inference Server, please refer to [Triton Server](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/release/tao4.0_ds6.3ga/triton_server.md)
 - Stand-alone Triton Inference server, please refer to [Triton grpc server](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/release/tao4.0_ds6.3ga/triton_server_grpc.md).
 
Please enable Triton or Triton gRPC inferencing with the app YAML configurations.

E.G. To set deepstream-bodypose2d-app running with native Triton Inference Server, the "primary-gie" part in deepstream-bodypose2d-app/bodypose2d_app_config.yml can be modified as following:

```
primary-gie:
  #0:nvinfer, 1:nvinfeserver
  plugin-type: 1
  #config-file-path: ../../../configs/nvinfer/bodypose2d_tao/bodypose2d_pgie_config.yml
  config-file-path: ../../../configs/triton/bodypose2d_tao/bodypose2d_pgie_config.yml
  #config-file-path: ../../../configs/triton-grpc/bodypose2d_tao/bodypose2d_pgie_config.yml
  unique-id: 1

```

## Build and Run
Build the application
```
    make
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
    cd apps/tao_others
```

Start to run the 2D bodypose application
```
    cd deepstream-bodypose2d-app
    ./deepstream-bodypose2d-app [1:file sink|2:fakesink|3:display sink|4:rtsp output]  \
     <bodypose2d model config file> <udp port> <rtsp port> <input uri> \
     ... <input uri> <out filename>

OR
    ./deepstream-bodypose2d-app <app YAML config file>
```
Start to run the facial landmark application
```
    cd deepstream-faciallandmark-app
    ./deepstream-faciallandmark-app [1:file sink|2:fakesink|3:display sink]  \
    <faciallandmark model config file> <input uri> ... <input uri> <out filename>
OR
    ./deepstream-faciallandmark-app <app YAML config file>
```
Start to run the emotions application
```
    cd deepstream-emotion-app
    ./deepstream-emotion-app [1:file sink|2:fakesink|3:display sink]  \
     <faciallandmark model config> <input uri> ... <input uri> <out filename>
OR
    ./deepstream-emotion-app <app YAML config file>
```

Start to run the gazenet application
```
    cd deepstream-gaze-app
    ./deepstream-gaze-app [1:file sink|2:fakesink|3:display sink]  \
    <faciallandmark model config> <input uri> ... <input uri> <out filename>
OR
    ./deepstream-gaze-app <app YAML config file>
```

Start to run the gesture application
```
    cd deepstream-gesture-app
    ./deepstream-gesture-app [1:file sink|2:fakesink|3:display sink]  \
     [1:right hand|2:left hand|3:both hands] <bodypose2d model config file> \
     <input uri> ... <input uri> <out filename>
OR
    ./deepstream-gesture-app <app YAML config file>
```

Start to run the heartrate application
```
    cd deepstream-heartrate-app
    ./deepstream-heartrate-app [1:file sink|2:fakesink|3:display sink]  \
     <input uri> ... <input uri> <out filename>
OR
    ./deepstream-heartrate-app <app YAML config file>
```

Start to run the pose classification application
```
    cd deepstream-pose-classification
    ./deepstream-pose-classification-app <app YAML config file>
```

Start to run the mdx perception application
```
    cd deepstream-mdx-perception-app
    ./deepstream-mdx-perception-app -c <txt config file>
OR
    ./deepstream-mdx-perception-app <app YAML config file>
```


A sample of 2D bodypose:

`./deepstream-bodypose2d-app 1 ../../../configs/nvinfer/bodypose2d_tao/sample_bodypose2d_model_config.txt 0 0 file:///usr/data/bodypose2d_test.png ./body2dout`

or

`./deepstream-bodypose2d-app bodypose2d_app_config.yml`

A sample of facial landmark:

`./deepstream-faciallandmark-app 1 ../../../configs/nvinfer/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./landmarkout`

or
`./deepstream-faciallandmark-app faciallandmark_app_config.yml`

A sample of emotions:

`./deepstream-emotion-app 1 ../../../configs/nvinfer/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./emotion`

or
`./deepstream-emotion-app emotion_app_config.yml`

A sample of gazenet:

`./deepstream-gaze-app 1 ../../../configs/nvinfer/facial_tao/sample_faciallandmarks_config.txt file:///usr/data/faciallandmarks_test.jpg ./gazenet`

or
`./deepstream-gaze-app gazenet_app_config.yml`

A sample of gesture:

`./deepstream-gesture-app 1 1 ../../../nvinfer/configs/bodypose2d_tao/sample_bodypose2d_model_config.txt file:///usr/data/bodypose2d_test.png ./gesture`

or
`./deepstream-gesture-app gesture_app_config.yml`

A sample of heartrate:

`./deepstream-heartrate-app 1 file:///usr/data/test_video.mp4 ./heartrate`

or
`./deepstream-heartrate-app heartrate_app_config.yml`

A sample of mdx perception:

`./deepstream-mdx-perception-app  -c ../../../configs/app/peoplenet_reidentification.yml`

A sample of pose classification:

`./deepstream-pose-classification-app ../../../configs/app/deepstream_pose_classification_config.yaml`

## Known Issue

1.The GazeNet and HeartRateNet models are all multiple input layers models. DeepStream can generate engine from such models but the implementation of buffer allocation has some problems. So if running the GazeNet and HeartRateNet sample applications without engine, they will fail with core dump for the first time running. The engine will be generated after the first time running. When running the applications again, they will work.

Another workaround is to generate the engines outside the applications. The 'download_models.sh' script will download the GazeNet and HeartRateNet models. Please refer to the TAO tao-converter tool document: https://docs.nvidia.com/tao/tao-user-guide/text/tensorrt.html
