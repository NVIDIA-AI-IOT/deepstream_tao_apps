## Description
The facial landmarks estimation deepstream sample application identify landmarks in human face with face detection model and facial landmarks estimation model.
With the TAO 3.0 pretrained facial landmarks estimation model, the application can idetify 80 landmarks in one human face.

The TAO 3.0 pretrained models used in this sample application:
* [Facial Landmark Estimation](https://ngc.nvidia.com/catalog/models/nvidia:tao:fpenet).
* [FaceNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:facenet)

## Prerequisition

* DeepStream SDK 6.0 and above

  Current DeepStream 6.0 EA version is available in https://developer.nvidia.com/deepstream-sdk-6.0-members-page for specific users.

## Application Pipeline

The application pipeline graph

![faciallandmarks application pipeline](faciallandmarks_pipeline.png)

## Build And Run
The application can be build and run seperately.

```
cd apps/tao_others/deepstream-faciallandmark-app
```

For Jetson platform
```
export CUDA_VER=10.2
```

For dGPU
```
export CUDA_VER=11.1
```

Build the applications and run to inference one picture.
```
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
./deepstream-faciallandmark-app 2 file:///usr/data/faciallandmarks_test.jpg ./landmarks
```

##
