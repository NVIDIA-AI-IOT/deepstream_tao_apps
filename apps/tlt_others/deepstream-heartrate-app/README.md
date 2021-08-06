## Description
The HeartRate deepstream sample application measures a person's heart rate with the face information. 

The TLT 3.0 pretrained models used in this sample application:
* [FaceNet](https://ngc.nvidia.com/catalog/models/nvidia:tlt_facenet)
* [HeartRateNet](https://ngc.nvidia.com/catalog/models/nvidia:tlt_heartratenet)

## Prerequisition

* DeepStream SDK 6.0 and above

  Current DeepStream 6.0 EA version is available in https://developer.nvidia.com/deepstream-sdk-6.0-members-page for specific users.
* gst-nvdsvideotemplate plugin

  Since the HeartRateNet is a multi-input network, the gst-nvinfer plugin can not support HeartRateNet inferencing.
  The gst-nvdsvideotemplate plugin is used in this sample to do the HeartRAteNet inference. The libnvds_heartrateinfer.so is the customized library for gst-nvdsvideotemplate to infer the batched iamges with the Nvidia proprietary inference library.

## Application Pipeline
The application pipeline graph

![HeartRate application pipeline](HR_pipeline.png)

## Build And Run
The application can be build and run seperately.
For Jetson platform

Copy the gst-nvdsvideotemplate plugin source code from DeepStream for servers and workstations package and copy from the following folder:

/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate

```
export CUDA_VER=10.2
cd /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate
make
cp libnvdsgst_videotemplate.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
rm -rf ~/.cache/gstreamer-1.0/
```

For dGPU
```
export CUDA_VER=11.1
```

Build the applications and run to inference one video.
```
cd apps/tlt_others/deepstream-heartrate-app/heartrateinfer_impl
make
cd ../
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
./deepstream-heartrate-app 1 file:///usr/data/test_video.mp4 ./heartrate
```

## Limitations
The sample works only with videos with human faces.

The pretrained model is trained with limited face images. There are some requirements for the video to be recognized:
* The expected range for a person of interest is 0.5 meter.
* above 100 Lux 
* allow up to 10 secs to adapt for lighting changes and face movement
* keep head stable and straight with regard to the camera
* whole face is well illuminated
* do not wear a mask 
* Heart Rate only supports 15-30 FPS.
* Head Angle Relative to Camera Plane
   Yaw: -75 to 75 degrees
   Pitch: -60 to 45 degrees
   Roll: -45 to 45 degrees

## Known Issue
The HeartRateNet is a multiple input layers model. DeepStream can generate engine from such models but the implementation of buffer allocation has some problems. So if running the HeartRateNet sample application without engine, they will fail with core dump for the first time running. The engine will be generated after the first time running. When running the applications again, they will work.

Another workaround is to generate the engines outside the applications. Please r
efer to the TLT tlt-converter tool document: https://docs.nvidia.com/tlt/tlt-user-guide/text/tensorrt.html
