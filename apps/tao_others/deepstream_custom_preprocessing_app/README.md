# NV12/RGBA-To-Tensor-Preprocess

## Prerequisites

Please follow instructions in the apps/sample_apps/deepstream-app/README on how
to install the prerequisites for the Deepstream SDK, the DeepStream SDK itself,
and the apps.

You must have the following development packages installed
   GStreamer-1.0
   GStreamer-1.0 Base Plugins
   GStreamer-1.0 gstrtspserver
   X11 client-side library

To install these packages, execute the following command:

```bash
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
libgstrtspserver-1.0-dev libx11-dev
```

## Purpose

This sample demonstrates how to replace the Deepstream native NV12/RGBA->RGB(tensor) scaling, padding and normalization functionality with [roiconvert](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/roiconvert).

## Features

1. The input to nvdspreprocess can be NV12 (Block Linear / Pitch) or RGBA.
2. Support Dgpu and Jetson.
3. Support normalization formula R/G/B_output = (R/G/B - offset_R/G/B) * scale_R/G/B.
4. Support processing full frame/ROI/Objects.
5. Support affine transformation.

## To Compile

```bash
$ cd deepstream_custom_preprocessing_app
$ git submodule update --init --recursive
$ cd nvdspreprocess_lib
$ export CUDA_VER=12.8 #For DS8.0 on both x86 & Jetson, CUDA_VER=12.8
$ make
```

NOTE: To improve performance on specific GPUs, please add "-gencode=arch=compute_xx,code=sm_xx" in Makefile. Computing capability can be found in this link https://developer.nvidia.com/zh-cn/cuda-gpus#compute.

## Usage

### Processing Full Frame(nvdspreprocess + pgie)
Sample:

```bash
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0  filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_1 nvstreammux name=mux batch-size=2 width=1920 height=1080 ! nvdspreprocess config-file=config_preprocess_frame.txt ! nvinfer input-tensor-meta=true config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml ! nvmultistreamtiler ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! nvdsosd ! nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4
```

### Processing ROIs(nvdspreprocess + pgie)
Sample:

```bash
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0  filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_1 nvstreammux name=mux batch-size=2 width=1920 height=1080 ! nvdspreprocess config-file=config_preprocess_roi.txt ! nvinfer input-tensor-meta=true config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml ! nvmultistreamtiler ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! nvdsosd ! nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4
```

### Processing Objects(pgie + nvdspreprocess + sgie)
Sample:

```bash
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0  filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_1 nvstreammux name=mux batch-size=2 width=1920 height=1080  ! nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml ! nvdspreprocess config-file=config_preprocess_sgie.txt ! nvinfer unique-id=4 input-tensor-meta=true config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_secondary_vehicletypes.yml ! nvmultistreamtiler ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! nvdsosd ! nvvideoconvert ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4
```

### Processing RGBA Input To Nvdspreprocess
Sample:

```bash
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg ! jpegparse ! jpegdec  ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg ! jpegparse  !  jpegdec  ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=RGBA' ! mux.sink_1 nvstreammux name=mux batch-size=2 width=1920 height=1080 ! nvdspreprocess config-file=config_preprocess_frame.txt ! nvinfer input-tensor-meta=true  batch-size=2 config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt ! nvmultistreamtiler ! nvdsosd ! nveglglessink
```

### Processing Affine Transformation Tensors
If you want to generate affine transformation tensors, please use get_affine_matrx.py to generate affine matrix first. Taking scaling 1920x1080 to 960x544 and rotating 45 degrees for example, run the followming code.
```bash
$ python3 get_affine_matrx.py #please modify the code if using other values.
```
Then fill scale-type and affine-matrix in the configuration file of nvdspreprocess as shown below.
```bash
scale-type=2
affine-matrix=0.35355339;-0.35355339;331.50757595;0.35355339;0.35355339;-258.33008589
```

## Configurations In [user-configs]

<table>
<thead>
  <tr>
    <th>Configurations</th>
    <th colspan="6">Meaning</th>
    <th colspan="6">Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>pixel-normalization-factor</td>
    <td colspan="6"> works as "scale" in the formula R/G/B_Output = (R/G/B - offset_R/G/B) * scale_R/G/B</td>
    <td colspan="6">pixel-normalization-factor=0.003921568;0.003921568;0.003921568 <br>The default value is 1;1;1.</td>
  </tr>
  <tr>
    <td>offsets</td>
    <td colspan="6">works as "offset" in the formula R/G/B_output = (R/G/B - offset_R/G/B) * scale_R/G/B.</td>
    <td colspan="6">offsets=10;10;10 <br>The default value is 0;0;0.</td>
  </tr>
  <tr>
    <td>scaling-filter</td>
    <td colspan="6">Scaling Interpolation method. 0=Nearest 1=Bilinear 2=Default(Nearest)</td>
    <td colspan="6">scaling-filter=0 <br>The default is 0.</td>
  </tr>
  <tr>
    <td>scale-type</td>
    <td colspan="6">Scaling type. 0: "maintain-aspect-ratio=0", 1: "maintain-aspect-ratio=1 and center" 2: matrix.</td>
    <td colspan="6">scale-type:1 <br>The default is 1.</td>
  </tr>
  <tr>
    <td>affine-matrix</td>
    <td colspan="6">if scale-type is set to 2. affine-matrix is needed.</td>
    <td colspan="6">affine-matrix=1.0;0.0;0.0;0.0;1.0;0.0 <br>The default value is 1.0;0.0;0.0;0.0;1.0;0.0.</td>
  </tr>
</tbody>
</table>
## Performance
The below is a time consumption comparison using the DeepStream native library and the high-performance roiconvert library.

Prerequisites:
|Items              |  values           |
| ----------------  | ----------------  |
|Device             | RTX 6000 & AGX Orin 64G   |
|DeepStream Version |   8.0             |
|Input Format       |   NV12            |
|batch-num range    |  40-45            |
|batch-size         |16(fullframe) 64(ROI)      |
|Power Mode         |MAX perf and power         |


### Processing Full Frame
```bash
#Pipeline for pitch layout
gst-launch-1.0 -v filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0  filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_1 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_2 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_3 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_4 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_5 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_6 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_7 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_8 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_9 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_10 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_11 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_12 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_13 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_14 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_15  nvstreammux name=mux batch-size=16 width=1920 height=1080  ! nvdspreprocess config-file=config_preprocess_frame.txt ! fakesink
#Pipeline for block linear
gst-launch-1.0 -v filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0  filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_1 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_2 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_3 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_4 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_5 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_6 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_7 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_8 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_9 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_10 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_11 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_12 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_13 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_14 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_15  nvstreammux name=mux batch-size=16 width=1280 height=720  ! nvdspreprocess config-file=config_preprocess_frame.txt ! fakesink
```
Perf On Jetson Orin

<table>
  <tr>
    <th>Input</th>
    <th>Function</th>
    <th>native</th>
    <th>roiconvert</th>
  </tr>
  <tr>
   <td>pitchLayout(Nearest)</td>
   <td>
      <table>
         <tr><td>CustomAsyncTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors with batch-size=16</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>5.540</td></tr>
        <tr><td>8.024</td></tr>
        <tr><td>13.564(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>9.174</td></tr>
        <tr><td>9.174(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
  <tr>
   <td>pitchLayout(Bilinear)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors with batch-size=16</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>6.316</td></tr>
        <tr><td>8.219</td></tr>
        <tr><td>14.535(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>12.923</td></tr>
        <tr><td>12.923(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>blockLinear(Nearest)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors with batch-size=16</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>6.085</td></tr>
        <tr><td>7.862</td></tr>
        <tr><td>13.947(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>8.709</td></tr>
        <tr><td>8.709(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>blockLinear(Bilinear)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors with batch-size=16</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>6.031</td></tr>
        <tr><td>7.728</td></tr>
        <tr><td>13.759(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>13.551</td></tr>
        <tr><td>13.551(ms)</td></tr>
      </table>
    </td>
  </tr>
</table>

Perf On Dgpu RTX A6000
<table>
  <tr>
    <th>Input</th>
    <th>Function</th>
    <th>native</th>
    <th>roiconvert</th>
  </tr>
  <tr>
   <td>pitchLayout(Nearest)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors with batch-size=16</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>1.641</td></tr>
        <tr><td>3.749</td></tr>
        <tr><td>5.391(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>4.643</td></tr>
        <tr><td>4.643(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>pitchLayout(Bilinear)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors with batch-size=16</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>1.588</td></tr>
        <tr><td>5.815</td></tr>
        <tr><td>7.403(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>4.597</td></tr>
        <tr><td>4.597(ms)</td></tr>
      </table>
    </td>
  </tr>
</table>

### Processing Objects
```bash
#Pipeline for pitch layout
gst-launch-1.0 -v filesrc location=$object_file ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0  filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_1 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_2 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_3 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_4 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_5 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_6 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_7 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_8 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_9 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_10 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_11 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_12 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_13 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_14 filesrc location=$object_file  ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_15  nvstreammux name=mux batch-size=16 width=1920 height=1080  ! nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml ! nvdspreprocess config-file=config_preprocess_sgie.txt ! fakesink
```
Perf on Jetson Orin
<table>
  <tr>
    <th>Input</th>
    <th>Function</th>
    <th>Native</th>
    <th>roiconvert</th>
  </tr>
  <tr>
   <td>pitchLayout(Nearest)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors for objects</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>9.723</td></tr>
        <tr><td>69.228</td></tr>
        <tr><td>78.951(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>17.596</td></tr>
        <tr><td>17.596(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>pitchLayout(Bilinear)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors for objects</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>15.929</td></tr>
        <tr><td>86.215</td></tr>
        <tr><td>102.144(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>21.906</td></tr>
        <tr><td>21.906(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>blockLinear(Nearest)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors for objects</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>7.347</td></tr>
        <tr><td>11.238</td></tr>
        <tr><td>18.585(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>10.439</td></tr>
        <tr><td>10.439(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>blockLinear(Bilinear)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors for objects</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>12.820</td></tr>
        <tr><td>12.675</td></tr>
        <tr><td>25.495(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>12.502</td></tr>
        <tr><td>12.502(ms)</td></tr>
      </table>
    </td>
  </tr>
</table>

Perf on Dgpu RTX A6000
<table>
  <tr>
    <th>Input</th>
    <th>Function</th>
    <th>Native</th>
    <th>roiconvert</th>
  </tr>
  <tr>
   <td>pitchLayout(Nearest)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors for objects</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>37.743</td></tr>
        <tr><td>21.874</td></tr>
        <tr><td>59.617(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>7.837</td></tr>
        <tr><td>7.837(ms)</td></tr>
      </table>
    </td>
  </tr>
  <tr>
   <td>pitchLayout(Bilinear)</td>
   <td>
      <table>
         <tr><td>CustomTransformation</td></tr>
         <tr><td>CustomTensorPreparation</td></tr>
         <tr><td>Time to generate 6 batches of tensors for objects</td></tr>
      </table>
   </td>
    <td>
      <table>
        <tr><td>20.757</td></tr>
        <tr><td>15.944</td></tr>
        <tr><td>36.701(ms)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><td>0</td></tr>
        <tr><td>17.554</td></tr>
        <tr><td>17.554(ms)</td></tr>
      </table>
    </td>
  </tr>
</table>