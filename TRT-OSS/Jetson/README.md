# Build Jetson TensorRT OSS Plugin

For DeepStream 6.1.1 GA and 6.2 GA, the TensorRT OSS plugin is not needed.

Below are the steps to build [TensorRT OSS](https://github.com/NVIDIA/TensorRT) on Jetson device for Jetson libnvinfer_plugin.so. For cross-compiling, refer to TensorRT OSS README.

## libnvinfer_plugin.so.8.6.2 Provided Here

 **libnvinfer_plugin.so.8.6.2** provided in this folder was built with:

> Jetson Xavier
> Jetpack GA (CUDA-12.2, cuDNN v8.6, TensorRT8.6.1.2 )

**Note**

You can get the prebuild lib by using "wget https://nvidia.box.com/shared/static/1ih810ui4z52nvwznfk1xaxvdej3mauw -O libnvinfer_plugin.so.8.6.2" if you met some LFS issue.


## Build TensorRT OSS Plugin - libnvinfer_plugin.so

The TensorRT OSS source code does not contains the patch for 8.6 branch
### Get TensorRT OSS Plugin Library

| DeepStream Release  | Jetpack Version  | TRT Version     | TRT_OSS_CHECKOUT_TAG  |
| ------------------- | ---------------  | --------------- | --------------------- |
| 5.0                 | 4.4 GA  - 4.5    | TRT 7.1.3       | release/7.1           |
| 5.0.1               | 4.4 GA - 4.5     | TRT 7.1.3       | release/7.1           |
| 5.1                 | 4.5.1            | TRT 7.1.3       | release/7.1           |
| 6.0 EA              | 4.5.1            | TRT 7.1.3       | release/7.1           |
| 6.0 GA              | 4.6              | TRT 8.0.1       | release/8.0           |
| 6.0.1               | 4.6.1 / 4.6.2    | TRT 8.2.1       | release/8.2           |
| 6.1                 | 5.0.1            | TRT 8.4.0.11    | release/8.4           |
| 6.1.1               | 5.0.2            | TRT 8.4.1       | OSS not needed        |
| 6.2                 | 5.1              | TRT 8.5.1       | OSS not needed        |
| 6.3                 | 5.1.2            | TRT 8.5.3       | OSS not needed        |
| 6.4                 | 6.0              | TRT 8.6.1.2     | binary only   |
| 7.0                 | 6.0              | TRT 8.6.1.2     | binary only   |
```

### Replace "libnvinfer_plugin.so*"

```
sudo mv /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.x.y ${HOME}/libnvinfer_plugin.so.8.x.y.bak   // backup original libnvinfer_plugin.so.x.y
sudo cp $DOWNLOAD_PATH/libnvinfer_plugin.so.8.m.n  /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.x.y
sudo ldconfig
```

## GPU_ARCHS

1. GPU_ARCHs value can be got from "deviceQuery" CUDA sample 

```
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

2. Can also find GPU_ARCHs from below table

| Jetson Platform | GPU_ARCHS |
| --------------- | --------- |
| TX1 / NANO      | 53        |
| TX2             | 62        |
| Xavier / NX     | 72        |
| Orin            | 87        |

