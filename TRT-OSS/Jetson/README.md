# Build Jetson TensorRT OSS Plugin

Below are the steps to build [TensorRT OSS](https://github.com/NVIDIA/TensorRT) on Jetson device for Jetson libnvinfer_plugin.so. For cross-compiling, refer to TensorRT OSS README.

## libnvinfer_plugin.so.8.4.0.11 Provided Here

 **libnvinfer_plugin.so.8.4.0.11** provided in this folder was built with:

> Jetson Xavier
> Jetpack5.0.1 (CUDA-11.6, cuDNN v8.2, TensorRT8.4.0.11 )

**Note**

You can get the prebuild lib by using "wget https://nvidia.box.com/shared/static/gcp6ylk1ku0zfobhj0sv8vpraz6yzaf9 -O libnvinfer_plugin.so.8.4.0.11" if you met some LFS issue.


## Build TensorRT OSS Plugin - libnvinfer_plugin.so

### 1. Upgrade Cmake

TensorRT OSS requires cmake \>= v3.13, while the default cmake on Jetson/UBuntu 20.04 is cmake 3.10.2, so upgrade it by

```
wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4.tar.gz
tar xvf cmake-3.19.4.tar.gz
cd cmake-3.19.4/
mkdir $HOME/install
./configure --prefix=$HOME/install
make -j$(nproc)
sudo make install
```

### 2. Build TensorRT OSS Plugin

| DeepStream Release  | Jetpack Version  | TRT Version     | TRT_OSS_CHECKOUT_TAG  |
| ------------------- | ---------------  | --------------- | --------------------- |
| 5.0                 | 4.4 GA  - 4.5    | TRT 7.1.3       | release/7.1           |
| 5.0.1               | 4.4 GA - 4.5     | TRT 7.1.3       | release/7.1           |
| 5.1                 | 4.5.1            | TRT 7.1.3       | release/7.1           |
| 6.0 EA              | 4.5.1            | TRT 7.1.3       | release/7.1           |
| 6.0 GA              | 4.6              | TRT 8.0.1       | release/8.0           |
| 6.0.1               | 4.6.1            | TRT 8.2.1       | release/8.2           |
| 6.1                 | 5.0.1            | TRT 8.4.0.11    | release/8.4           |

```
git clone -b $TRT_OSS_CHECKOUT_TAG https://github.com/nvidia/TensorRT        // replace with release/8.x for  TensorRT 8.X
cd TensorRT/
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
cd $TRT_SOURCE
mkdir -p build && cd build
$HOME/install/bin/cmake .. -DGPU_ARCHS="53 62 72"  -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
make nvinfer_plugin -j$(nproc)
```

After building ends successfully, libnvinfer_plugin.so* will be generated under `pwd`/out/.

### 3. Replace "libnvinfer_plugin.so*"

```
sudo mv /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.x.y ${HOME}/libnvinfer_plugin.so.8.x.y.bak   // backup original libnvinfer_plugin.so.x.y
sudo cp $TRT_SOURCE/build/libnvinfer_plugin.so.8.m.n  /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.x.y
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

