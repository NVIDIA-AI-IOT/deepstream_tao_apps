[[_TOC_]]
# Build Jetson TensorRT OSS Plugin

Below are the steps to build [TensorRT OSS](https://github.com/NVIDIA/TensorRT) on Jetson device for Jetson libnvinfer_plugin.so. For cross-compiling, refer to TensorRT OSS README.

## libnvinfer_plugin.so.7.0.0.1 Provided Here

 **libnvinfer_plugin.so.7.0.0.1** provided in this folder was built with:

> Jetson Xavier  
> Jetpack4.4DP (CUDA-10.2, cuDNN v8.0, TensorRT 7.1.0.16)

## Build TensorRT OSS Plugin - libnvinfer_plugin.so

### 1. Upgrade Cmake

TensorRT OSS requires cmake \>= v3.13, while the default cmake on Jetson/UBuntu 18.04 is cmake 3.10.2, so upgrade it by

```
sudo apt remove --purge --auto-remove cmake
wget https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5.tar.gz
tar xvf cmake-3.13.5.tar.gz
cd cmake-3.13.5/
./configure
make -j$(nproc)
sudo make install
sudo ln -s /usr/local/bin/cmake /usr/bin/cmake
```

### 2. Build TensorRT OSS Plugin

```
git clone -b release/7.0 https://github.com/nvidia/TensorRT
cd TensorRT/
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
cd $TRT_SOURCE
mkdir -p build && cd build
### NOTE: below -DGPU_ARCHS=72 is for Xavier or NX, for other Jetson platform, please change "72" referring to below "GPU_ARCH" section
/usr/local/bin/cmake .. -DGPU_ARCHS=72  -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
make nvinfer_plugin -j$(nproc)
```

After building ends successfully, libnvinfer_plugin.so* will be generated under `pwd`/out/.

### 3. Replace "libnvinfer_plugin.so*"

```
sudo mv /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.x.y ${HOME}/libnvinfer_plugin.so.7.x.y.bak   // backup original libnvinfer_plugin.so.x.y
sudo cp `pwd`/out/libnvinfer_plugin.so.7.m.n  /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.x.y
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

