# Build Jetson TensorRT OSS Plugin

Below are the steps to build [TensorRT OSS](https://github.com/NVIDIA/TensorRT) on Jetson device for Jetson libnvinfer_plugin.so. For cross-compiling, refer to TensorRT OSS README.

## libnvinfer_plugin.so.7.1.3 Provided Here

 **libnvinfer_plugin.so.7.1.3** provided in this folder was built with:

> Jetson NX  
> Jetpack4.4GA (CUDA-10.2, cuDNN v8.0, TensorRT 7.1.3)

**Note**

You can get teh prebuild lib using `wget https://nvidia.box.com/shared/static/ezrjriq08q8fy8tvqcswgi0u6yn0bomg.1 -O libnvinfer_plugin.so.7.0.0.1` if you met some LFS issue.

## Build TensorRT OSS Plugin - libnvinfer_plugin.so

### 1. Upgrade Cmake

TensorRT OSS requires cmake \>= v3.13, while the default cmake on Jetson/UBuntu 18.04 is cmake 3.10.2, so upgrade it by

```
sudo dpkg --force-all -r cmake
wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4.tar.gz
tar xvf cmake-3.19.4.tar.gz
cd cmake-3.19.4/
./configure
make -j$(nproc)
sudo make install
sudo ln -s /usr/local/bin/cmake /usr/bin/cmake
```

### 2. Build TensorRT OSS Plugin

```
git clone -b release/7.1 https://github.com/nvidia/TensorRT
cd TensorRT/
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
cd $TRT_SOURCE
mkdir -p build && cd build
/usr/local/bin/cmake .. -DGPU_ARCHS="53 62 72" -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
make nvinfer_plugin -j$(nproc)
```

After building ends successfully, libnvinfer_plugin.so* will be generated under `pwd`/out/.

### 3. Replace "libnvinfer_plugin.so*"

```
sudo mv /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.x.y ${HOME}/libnvinfer_plugin.so.7.x.y.bak   // backup original libnvinfer_plugin.so.x.y
sudo cp `pwd`/out/libnvinfer_plugin.so.7.m.n  /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.x.y
sudo ldconfig
```

### Known issues

If you ran into "[ERROR] IPluginV2DynamicExt requires network without implicit batch dimension" while running yolov3 model, please try to use the libnvinfer_plugin.so.7.1.3_nano_tx2_xavier_nx_for_yolov3
