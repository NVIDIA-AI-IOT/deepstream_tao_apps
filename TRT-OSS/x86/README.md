# Build x86 TensorRT OSS Plugin

For DeepStream 6.1.1 GA, 6.2 GA and 6.3 GA, the TensorRT OSS plugin is not needed.

Below are the steps to build [TensorRT OSS](https://github.com/NVIDIA/TensorRT)  for x86 libnvinfer_plugin.so. For cross-compiling, refer to TensorRT OSS README.

## libnvinfer_plugin.so.8.6.2 provided Here

 **libnvinfer_plugin.so.8.6.2** provided in this folder was built with:

> Ubuntu 22.04 LTS  
> cuda-12.1
> cuDNN 8.9.3
> TensorRT 8.6.2

**Note**

You can get the prebuild lib using `wget https://nvidia.box.com/shared/static/eyqxd1g5kya51wk76i3st5e3m3xhyyfq libnvinfer_plugin.so.8.6.2` if you met some LFS issue.

If the environment is different from above, you **MUST** build the TRT OSS plugin by yourself. 

## Build TensorRT OSS Plugin - libnvinfer_plugin.so

Please refer to the Build Guidance under https://github.com/NVIDIA/TensorRT

*Note:*  
*Make sure the GPU_ARCHS of the GPU you are using is in TensorRT OSS [CMakeLists.txt](https://github.com/NVIDIA/TensorRT/blob/master/CMakeLists.txt#L84). If not, you need to specify "GPU_ARCHS" in the build command.*

### 1. Installl Cmake (>= 3.13)

For TensorRT8.6, please ignore this step. 

TensorRT OSS requires cmake >= v3.13, so install cmake 3.13 if your cmake version is lower than 3.13

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
| DeepStream Release  | TRT Version     | TRT_OSS_CHECKOUT_TAG  |
| ------------------- | --------------- | --------------------- |
| 5.0                 | TRT 7.0.0       | release/7.0           |
| 5.0.1               | TRT 7.0.0       | release/7.0           |
| 5.1                 | TRT 7.2.X       | 21.03                 |
| 6.0 EA              | TRT 7.2.2       | 21.03                 |
| 6.0 GA              | TRT 8.0.1       | release/8.0           |
| 6.0.1               | TRT 8.2.1       | release/8.2           |
| 6.1                 | TRT 8.2.5.1     | release/8.2           |
| 6.1.1 GA            | TRT 8.4.1.11    | no OSS plugin is needed |
| 6.2 GA              | TRT 8.5.1       | no OSS plugin is needed |
| 6.3                 | TRT 8.5.3       | no OSS plugin is needed |
| 6.4                 | TRT 8.6.2       | binary plugin only    |
| 7.0                 | TRT 8.6.2       | binary plugin only    |


### 3. Replace "libnvinfer_plugin.so*"

```
// backup original libnvinfer_plugin.so.x.y, e.g. libnvinfer_plugin.so.8.0.0
sudo mv /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.1 ${HOME}/libnvinfer_plugin.so.8.6.1.bak
// only replace the real file, don't touch the link files, e.g. libnvinfer_plugin.so, libnvinfer_plugin.so.8
sudo cp TRT-OSS/x86/TRT8.6/libnvinfer_plugin.so.8.6.2  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.1
sudo rm /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
sudo rm /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8
sudo ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.1 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8
sudo ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
sudo ldconfig
```

## How to Get GPU_ARCHS

Can use either method to get GPU_ARCHs
1. GPU_ARCHS value can be got by "deviceQuery" CUDA sample 

```
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

2. If there is not "/usr/local/cuda/samples" in your system, you could use the deviceQuery.cpp in this folder,

```
nvcc deviceQuery.cpp -o deviceQuery
./deviceQuery
```

There will be output like below, which indicates the "GPU_ARCHS" is **75**.

```
./deviceQuery

Detected 2 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    7.5
```

