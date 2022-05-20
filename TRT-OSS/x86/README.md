# Build x86 TensorRT OSS Plugin

Below are the steps to build [TensorRT OSS](https://github.com/NVIDIA/TensorRT)  for x86 libnvinfer_plugin.so. For cross-compiling, refer to TensorRT OSS README.

## libnvinfer_plugin.so.8.2.5.1 Provided Here

 **libnvinfer_plugin.so.8.2.5.1** provided in this folder was built with:

> Ubuntu 20.04 LTS  
> cuda-11.6
> cuDNN 8.2.1
> TensorRT 8.2.5.1

**Note**

You can get the prebuild lib using `wget https://nvidia.box.com/shared/static/mwtq4z847uz3v37ba8ntmk3ahfv5fnrm -O libnvinfer_plugin.so.8.2.5.1` if you met some LFS issue.

If the environment is different from above, you **MUST** build the TRT OSS plugin by yourself. 

## Build TensorRT OSS Plugin - libnvinfer_plugin.so

Please refer to the Build Guidance under https://github.com/NVIDIA/TensorRT

*Note:*  
*Make sure the GPU_ARCHS of the GPU you are using is in TensorRT OSS [CMakeLists.txt](https://github.com/NVIDIA/TensorRT/blob/master/CMakeLists.txt#L84). If not, you need to specify "GPU_ARCHS" in the build command.*

### 1. Installl Cmake (>= 3.13)

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

```
git clone -b $TRT_OSS_CHECKOUT_TAG https://github.com/nvidia/TensorRT //check TRT_OSS_CHECKOUT_TAG in the above table
cd TensorRT/
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
cd $TRT_SOURCE
mkdir -p build && cd build
## NOTE: as mentioned above, please make sure your GPU_ARCHS in TRT OSS CMakeLists.txt
## if GPU_ARCHS is not in TRT OSS CMakeLists.txt, add -DGPU_ARCHS=xy as below, for xy, refer to below "How to Get GPU_ARCHS" section
$HOME/install/bin/cmake .. -DGPU_ARCHS=xy  -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
make nvinfer_plugin -j$(nproc)
```

After building ends successfully, libnvinfer_plugin.so* will be generated under `pwd`/out/.

### 3. Replace "libnvinfer_plugin.so*"

```
// backup original libnvinfer_plugin.so.x.y, e.g. libnvinfer_plugin.so.8.0.0
sudo mv /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.p.q ${HOME}/libnvinfer_plugin.so.8.p.q.bak
// only replace the real file, don't touch the link files, e.g. libnvinfer_plugin.so, libnvinfer_plugin.so.8
sudo cp $TRT_SOURCE/`pwd`/out/libnvinfer_plugin.so.8.m.n  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.p.q
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

