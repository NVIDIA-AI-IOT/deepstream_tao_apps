## Description
The nvOCDR deepstream sample application for optical character detection and recognition.

The TAO pretrained models used in the sample application are:
* [OCDNet deployable_onnx_v2.4](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocdnet)
* [OCRNet deployable_v2.1.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocrnet)

## Prerequisition
* DeepStream SDK 6.2 GA and above

#### **Set up the development environment**:
$git lfs pull

$sudo apt update && sudo apt install -y libopencv-dev

#### **Prepare the nvocdr library**:
Refer to [NVIDIA-Optical-Character-Detection-and-Recognition-Solution](https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution/tree/main)

1.prepare the libnvocdr.so lib
```shell
$ git clone https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution.git
$ cd NVIDIA-Optical-Character-Detection-and-Recognition-Solution
$ make
$ cp libnvocdr.so ../nvocdr_libs/x86/
# On Jetson platform:
# $ cp libnvocdr.so ../nvocdr_libs/aarch64/
```

2.prepare the libnvocdr_impl.so
```shell
$ cd deepstream
$ make
$ cp libnvocdr_impl.so ../../nvocdr_libs/x86/
# On Jetson platform:
# $ cp libnvocdr_impl.so ../../nvocdr_libs/aarch64/
```

#### **Get the TensorRT OSS plugin library (Optional)**:

**Notes: If you're using DeepStream 6.4 and above, you can skip this step.**

1.To avoid affecting the results of other apps, please replace the TensorRT plugin with the original one after running this app.  

2.Please replace the 'x' of libnvinfer_plugin.so.8.x.x in the shell command line with the actual value in your environment.

For X86 platform
```shell
$wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.0/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8_1.0-1_amd64.deb
$dpkg-deb -xv nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8_1.0-1_amd64.deb debs
$cd debs/var/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8
$dpkg-deb  -xv libnvinfer-plugin8_8.6.0.12-1+cuda11.8_amd64.deb deb_file
$cp deb_file/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.0  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.x.x
```

For Jetson platform
- Get TensorRT OSS repository
```shell
git clone -b release/8.6 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
```

- Compile TensorRT `libnvinfer_plugin.so`:
```shell
mkdir build && cd build
cmake .. -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/
make nvinfer_plugin -j4
```

- Copy the `libnvinfer_plugin.so` to the system library path
```shell
cp libnvinfer_plugin.so.8.6.x /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.x.x
```

## Build And Run
The application can be build and run seperately.

```
1.
Go to the deepstream-nvocdr-app directory

2.
#Generate OCDNet engine with dynmaic batch size and max batch size is 4:
/usr/src/tensorrt/bin/trtexec --onnx=../../../models/nvocdr/ocdnet.onnx --minShapes=input:1x3x736x1280 --optShapes=input:1x3x736x1280 --maxShapes=input:1x3x736x1280 --fp16 --saveEngine=../../../models/nvocdr/ocdnet.fp16.engine

#Generate OCRNet engine with dynamic batch size and max batch size is 32:
/usr/src/tensorrt/bin/trtexec --onnx=../../../models/nvocdr/ocrnet.onnx --minShapes=input:1x1x64x200 --optShapes=input:32x1x64x200 --maxShapes=input:32x1x64x200 --fp16 --saveEngine=../../../models/nvocdr/ocrnet.fp16.engine

3.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./nvocdr_libs/x86
# On Jetson platform:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./nvocdr_libs/aarch64
make

4.
# Set the 'source-list' as the path of your sources in the nvocdr_app_config.yml
# On Jetson platform:
# modify the 'customlib-name' field in the nvocdr_app_config.yml first
./deepstream-nvocdr-app nvocdr_app_config.yml
```