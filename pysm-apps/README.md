- [1.Description](#1description)
- [2.Prerequisites](#2prerequisites)
- [3.Download Models](#3download-models)
- [4.Prepare Python Environment](#4prepare-python-environment)
- [5.Detection & Instance segmention Sample](#5detection--instance-segmention-sample)
- [6.Semantic Segmention Sample](#6semantic-segmention-sample)
- [7.Car License Recognization Sample](#7car-license-recognization-sample)
- [8.Optical Character Detection And Recognition Sample](#8optical-character-detection-and-recognition-sample)
- [9.MDX Perception Sample Application Sample](#9mdx-perception-sample-application-sample)

=====================================================================

### 1.Description

This document describes how to use Pyservicemaker's Flow API to quickly build applications for TAO pre-trained models.

The following pyservicemaker applications with TAO models are provided:

- [Detection & Instance Segmention Sample](tao_detection/deepstream_det_app.py)

- [Semantic Segmention Sample](tao_segmentation/deepstream_seg_app.py)

- [MDX sample with embedding Models](tao_others/deepstream-mdx-perception-app/deepstream_mdx_perception_app.py)

- [OCD/OCR sample](tao_others/deepstream-nvocdr-app/deepstream_nvocdr_app.py)

- [Pose Classification Sample](tao_others/deepstream-pose-classification/deepstream_pose_classification_app.py)

- [Car License Plate Recognition Sample](tao_others/deepstream_lpr_app/deepstream_lpr_app.py)

### 2.Prerequisites

* [DeepStream SDK 8.0 GA](https://developer.nvidia.com/deepstream-sdk)

    Make sure deepstream-test1 sample can run successful to verify your installation. According to the
    [document](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html),
    please run below command to install additional audio video packages.

```bash
    # install pyservicemaker wheel
    /opt/nvidia/deepstream/deepstream/install.sh
    # install deepstream dependencies
    /opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

* Eigen development packages
```bash
    sudo apt install libeigen3-dev
    cd /usr/include
    sudo ln -sf eigen3/Eigen Eigen
```

### 3.Download Models

Run below script to download models

```bash
$ git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git

# build post processor libraries
$ make

# download tao models
$ ./download_models.sh
```

### 4.Prepare Python Environment

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a venv for pyservicemaker
uv venv pysmvenv

# Activate the pysmvenv environment
source pysmvenv/bin/activate

# install numpy & opencv-python
uv pip install numpy opencv-python cuda-python
```

### 5.Detection & Instance Segmention Sample

Run TAO Detection Pyservicemaker Application

```bash
cd pysm-apps/tao_detection
python3 deepstream_det_app.py ../../configs/app/det_app_config.yml
```

Run TAO Instance Segmention Application

```bash
SHOW_MASK=1 python3 deepstream_det_app.py ../../configs/app/ins_seg_app.yml
```

The application saves the detection results as `pysm_det_output.mp4`

### 6.Semantic Segmention Sample

Run TAO Semantic Segmention Application

```bash
cd pysm-apps/tao_segmentation
python3 deepstream_seg_app.py ../../configs/app/seg_app_config.yml
```

The application will save the semantic segmentation results to `pysm_seg_output.mp4`, and save the segmentation meta as an image every 100 frames. At the same time, the `FPS` will be output in the terminal.

### 7.Car License Recognization Sample

Run Car License Recognization Application

A sample of US car plate recognition:

```shell
cd pysm-apps/deepstream_lpr_app
cp  ../../../apps/tao_others/deepstream_lpr_app/dict_us.txt dict.txt
python3 deepstream_lpr_app.py ../../../configs/app/lpr_app_us_config.yml
```

A sample of Chinese car plate recognition:

```shell
cp  ../../../apps/tao_others/deepstream_lpr_app/dict_ch.txt dict.txt
python3 deepstream_lpr_app.py ../../../configs/app/lpr_app_ch_config.yml
```
Note: please refer to ../../../apps/tao_others/deepstream_lpr_app/README.md for more details.

### 8.Optical Character Detection And Recognition Sample

Run nvOCDR Application

```shell
cd pysm-apps/deepstream-nvocdr-app
python3 deepstream_nvocdr_app.py  ../../../configs/app/nvocdr_app_config.yml
```

Note: please refer to ../../../apps/tao_others/deepstream-nvocdr-app/README.md for more details.

### 9.MDX Perception Sample Application Sample

Run nvOCDR Application

```shell
cd pysm-apps/deepstream-mdx-perception-app
python3 deepstream_mdx_perception_app.py ../../../configs/app/peoplenet_reidentification.yml
python3 deepstream_mdx_perception_app.py ../../../configs/app/retail_object_detection_recognition.yml
```

Note: please refer to ../../../apps/tao_others/deepstream-mdx-perception-app/README.md for more details.
