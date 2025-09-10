This document describes the procedure to download and run the TAO pre-trained purpose-built models in DeepStream.

The following pre-trained models are provided:

- VehicleMakeNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:vehiclemakenet)
- VehicleTypeNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:vehicletypenet)
- TrafficeCamNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet)
- PeopleNet (https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplenet)
- PeopleNet Transformer v2 (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer_v2)

*******************************************************************************************
## 1. Download the config files

*******************************************************************************************
```bash
$ git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
$ cd deepstream_tao_apps/deepstream_app_tao_configs/
$ sudo apt install -y wget zip
```

*******************************************************************************************
## 2. Prepare Pretrained Models
*******************************************************************************************
Choose one of the following three inferencing methods:
- For TensorRT based inferencing, please run the following commands

```bash
$ sudo cp -a * /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
$ cd /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/
$ sudo ./download_models.sh
```

- For Triton Inference Server based inferencing, the DeepStream application works as the Triton client:
  * To set up the native Triton Inference Sever, please refer to [triton_server](triton_server.md).

  * To set up the separated Triton Inference Sever, please refer to [triton_server_grpc](triton_server_grpc.md)

*******************************************************************************
## 3. Run the models in DeepStream
*******************************************************************************
```bash
$ sudo deepstream-app -c deepstream_app_source1_$MODEL.txt
```
e.g.
```bash
$ sudo deepstream-app -c deepstream_app_source1_trafficcamnet_vehiclemakenet_vehicletypenet.txt
```

The yaml config files can also be used
```bash
$ sudo deepstream-app -c deepstream_app_source1_$MODEL.yml
```
e.g.
```bash
$ sudo deepstream-app -c deepstream_app_source1_trafficcamnet_vehiclemakenet_vehicletypenet.yml
```

**Note:**

1. For which model of the *deepstream_app_source1_$MODEL.txt* uses, please find from the **[primary-gie]** section in it, for example

   Below is the **[primary-gie]** config of deepstream_app_source1_peoplenet.txt, which indicates it uses peoplenet model by default, and user can change to peoplenet_transformer_v2 by commenting "config-file="config-file=nvinfer/config_infer_primary_peoplenet.txt" and uncommenting the corresponding "config-file=" .

   ```ini
   [primary-gie]
   enable=1
   #(0): nvinfer; (1): nvinferserver
   plugin-type=0
   gpu-id=0
   # Modify as necessary
   batch-size=1
   #Required by the app for OSD, not a plugin property
   bbox-border-color0=1;0;0;1
   bbox-border-color1=0;1;1;1
   bbox-border-color2=0;0;1;1
   bbox-border-color3=0;1;0;1
   gie-unique-id=1
   config-file=nvinfer/config_infer_primary_peoplenet.txt
   #config-file=nvinfer/config_infer_primary_peoplenet_transformer_v2.txt
   #config-file=triton/config_infer_primary_peoplenet.txt
   #config-file=triton/config_infer_primary_peoplenet_transformer_v2.txt
   #config-file=triton-grpc/config_infer_primary_peoplenet.txt
   #config-file=triton-grpc/config_infer_primary_peoplenet_transformer_v2.txt
   ```

2. The GIE can be set to nvinfer or nvinferserver with the configuration file. For Triton grpc mode, the DeepStream application should run in different machine or terminal from the server.

*******************************************************************************
## 4. Related Links
*******************************************************************************
deepstream-tao-app : https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps 

TAO Toolkit Guide : https://docs.nvidia.com/tao/tao-toolkit/index.html
