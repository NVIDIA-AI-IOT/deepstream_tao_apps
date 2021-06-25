## Description
The bodypose2D sample application uses bodypose2D model to detect human body parts coordinates. The application can output the 18 body parts:
- nose
- neck
- right shoulder
- right elbow
- right hand
- left shoulder
- left elbow
- left hand
- right hip
- right knee
- right foot
- left hip
- left knee
- left foot
- right eye
- left eye
- right ear 
- left ear

## Model

The bodypose2D backbone is provided by TLT 3.0 [bodypose 2D estimation](https://ngc.nvidia.com/catalog/models/nvidia:tlt_bodyposenet). 
  
There is blog to introduce how to train and optimize the bodypose 2D estimation model:
  
https://developer.nvidia.com/blog/training-optimizing-2d-pose-estimation-model-with-tlt-part-1/

## Prerequisition

* DeepStream SDK 6.0 and above
  Current DeepStream 6.0 EA version is available in https://developer.nvidia.com/deepstream-sdk-6.0-members-page for specific users.

## Application Pipeline
The application pipeline graph

![bodypose2D application pipeline](bodypose2d_pipeline.png)

## Build And Run
The application can be build and run seperately.

```
cd apps/tlt_others/deepstream-bodypose2d-app
```

For Jetson platform
```
export CUDA_VER=10.2
```

For dGPU
```
export CUDA_VER=11.1
```

Build the applications and run to inference one picture.
```
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
./deepstream-bodypose2d-app 2 ../../../configs/bodypose2d_tlt/sample_bodypose2d_model_config.txt file:///usr/data/bodypose2d_test.png ./body2dout
```

## Post_processing Parameters
The bodypose2D sample application can config post-processing parameters with configuration file. There is a sample of the configuration file "sample_bodypose2d_model_config.txt".
The definition and explanation of the parameters are in following table:

Post-process parameters
    
|Parameter | Type | Description / Instructions| Default |
|----------|------|---------------------------|---------|
|pafMapWidth|int|Width of the part-affinity feature map output. This will be equal to 1/2 of the network input width. For example, if network input width is 384, pafMapWidth will be equal to 192 (Post processing would be done at ½ resolution of the input size)|192|
|pafMapHeight|int|Height of the part-affinity feature map output. This will be equal to 1/2 of the network input height. For example, if network input height is 288, pafMapWidth will be equal to 144|144|
|pafMapChannels|int|Number of channels in part-affinity feature map output. This corresponds to the (number of skeleton connection / edges * 2)|38|
|heatMapWidth|int|Width of the heatmap output. This will be equal to 1/8 of the network input width. For example, if network input width is 384, pafMapWidth will be equal to 48|48|
|heatMapHeight|int|Height of the heatmap output. This will be equal to 1/8 of the network input height. For example, if network input height is 288, pafMapWidth will be equal to 36|36|
|heatMapChannels|int|Number of channels in heatmap output. This corresponds to (number of keypoints + 1)|19|
|featUpsamplingFactor|int|Upsampling factor to use for heatmap to match with the part affinity maps output. For instance, here heatmap output is 1/8th of the network input, where as part affinity map is 1/2th of the network input, so the upsampling factor is 4|4|
|nmsWindowSize|int|Size of the window (kernel size) to be used for Non-max suppression.|7|
|threshHeat|float|Threshold value to use for filtering peaks (heatmap) after Non-max suppression|0.05|
|threshVectorScore|float|Threshold value to use for suppressing connections in part affinity fields|0.05|
|threshVectorCnt1|int|Threshold value for number of qualified points out of all the sampled points along each connection. By default, 10 points are sampled along each connection to verify if it qualifies as a connection candidate.|8|
|threshPartCnt|int|Minimum number of parts needed to qualify as a successful detection.|4|
|threshHumanScore|float|Minimum overall confidence score needed to qualify as a successful detection|0.4|

Please make sure the model related parameters in postprocessing config file match the parameters in nvinfer config file. E.G. the "batchSize" value in postprocessing config file should be the same as the "batch-size" value in nvinfer config file.

## Model parameters

Please refer to /opt/nvidia/deepstream/deepstream/sources/includes/cvcore_headers/cv/core/Model.h

The model parameters can be modified according to the actual model.
