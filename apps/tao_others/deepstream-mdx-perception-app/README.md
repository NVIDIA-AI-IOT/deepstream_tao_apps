## Description
The MDX perception sample application drives two Deepstream pipelines, i.e. retail 
item recognition and people ReID. Retail item recognition pipeline detects retail 
items from a video and extracts the embedding vector out of every detection bounding box. 
The embedding vector can form a query to a database of embedding vectors and find
the closest match.
People ReID detects people from a video and extracts the embedding vector out of 
every detection bounding box. The pipelines have a primary GIE module detecting
the objects of interest from a video frame. The secondary GIE module extracts an
embedding vector from the primary GIE result.

The TAO 4.0 pretrained models used in this sample application:

* [Retail Object Detection 100 classes](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_detection).
* [Retail Object Detection Binary](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_detection)
* [Retail Object Recognition](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition)
* [ReIdentificationNet Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet)
* [PeopleNet Transformer Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer)

## Prerequisition

* DeepStream SDK 6.2 GA and above

* NvDsInferParseCustomDDETRTAO
The custom post-processing plugin for Deformable DETR. The source code of tis plugin 
is included in `post_processor/nvdsinfer_custombboxparser_tao.cpp`.

* NvDsInferParseCustomEfficientDetTAO
The custom post-processing plugin for EfficientDet architecture. The source code 
of tis plugin is included in `post_processor/nvdsinfer_custombboxparser_tao.cpp`.

## Application Pipeline
The application pipeline graph

![MDX perception application pipeline](mdx_perception_pipeline.png)

## Build And Run
The application can be build and run seperately. Download the pre-trained models if haven't.
```
export DS_TAO_APPS_HOME=<path to this repo>
cd $DS_TAO_APPS_HOME
download_models.sh
```

Build the applications and run to inference one picture.
```
cd $DS_TAO_APPS_HOME/apps/tao_others/deepstream-mdx-perception-app
make
./deepstream-mdx-perception-app  -c ../../../configs/app/peoplenet_reidentification.yml
./deepstream-mdx-perception-app  -c ../../../configs/app/retail_object_detection_recognition.yml
```
