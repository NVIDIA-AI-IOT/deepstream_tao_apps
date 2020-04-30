[[_TOC_]]

# Parser (nvdsinfer_custombboxparser_frcnn_tlt.*)

Refer to the DeepStream SDK documentation for a description of the library.

--------------------------------------------------------------------------------
## Pre-requisites

- TensorRT 7.x

--------------------------------------------------------------------------------
## Compiling

```
make
```

--------------------------------------------------------------------------------
This source has been written to parse the output layers of the Faster-RCNN model.
To use this library for bounding box output parsing instead of the inbuilt parsing
function, modify the following parameters in [property] section of primary infer
configuration file:

parse-bbox-func-name=NvDsInferParseCustomFrcnnTLT
custom-lib-path=/path/to/this/directory/libnvds_infercustomparser_frcnn_tlt.so

# Label File (frcnn_labels.txt)

The label file is a text file, containing the names of the classes that the FasterRCNN model is trained to detect. The order where the classes are listed here must match that where the model predicts the output. This order is derived from the order the objects are instantiated in the target_class_mapping field of the FasterRCNN experiment specification file. During the training, TLT FasterRCNN will make all the class names in lower case and sort them in alphabetical order. For example, if the target_class_mapping label file is:

```
  target_class_mapping {
    key: "car"
    value: "car"
  }
  target_class_mapping {
    key: "person"
    value: "person"
  }
  target_class_mapping {
    key: "bicycle"
    value: "bicycle"
  }

```
The corresponding label file is
```
bicycle
car
person
```
