# Parser (nvdsinfer_custombboxparser_yolov3_tlt.*)

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

This source has been written to parse the output layers of the YoloV3 model.
To use this library for bounding box output parsing instead of the inbuilt parsing
function, modify the following parameters in [property] section of primary infer
configuration file:

parse-bbox-func-name=NvDsInferParseCustomYOLOV3TLT
custom-lib-path=/path/to/this/directory/libnvds_infercustomparser_yolov3_tlt.so

# Label File (yolov3_labels.txt)

The same as ssd_labels.txt, so please refer to ssd_labels.txt.
