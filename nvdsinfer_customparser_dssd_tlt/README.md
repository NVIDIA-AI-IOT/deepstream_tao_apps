[[_TOC_]]
# Parser (nvdsinfer_custombboxparser_dssd_tlt.*)

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
This source has been written to parse the output layers of the ssd tlt detector.
To use this library for bounding box output parsing instead of the inbuilt parsing
function, modify the following parameters in [property] section of primary infer
configuration file:

parse-bbox-func-name=NvDsInferParseCustomSSDTLT
custom-lib-path=/path/to/this/directory/libnvds_infercustomparser_dssd_tlt.so

# Label File (dssd_labels.txt)

The same as ssd_labels.txt, so please refer to ssd_labels.txt and the README.md under nvdsinfer_customparser_ssd_tlt.

