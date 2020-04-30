[[_TOC_]]
# Parser (nvdsinfer_custombboxparser_ssd_tlt.*)

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
This source has been written to parse the output layers of the ssd model detector.
To use this library for bounding box output parsing instead of the inbuilt parsing
function, modify the following parameters in [property] section of primary infer
configuration file:

parse-bbox-func-name=NvDsInferParseCustomSSDTLT
custom-lib-path=/path/to/this/directory/libnvds_infercustomparser_ssd_tlt.so

# Label File (ssd_labels.txt)

The order in which the classes are listed here must match the order in which the model predicts the output. This order is derived from the order in which the objects are instantiated in the `dataset_config` field of the SSD experiment config file as mentioned in Transfer Learning Toolkit user guide. For example, if the `dataset_config` is like this:

```
dataset_config {
  data_sources {
    ...
  }
  data_sources {
    ...
  }
  validation_fold: 0
  image_extension: "jpg"
  target_class_mapping {
    key: "AutoMobile"
    value: "car"
  }
  target_class_mapping {
    key: "Automobile"
    value: "car"
  }
  target_class_mapping {
    key: "Bicycle"
    value: "bicycle"
  }
  target_class_mapping {
    key: "Heavy Truck"
    value: "car"
  }
  target_class_mapping {
    key: "Motorcycle"
    value: "bicycle"
  }
  target_class_mapping {
    key: "Person"
    value: "person"
  }

  ...

  }
  target_class_mapping {
    key: "traffic_light"
    value: "road_sign"
  }
  target_class_mapping {
    key: "twowheeler"
    value: "bicycle"
  }
  target_class_mapping {
    key: "vehicle"
    value: "car"
  }
}
```

The corresponding label file will be

```
bicycle
car
person
road_sign
```
