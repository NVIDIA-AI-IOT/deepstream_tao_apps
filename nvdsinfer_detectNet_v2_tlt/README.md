[[_TOC_]]

# Parser

detectnet_v2 parser is already in DeepStream release package.

# Label File (detectnet_v2_labels.txt)

The label file is a text file, containing the names of the classes that the DetectNet_v2 model is trained to detect. The order in which the classes are listed here must match the order in which the model predicts the output. This order is derived from the order the objects are instantiated in the cost_function_config field of the DetectNet_v2 experiment config file. Here's an example of notebook file of the DetectNet_v2 sample in TLT docker image, the cost_function_config parameter looks like this:

```
cost_function_config {
  target_classes {
    name: "sheep"
    class_weight: 1.0
    coverage_foreground_weight: 0.05
    objectives {
      name: "cov"
      initial_weight: 1.0
      weight_target: 1.0
    }
    objectives {
      name: "bbox"
      initial_weight: 10.0
      weight_target: 1.0
    }
  }
  target_classes {
    name: "bottle"
    class_weight: 1.0
    coverage_foreground_weight: 0.05
    objectives {
      name: "cov"
      initial_weight: 1.0
      weight_target: 1.0
    }
    objectives {
      name: "bbox"
      initial_weight: 10.0
      weight_target: 1.0
    }
  }
  target_classes {
    name: "horse"
    class_weight: 1.0
    coverage_foreground_weight: 0.05
    objectives {
      name: "cov"
      initial_weight: 1.0
      weight_target: 1.0
    }
    objectives {
      name: "bbox"
      initial_weight: 10.0
      weight_target: 1.0
    }
  }
  ..
  ..
  target_classes {
    name: "boat"
    class_weight: 1.0
    coverage_foreground_weight: 0.05
    objectives {
      name: "cov"
      initial_weight: 1.0
      weight_target: 1.0
    }
    objectives {
      name: "bbox"
      initial_weight: 10.0
      weight_target: 1.0
    }
  }
  target_classes {
    name: "car"
    class_weight: 1.0
    coverage_foreground_weight: 0.05
    objectives {
      name: "cov"
      initial_weight: 1.0
      weight_target: 1.0
    }
    objectives {
      name: "bbox"
      initial_weight: 10.0
      weight_target: 1.0
    }
  }
  enable_autoweighting: False
  max_objective_weight: 0.9999
  min_objective_weight: 0.0001
}
```

Here's an example of the corresponding, classification_labels.txt:

```
sheep
bottle
horse
..
..
boat
car
```