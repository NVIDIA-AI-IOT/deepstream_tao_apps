# DS sample for TAO multi-task model


This is a standalone sample that run single stage multi-task network from TAO without detection.
NVIDIA does not provide any pretrained TAO model for multi-task network. 
To train a multi-task model please check https://docs.nvidia.com/tao/tao-toolkit/text/multitask_image_classification.html.



## Prerequisites


Change following values in deepstream_det_app.c to fit your model input sizes for better visualization of labels on images.

~~~
#define MUXER_OUTPUT_WIDTH 60  
#define MUXER_OUTPUT_HEIGHT 80

#define TILED_OUTPUT_WIDTH 60   
#define TILED_OUTPUT_HEIGHT 80
~~~


### Set output-blob-names in pgie_multi_task_tao_config.txt:


Refer to class_mapping.json which is from model training, this is an example:

~~~
{"tasks": ["base_color", "category", "season"], "class_mapping": {"base_color": {"0": "Black", "1": "Blue", "2": "Brown", "3": "Green", "4": "Grey", "5": "Navy Blue", "6": "Pink", "7": "Purple", "8": "Red", "9": "Silver", "10": "White"}, "category": {"0": "Bags", "1": "Bottomwear", "2": "Eyewear", "3": "Fragrance", "4": "Innerwear", "5": "Jewellery", "6": "Sandal", "7": "Shoes", "8": "Topwear", "9": "Watches"}, "season": {"0": "Fall", "1": "Spring", "2": "Summer", "3": "Winter"}}}

~~~

Then the output-blob-names are base_color/Softmax and category/Softmax and season/Softmax:

~~~
output-blob-names=base_color/Softmax;category/Softmax;season/Softmax
~~~

