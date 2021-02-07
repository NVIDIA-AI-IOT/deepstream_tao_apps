/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

extern int seg_main(int, char **);
extern int det_main(int, char **);

enum class ModelType {
    segmentation,
    detection,
    unknown
};

int
main (int argc, char *argv[]) {
    const string unet{"unet"};
    const vector<string> det_models{"frcnn", "ssd", "dssd", "yolov3", "yolov4"
        , "peopleSegNet", "retina"};
    vector<string> str_argv(&argv[1], argv+argc);

    auto checkModelType = [&unet, &det_models](vector<string> & vec_str) ->ModelType {
        for(string &str : vec_str) {
            if(string::npos != str.find(unet))
                return ModelType::segmentation;
            for(auto &det_model : det_models){
                if(string::npos != str.find(det_model))
                    return ModelType::detection;
            }
        }
        return ModelType::unknown;
    };

    auto usageInfo = [](int argc, char **argv) {
        cout << "For detection model:"<<endl;
        det_main(argc, argv);
        cout << "For segmentation model:"<<endl;
        seg_main(argc, argv);
    };

    if(argc == 1){
        usageInfo(argc, argv);
        return 0;
    }

    switch(checkModelType(str_argv)){
        case ModelType::segmentation:
            seg_main(argc, argv);
            return 0;
        case ModelType::detection:
            det_main(argc, argv);
            return 0;
        case ModelType::unknown:
        default:
            usageInfo(argc, argv);
            return 0;
    }
}
