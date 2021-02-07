################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2019-2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import argparse
import os
import xml.etree.ElementTree as ET

def parse_args(args=None):
    parser = argparse.ArgumentParser('Converting xml labels to KITTI format.')
    parser.add_argument('-i', '--input_label_dir', type=str, required=True, help='directory of the input xml labels')
    parser.add_argument('-o', '--output_label_dir', type=str, required=True, help='directory of the output KITTI labels')
    parser.add_argument('-d', '--encode_difficult', action="store_true", required=False, help='Whether or not to encode the difficult object into KITTI labels')
    args, _ = parser.parse_known_args(args)
    return args


def xml_to_kitti(input_dir, output_dir, encode_difficult, classes):
    if not os.path.exists(input_dir):
        raise ValueError('input_dir not found.')
    if not os.path.exists(output_dir):
        raise ValueError('output_dir not found.')
    for annot in os.listdir(input_dir):
        et = ET.parse(os.path.join(input_dir, annot))
        element = et.getroot()
        element_objs = element.findall('object')
        element_width = int(element.find('size').find('width').text)
        element_height = int(element.find('size').find('height').text)
        element_depth = int(element.find('size').find('depth').text)
        assert element_depth == 3
        assert len(element_objs) > 0, 'No objects in {}.'.format(os.path.join(input_dir, annot))
        lines = []
        for element_obj in element_objs:
            difficulty = int(element_obj.find('difficult').text) == 1
            if difficulty and encode_difficult:
                dif = '1'
            else:
                dif = '0'
            line = ''
            class_name = element_obj.find('name').text
            assert class_name in classes
            line += class_name
            line += ' '
            line += '0 {} 0 '.format(dif)
            obj_bbox = element_obj.find('bndbox')
            x1 = int(round(float(obj_bbox.find('xmin').text)))
            y1 = int(round(float(obj_bbox.find('ymin').text)))
            x2 = int(round(float(obj_bbox.find('xmax').text)))
            y2 = int(round(float(obj_bbox.find('ymax').text)))
            line += str(x1)
            line += ' '
            line += str(y1)
            line += ' '
            line += str(x2)
            line += ' '
            line += str(y2)
            line += ' '
            line += '0 0 0 0 0 0 0\n'
            lines.append(line)
        with open(os.path.join(output_dir, os.path.basename(annot).split('.')[0]+'.txt'), 'w') as f:
            f.writelines(lines)


if __name__ =='__main__':
    classes = ['horse',
               "pottedplant",
               "train",
               "person",
               "bird",
               "car",
               "chair",
               "tvmonitor",
               "bus",
               "sofa",
               "dog",
               "motorbike",
               "bicycle",
               "sheep",
               "boat",
               "cat",
               "bottle",
               "diningtable",
               "cow",
               "aeroplane",
               "background",
               ]
    args = parse_args()
    xml_to_kitti(args.input_label_dir, args.output_label_dir, args.encode_difficult, classes)

