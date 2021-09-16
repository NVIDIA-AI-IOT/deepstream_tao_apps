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

# download
cd ~/tao-experiments/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtest_06-Nov-2007.tar
# Splitting datasets
mkdir -p ~/tao-experiments/data/voc0712trainval/images
mkdir -p ~/tao-experiments/data/voc0712trainval/labels
mkdir -p ~/tao-experiments/data/voc07test/images
mkdir -p ~/tao-experiments/data/voc07test/labels
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc07test/images ~/tao-experiments/data/VOCdevkit/VOC2007/JPEGImages/'{}'.jpg
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/images ~/tao-experiments/data/VOCdevkit/VOC2007/JPEGImages/'{}'.jpg
cat ~/tao-experiments/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/images ~/tao-experiments/data/VOCdevkit/VOC2012/JPEGImages/'{}'.jpg
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc07test/labels ~/tao-experiments/data/VOCdevkit/VOC2007/Annotations/'{}'.xml
cat ~/tao-experiments/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/labels ~/tao-experiments/data/VOCdevkit/VOC2007/Annotations/'{}'.xml
cat ~/tao-experiments/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt | xargs -I'{}' mv -t ~/tao-experiments/data/voc0712trainval/labels ~/tao-experiments/data/VOCdevkit/VOC2012/Annotations/'{}'.xml

