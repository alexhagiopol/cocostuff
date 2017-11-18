#!/usr/bin/env bash
set -e
list_file=cocostuff/list/val.txt
iterations=`cat ${list_file} | wc -l`
weights_file=cocostuff/model/deeplabv2_vgg16/train_iter_20000.caffemodel
feature_dir=cocostuff/features/deeplabv2_vgg16/val/fc8
model_file=cocostuff/config/deeplabv2_vgg16/test_val.prototxt
caffe_bin=deeplab-public-ver2/.build_release/tools/caffe.bin
mkdir -p ${feature_dir}
${caffe_bin} test --model=${model_file} --weights=${weights_file} --iterations=1
