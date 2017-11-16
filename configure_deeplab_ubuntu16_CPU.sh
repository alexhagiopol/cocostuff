#!/usr/bin/env bash
set -e
# export paths
echo export LD_LIBRARY_PATH=$PWD:\$LD_LIBRARY_PATH >> ~/.bashrc
cd ..
echo export CUDA_HOME=$PWD >> ~/.bashrc
source ~/.bashrc
# caffe's dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y libmatio-dev
sudo apt-get install -y python-numpy
# python script dependencies
sudo apt-get install -y python-matplotlib
sudo apt-get install -y python-opencv
sudo apt-get install -y python-scipy
# download dataset
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip
unzip downloads/cocostuff-10k-v1.1.zip -d dataset/
# update submodule
git submodule update --init models/deeplab/deeplab-public-ver2
# make folders
mkdir models/deeplab/deeplab-public-ver2/cocostuff && mkdir models/deeplab/deeplab-public-ver2/cocostuff/data
# make symlinks to data
cd models/deeplab/cocostuff/data && ln -s ../../../../dataset/images images && cd ../../../..
# run anootation conversion script
python dataset/code/conversion/convertAnnotationsDeeplab.py
# go into deeplab dir
cd models/deeplab/deeplab-public-ver2
# copy makefile config example to actual makefile config
cp Makefile.config.example.CPU Makefile.config
# compile
make all -j$(nproc)
# run tests to make sure your CPU and software work together
make runtest -j$(nproc)
