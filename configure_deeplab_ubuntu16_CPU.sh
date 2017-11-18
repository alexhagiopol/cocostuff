#!/usr/bin/env bash
set -e
# basics
sudo apt-get install -y build-essential unzip
# caffe's dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y libmatio-dev
sudo apt-get install -y python-numpy

# needed for CPU-only multithreading
sudo apt-get install -y libopenblas-dev
NUMBER_OF_CORES=16
echo "export OPENBLAS_NUM_THREADS=($NUMBER_OF_CORES)" >> ~/.bash_profile
# python script dependencies
sudo apt-get install -y python3-matplotlib
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-scipy
# download dataset
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip
unzip downloads/cocostuff-10k-v1.1.zip -d dataset/
# update submodule
git submodule update --init models/deeplab/deeplab-public-ver2
# make folders
mkdir models/deeplab/deeplab-public-ver2/cocostuff && mkdir models/deeplab/deeplab-public-ver2/cocostuff/data
make symlinks to data
cd models/deeplab/cocostuff/data && ln -s ../../../../dataset/images images && cd ../../../..
# run annotation conversion script
python dataset/code/conversion/convertAnnotationsDeeplab.py

# install cuda toolkit - still need this even for cpu-only path
wget -O cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb "https://www.dropbox.com/s/08ufs95pw94gu37/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb?dl=1"
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
# no driver and samples. cuda toolkit not cuda!
sudo apt-get install -y cuda-toolkit-8-0

# go into deeplab dir to compile
cd models/deeplab/deeplab-public-ver2
# copy makefile config example to actual makefile config
cp Makefile.config.example.CPU Makefile.config
# compile
make all -j$(nproc)
# run tests to make sure your CPU and software work together
make runtest -j$(nproc)
# download base VGG16 model and place in path
wget --directory-prefix=models/deeplab/cocostuff/model/deeplabv2_vgg16 http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/deeplabv2_vgg16_init.caffemodel

