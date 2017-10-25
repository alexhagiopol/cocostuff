#!/usr/bin/env bash
set -e
# install dependencies for Ubuntu 16.04 and Pascal GPUs
# nvidia driver
sudo add-apt-repository ppa:xorg-edgers/ppa
sudo apt-get update
sudo apt-get install nvidia-375
# cuda toolkit
wget -O cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb "https://www.dropbox.com/s/08ufs95pw94gu37/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb?dl=1"
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
# cuDNN
wget -O cudnn-8.0-linux-x64-v5.1.tgz "https://www.dropbox.com/s/9uah11bwtsx5fwl/cudnn-8.0-linux-x64-v5.1.tgz?dl=1"
tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz
cd cuda/lib64
# export paths
# export LD_LIBRARY_PATH=/home/alex/cuda/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/home/alex/cuda/
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
cd ..
export CUDA_HOME=`pwd`
sudo apt-get install libcupti-dev
# caffe's misc dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y libmatio-dev
sudo apt-get install -y python-numpy
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
cp Makefile.config.example Makefile.config
# compile
make all -j$(nproc)
# run tests to make sure your GPU and software work together
make runtest -j$(nproc)
