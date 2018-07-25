#!/bin/bash

# ImageNet training setup script for DLAMI + p3 instance
# (tested on "Deep Learning AMI (Ubuntu) Version 11.0")
# https://aws.amazon.com/marketplace/fulfillment?productId=17364a08-2d77-4969-8dbe-d46dcfea4d64&ref_=dtl_psb_continue
  
echo 'Starting script'

sudo apt-get update
sudo apt-get upgrade -y

# Change nccl to 9-1
sed -i -e 's/cuda-9.0/cuda-9.1/g' ~/.bashrc
source ~/.bashrc

conda install pytorch torchvision cuda91 -c pytorch -y
conda install tqdm -y

# index file used to speed up evaluation
pushd ~/data/imagenet 
wget --no-clobber https://s3.amazonaws.com/yaroslavvb/sorted_idxar.p
popd

# https://gist.github.com/soumith/01da3874bf014d8a8c53406c2b95d56b
conda uninstall --force pillow -y 
# install libjpeg-turbo to $HOME/turbojpeg 
git clone https://github.com/libjpeg-turbo/libjpeg-turbo 
pushd libjpeg-turbo 
mkdir build 
cd build 
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/turbojpeg 
make 
make install 
popd 
# install pillow-simd with jpeg-turbo support 
git clone https://github.com/uploadcare/pillow-simd 
pushd pillow-simd 
CPATH=$HOME/turbojpeg/include LIBRARY_PATH=$HOME/turbojpeg/lib CC="cc -mavx2" python setup.py install 
# add turbojpeg to LD_LIBRARY_PATH 
export LD_LIBRARY_PATH="$HOME/turbojpeg/lib:$LD_LIBRARY_PATH" 
popd


# following GPU settings from below (not clear if helps) 
# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html
sudo nvidia-smi -ac 877,1530

cd ~/

DATA_DIR=~/data
if [ ! -d "$DATA_DIR" ]; then
    mkdir data
fi



# setting network settings - 
# https://github.com/aws-samples/deep-learning-models/blob/5f00600ebd126410ee5a85ddc30ff2c4119681e4/hpc-cluster/prep_client.sh
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216'
sysctl -w net.core.netdev_max_backlog=30000
sysctl -w net.core.rmem_default=16777216
sysctl -w net.core.wmem_default=16777216
sysctl -w net.ipv4.tcp_mem='16777216 16777216 16777216'
sysctl -w net.ipv4.route.flush=1



# AS TODO: download from s3 if efs doesn't exist
if [ ! -d "$DATA_DIR/imagenet" ]; then
    cd $DATA_DIR
    # cat get those files from
    # wget https://s3.amazonaws.com/yaroslavvb/imagenet-data-sorted.tar
    rsync --progress /efs/data/imagenet-data-sorted.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-data-sorted.tar
    rm $DATA_DIR/imagenet-data-sorted.tar
    mv $DATA_DIR/raw-data $DATA_DIR/imagenet

    # can get those files from
    # wget https://s3.amazonaws.com/yaroslavvb/imagenet-sz.tar
    rsync --progress /efs/data/imagenet-sz.tar $DATA_DIR
    tar -xvf $DATA_DIR/imagenet-sz.tar
    rm $DATA_DIR/imagenet-sz.tar
    
    cd ~/
fi

echo ok > /tmp/nv_setup_complete
