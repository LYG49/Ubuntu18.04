# Ubuntu 18.04 安裝RTX3080顯卡驅動
在這邊我們會安裝東西如下：

1.顯卡驅動

2.Cuda

3.Cudnn

4.虛擬環境virtualenv

5.Torch

6.tensorflow

# 第一步安裝顯卡驅動
這裡我們以RTX3080為例

先進去NVIDA官網把驅動程式下載下來

在等待RTX3080驅動程式的時候，先make和gcc安裝起來

sudo apt install make

sudo apt install gcc


我這邊是下載RTX3080，驅動程式版本為460.56

我們要到你下載的資料夾執行安裝驅動程式

正常文件都會跑到下載資料夾的部分

所以我們要先下指令到下載的資料夾

才能去執行安裝

cd 下載/

sudo bash NVIDA-Linux-x86_64-460.56.run(文件名會跟著下載檔案名稱進行改變)


顯卡安裝結束可以去查看顯卡狀態

nvidia-smi

如果要讓GPU自動更新狀態

watch -n 1 nvidia-smi

查看CPU狀態

top

# 第二步下載cuda

這邊我Cuda版本下載的是11.3

wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run

sudo sh cuda_11.3.0_465.19.01_linux.run


在/home/<user_name>的.bashre最下方新增並儲存

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export PATH=$PATH:/usr/local/cuda/bin


重新開啟終端機更新終端機環境

Source ~/.bashrc


驗證cuda 

nvcc -V


顯示：


nvcc: NVIDIA (R) Cuda compiler driver

Copyright (c) 2005-2021 NVIDIA Corporation

Built on Sun_Feb_14_21:12:58_PST_2021

Cuda compilation tools, release 11.2, V11.2.152

Build cuda_11.2.r11.2/compiler.29618528_0


如果版本下載錯誤移除方法

我以移除cuda版本10.2為例

sudo /usr/local/cuda-10.2/bin/cuda-uninstaller


# 第三步下載cudnn

先去網路上找Cudnn下載檔

載完解壓cuDNN包

tar -xvf cudnn-linux-x86 64-8.4.0.24_cuda11.6-archive.tar.xz(這邊是你下載的cudnn檔案)

將以下文件複製到 cuda 工具包目錄中

sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 

sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 

sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

測試cudnn有沒有安裝成功，在原先的資料夾下載位置使用

cp -r /usr/src/cudnn_samples_v8/ $HOME

cd $HOME/cudnn_samples_v8/mnistCUDNN

make clean && make

 ./mnistCUDNN

# 第四步安裝虛擬環境

安裝virtualenv

sudo apt-get install python3-pip

sudo pip3 install virtualenv

sudo pip3 install virtualenvwrapper

在/home/<user_name>裡的.bashrc新增並儲存

export WORKON_HOME=$HOME/.virtualenvs

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3

export VIRTUALENVWRAPPER_VIRTUALENV_ARGS=' -p /usr/bin/python3 '

export PROJECT_HOME=$HOME/Devel

source /usr/local/bin/virtualenvwrapper.sh

創造新的虛擬環境

mkvirtualenv <venv_name>

啟用已經存在的虛擬環境

Workon <venv>

# 第五步下載Torch

安裝torch
 
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

驗證torch
 
import torch
 
print(torch.__version__)
 
print(torch.cuda.is_available())
 
查看pytorch對應的cuda版本
 
import torch
 
torch.version.cuda

# 第六步下載tensorflow

安裝tensorflow
 
pip install tensorflow-gpu

驗證tensorflow
 
import tensorflow as tf
 
hello = tf.constant('Hello,TensorFlow!')
 
sess = tf.Session()
 
print(sess.run(hello))
