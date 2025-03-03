# SnowMocap - A free Blender mocap solution integrated with camera array 

[繁體中文文檔](docs/ch.md)

## Introduction

I'm building a simple but reliable 3D mocap solution for Blender.

The goal of this project is to make the mocap setup as cheap as possible, so no expensive mocap ir cameras or specialized tracking suit, but also capable enough to use in all sorts of project(game asset, 3D animation, etc...)

4 RGB wedcams + human keypoint detection were used to accomplish the 2D human feature extraction and multi-camera triangulation, and we can get the precise 3D Human keypoints.

Finally, use the predefined armature in Blender to translate the keypoint data and produce continuous 3D human skeletal animation.

## Get started

### Hardware
 - 4 webcams
 - 4 camera mounts (tripod)
 - 4 10m(or longer) USB signal booster cable
 - A computer with Nvidia GPU

### 1. Hardware Setup
**1. Room setup**

First, you need a space big enough for motion capturing. The room I used for this project is around 10m x 10m wide and 3m tall.

<img src="image/room.png">

**2. Camera array setup**
 
Install all 4 webcams in 4 corners of the room using camera mounts or tripods. 

I used 4 custom 3D printed camera mounts to fix webcams on the ceiling for best viewing angle.

<img src="image/camera_holder_3dprinted.png">

**3. USB cable setup**

The USB cable came with the webcams were around 1m, it not long enough to connect to the computer all across the room, so USB signal booster cables(>=10m) were required.

<img src="image/usb_signal_booster_cable.png">

The USB cables needs to have **signal boost feature,** or the webcam's video stream signal will start to decay if the usb cable is longer than 6m.

Connect the webcams to the computer, I used some hooks to route the USB cables along the ceiling to my computer.

<img src="image\camera_setup_work.png">

If you are using tripods, you can use duct tape to stick USB cables on the ground, to prevent tripping over by these cables.

**HARDWARE SETUP DONE!**

### 2. Software Setup
**1. Install miniconda**

**Virtual environment is recommended.**

https://www.anaconda.com/download/



You can still use local python to install the dependency and run this project, but it'll be a pain in the ass if you were trying to reinstall some packages.

Once miniconda is installed, open **Anaconda Prompt** and type:

```
conda create -n snowmocap python=3.8
```

To create a python 3.8 virtual environment for SnowMocap

**2. Install CUDA/CUDNN**

**CUDA/CUDNN installation is recommended.**

If you want to use cpu to run SnowMocap, you can skip this step.

But it take a lot longer to do human keypoint detection.

https://anaconda.org/nvidia/cuda-toolkit

```
conda install nvidia::cuda-toolkit
```

https://anaconda.org/anaconda/cudnn

```
conda install anaconda::cudnn
```

**3. Install rtmlib**

https://github.com/Tau-J/rtmlib

```
git clone https://github.com/Tau-J/rtmlib.git
cd rtmlib
pip install -r requirements.txt
pip install -e .
pip install onnxruntime-gpu
```

rtmlib's installation might need a little work around to get it working correctly.

I have test the project on two PCs, one with RTX4060 and other with RTX3060.

It turns out that **the onnxruntime/onnxruntime-gpu and cuda/cudnn version will be different for different GPUs.**

```python
#PC with RTX3060
cuda                  12.8.0
cudnn                 9.1.1.17
onnxruntime           1.16.0
onnxruntime-gpu       1.19.0
#PC with RTX4060
cuda                  12.6.3
cudnn                 9.1.1.17
onnxruntime           1.19.2
onnxruntime-gpu       1.19.2
```

You can use the following command to pip install a package in specific version.

```
pip install package_name==version
```

If you want to install onnxruntime-gpu in version 1.19.0, it'll looks like :

```
pip install onnxruntime-gpu==1.19.0
```

**4. Install SnowMocap**

```
git clone https://github.com/liaochikon/SnowMocap.git
cd SnowMocap
pip install -r requirements.txt
```

**SOFTWARE SETUP DONE!**

### 3. Calibration

### 4. Motion Capture

### 5. Blender Animation

## Modification

## Contact