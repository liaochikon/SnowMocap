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

First, you need a space big enough for motion capturing.The room I used for this project is around 10m x 10m wide and 3m tall.

img_room

**2. Camera array setup**
 
 Install all 4 webcams at 4 corners using camera mounts or tripods

img_camera_mount
img_camera_room_setup

**3. USB cable setup**

Use the USB signal booster cable to connect the webcams and your computer

### 2. Software Setup
**1. Install miniconda**

https://www.anaconda.com/download/

**Virtual environment is recommended.**

You can still use local python to install the dependency and run this project, but it'll be a pain in the ass if you were trying to reinstall some packages.

**2. Install rtmlib**

https://github.com/Tau-J/rtmlib

rtmlib's installation might need a little work around to get it working correctly.

I have test the project on two PCs, one with RTX4060 and other with RTX3060.

It turns out that **the onnxruntime/onnxruntime-gpu and cuda/cudnn version will be different for different GPUs.**

table_onnxruntime/onnxruntime-gpu_version
table_cuda/cudnn_version

**3. Install SnowMocap**



### 3. Calibration

### 4. Motion Capture

### 5. Blender Animation

## Modification

## Contact