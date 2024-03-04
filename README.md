<br />
<div align="center">
  <h1 align="center">SnowMocap</h1>
  <h2 align="center">This project is still work in progress!!!</h2>
  <p align="center">
    A free Blender 3D multi-camera mocap solution.
    <br />
  </p>
</div>
<img src="image/mocap_demo_gif.gif">

## About This Project

(Sorry for the bad english, it's not my first language. (´c_`) )

I'm tring to build a simple but reliable 3D mocap solution for blender.

The goal of this project is to make the mocap setup as cheap as possible, so no expensive mocap ir cameras and no tracking suit, but also capable enough to use in all sorts of project(game asset, 3D animation, etc...)

I'll be using RGB wedcams + human keypoint detection to accomplish the 2D human feature extraction and multi-camera triangulation to get precise 3D keypoint result.

Finally, use the IK function within blender and the predefined rig the produce human skeleton animation.

Because the animation is already in the animation timeline of Blender, so it can be exported or modified easily.

## My Setup
The camera I'm using are 4 ASUS webcams(ASUS webcam c3)

https://www.asus.com/tw/accessories/streaming-kits/all-series/asus-webcam-c3/

<img src="https://dlcdnwebimgs.asus.com/gain/397bff36-2a38-4408-a7a1-922a6520f39d/w692" height=400>

## Algorithm
The tracking method I choose is HRNet keypoint detection.

<img src="image/single_person_cam.gif" height=300>
<img src="https://github.com/HRNet/HRNet-Human-Pose-Estimation/raw/master/figures/hrnet.png" height=250>

Because of its high resolution and multi-sacle feature extraction makes it very suitable for the project.

HRNet outputs the most satisfying keypoint detection result in my opinion.






