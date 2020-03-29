The vehicle keypoints code is based on krrish94's CarKeypoints \[[code](https://github.com/krrish94/CarKeypoints)\].

# CarKeypoints

This repository contains inference code for using a modified [stacked-hourglass](https://github.com/krrish94/stacked-hourglass) to detect semantic keypoints on cars. 

The network outputs a likelihood of keypoint presence over every pixel of an input image (the input image is a 64 x 64 car bounding box).

Here is a 3D wireframe with reference keypoints.
<p align="center">
	<img src="assets/carkeypoints.png" />
</p>

## Setup

This code assumes you have the following packages installed.
* [Torch7](https://github.com/torch/torch7)
* Torch packages: `nn`, `cunn`, `cudnn`, `image`, `nngraph`


## Downloading the pre-trained model

Download the pre-trained model [here](https://www.dropbox.com/s/qezt3e02j4uawov/model.t7?dl=0).


## Running the inference code

To perform inference on a set of images, first edit `valid.txt` and add paths to the images you need to run inference on. **These images must only contain cropped car bounding boxes** (i.e., from any image that contains a car, pick only one car bounding box and crop the region of the image contained within that bounding box). These are the only kind of images the model has been trained on.

Then, run the inference script.
```
inference.lua
```

This will write a `results.txt` file (you can edit the name and path of this output file in `inference.lua`).

## Running inference for AIC19

To run inference on multiple images, put image paths in a text file (e.g. `fullpath_train.txt`). Then, run the inference script.
```
inference-imageset.lua
```

The result will be saved in `keypoint-train.txt`. Example results can be downloaded [here](https://drive.google.com/open?id=1m96n_1gsHy3iI9ruRGDGqaVXqjJgVcKf).
