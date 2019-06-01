The metadata classifier code is based on \[[code](https://github.com/pangwong/pytorch-multi-label-classifier)\]. We use the 29 -layer light CNN model with modifications on transformations, input size and the beginning layers.(*Wu, X., He, R., Sun, Z. and Tan, T., 2018. A light cnn for deep face representation with noisy labels. IEEE Transactions on Information Forensics and Security, 13(11), pp.2884-2896*).

## Setup

This code assumes you have the following packages installed.
- Python 3.6
- Pytorch 0.3.1

## Our pre-trained model

Download the pre-trained model [here](https://drive.google.com/file/d/1Ln-rK6k1vMjcC-0ySK9RMH6mLlXpdUsw/view?usp=sharing).
- The model is pretrained on CompCar dataset [link](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html).
- Then some images in track2 training set are used for further training. 

## Training



## Running inference for AIC19

To run inference on multiple images, put image paths in a text file (e.g. `fullpath_train.txt`). Then, run the inference script.
```
inference-imageset.lua
```

The result will be saved in `keypoint-train.txt`.

