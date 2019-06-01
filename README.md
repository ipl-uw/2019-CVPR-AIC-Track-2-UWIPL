# 2019-CVPR-AIC-Track-2-UWIPL
Repository for 2019 CVPR AI City Challenge Track 2 from IPL@UW

## Code structure
Our code consists of the following three components:

### 1. Video-Person-ReID
The vehicle reidentification code is based on Jiyang Gao's Video-Person-ReID \[[code](https://github.com/jiyanggao/Video-Person-ReID)\].

### 2. metadata
Metadata model for vehicle's type, brand and color.

### 3. CarKeypoints
The vehicle keypoints code is based on krrish94's CarKeypoints \[[code](https://github.com/krrish94/CarKeypoints)\].

## Training

Training of both Video-Person-ReID and metadata requires CarKeypoints's inference result on training set. For CarKeypoints, we use the pre-trained model \[[model](https://github.com/krrish94/CarKeypoints)\].

## Testing

Testing of both Video-Person-ReID and metadata requires CarKeypoints's inference result on testing set. In addition, Video-Person-ReID needs metadata's inference result on testing set.
