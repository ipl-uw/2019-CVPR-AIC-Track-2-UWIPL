# Video-Person-ReID for AIC19

The code is for the video-based vehicle reidentification task in AIC19 track 1 and 2 \[[link](https://www.aicitychallenge.org/)\].
The code is based on Jiyang Gao's Video-Person-ReID \[[code](https://github.com/jiyanggao/Video-Person-ReID)\].

### Requirement

PyTorch 0.3.1 <br />
Torchvision 0.2.0 <br />
Python 2.7 <br />

### Dataset

First download the AIC19 dataset \[[link](https://www.aicitychallenge.org/)\], and use the python scripts in `data_util/` to convert images, keypoints and metadata into desired file structure. Please copy the scripts to your path to `aic19-track2-reid` for simplicity.

1. Run `xml_reader_testdata.py` and `xml_reader_traindata.py` to convert images into desired file structure: `image_train_deepreid/carId/camId/imgId.jpg`.
2. Run `create_feature_files.py` to convert the keypoints into desired file structure as images:  `keypoint_train_deepreid/carId/camId/imgId.txt`.
3. Run `convert_metadata_imglistprob.py` to convert the metadata inference result of query (and test) tracks into `prob_v2m100_query.txt` and `imglist_v2m100_query.txt`. And then run `create_metadata_files.py` to convert the metadata into desired file structure as images:  `metadata_v2m100_query_deepreid/carId/camId/imgId.txt`. If using other metadata models, change `v2m100` to other names. Example txt output from the provided metadata model \[[link](https://github.com/ipl-uw/2019-CVPR-AIC-Track-2-UWIPL/tree/master/metadata)\] can be downloaded [here](https://drive.google.com/open?id=1X4geSMtsHCztwmhuUimjFjEZGUImsA7L).


### Training

To train the model, please run
<br />
`
python  main_video_person_reid.py --train-batch 16 --workers 0 --seq-len 4 --arch resnet50ta_surface_nu --width 224 --height 224 --dataset aictrack2 --use-surface --save-dir log --learning-rate 0.0001 --eval-step 50 --save-step 50 --gpu-devices 0 --re-ranking --metadata-model v2m100 --bstri
`
<br />

`arch` could be `resnet50ta_surface_nu` (Temporal Attention with keypoints feature, for AIC19 track 2) or `resnet50ta` (Temporal Attention, for AIC19 track 1). If using `resnet50ta`, do not use `--use-surface`.<br />

### Testing

To test the model, please run
<br />
`
python  main_video_person_reid.py --train-batch 16 --workers 0 --seq-len 4 --arch resnet50ta_surface_nu --width 224 --height 224 --dataset aictrack2 --use-surface --evaluate --pretrained-model log/checkpoint_ep300.pth.tar --save-dir log-test --gpu-devices 0 --re-ranking --metadata-model v2m100
`
<br />
Optionally, start from previously saved feature without redoing inference
<br />
`
python  main_video_person_reid.py --dataset aictrack2 --save-dir log --re-ranking --metadata-model v2m100 --load-feature --feature-dir feature_dir
`
<br />
`feature_dir` can be point to previously saved feature directory, e.g. `log/feature_ep0300`.<br />

The pre-trained model can be download at [here](https://drive.google.com/open?id=1jjwQhk8i4X12_DjCz9LlgrvL-9uKa2mE).<br/>
Besides, the confusion matrix of metadata model need to be put under `metadata/`. Example confusion matrix can be downloaded [here](https://drive.google.com/open?id=178oG9f8H58YgVWsk_KaxpWf_i3dr2wER).


### AIC19 track 1

For generating features for our AIC19 track 1 's testing \[[code](https://github.com/ipl-uw/2019-CVPR-AIC-Track-1-UWIPL)\], run
<br />
`
python  Graph_ModelDataGen.py
`
<br />
The pretrained model can be downloaded [here](https://drive.google.com/file/d/1C-uE8nPA3Rtu8tkHptRS8J87sayrB7Nj/view?usp=sharing). The model should be put under `log/`.<br/>
Besides, the data should be processed in a different manner:<br/>
Create video2img folder in the downloaded project (i.e., Video-Person-ReID/video2img/).
Put and run `python crop_img.py` in the same folder in the downloaded dataset (i.e., aic19-track1-mtmc/test). You need to creat a folder track1_test_img in the same path (i.e., aic19-track1-mtmc/test/track1_test_img). After that, create a folder track1_sct_img_test_big and run `python crop_img_big.py`. Then, create a folder log in the dowanloaded project (i.e., Video-Person-ReID/log) and put the downloaded model file of track1 ReID in this folder. Finally, run `python Graph_ModelDataGen.py` to obtain the feature files (q_camids3_no_nms_big0510.npy, qf3_no_nms_big0510.npy and q_pids3_no_nms_big0510.npy).<br />

### Reference

The code is based on Jiyang Gao's [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID).<br/>
The visualization code is adopted from KaiyangZhou's [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).<br/>
The re-ranking code is modified based on zhunzhong07's [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking).
