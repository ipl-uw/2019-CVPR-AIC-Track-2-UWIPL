# Video-Person-ReID for AIC19

The code is for the video-based vehicle reidentification task in AIC19 track 1 and 2 \[[link](https://www.aicitychallenge.org/)\].
The code is modified from Jiyang Gao's Video-Person-ReID \[[code](https://github.com/jiyanggao/Video-Person-ReID)\].

### Requirement

PyTorch 0.3.1 <br />
Torchvision 0.2.0 <br />
Python 2.7 <br />

### Dataset

First download the AIC19 dataset \[[link](https://www.aicitychallenge.org/)\], and use the python scripts in `data_util/` to convert images, keypoints and metadata into desired file structure.

1. Run `xml_reader_testdata.py` and `xml_reader_traindata.py` to convert images into desired file structure: `image_train_deepreid/carId/camId/imgId.jpg`.
2. Run `create_feature_files.py` to convert the keypoints into desired file structure as images:  `keypoint_train_deepreid/carId/camId/imgId.txt`.
3. Run `convert_metadata_imglistprob.py` to convert the metadata inference result of query (and test) tracks into `prob_metadatamodel_query.txt` and `imglist_metadatamodel_query.txt`. And then run `create_metadata_files.py` to convert the metadata into desired file structure as images:  `metadata_metadatamodel_query_deepreid/carId/camId/imgId.txt`.

### Training

To train the model, please run <br />
`
python  main_video_person_reid.py --train-batch 16 --workers 0 --seq-len 4 --arch resnet50ta_surface_nu --width 224 --height 224 --dataset aictrack2 --use-surface --save-dir log --learning-rate 0.0001 --eval-step 50 --save-step 50 --gpu-devices 0 --re-ranking --metadata-model metadatamodel --bstri
`
The pre-trained model can be download at [here](https://github.com/ipl-uw).

`arch` could be resnet50ta_surface_nu (Temporal Attention with keypoints feature, for AIC19 track 2) or resnet50ta (Temporal Attention, for AIC19 track 1). If using resnet50ta, do not use `--use-surface`.
`metadatamodel` is correspond to the folder name `metadata_metadatamodel_query_deepreid` created in previous step.

### Testing

To test the model, please run <br />
`
python  main_video_person_reid.py --train-batch 16 --workers 0 --seq-len 4 --arch resnet50ta_surface_nu --width 224 --height 224 --dataset aictrack2 --use-surface --evaluate --pretrained-model log/checkpoint_ep300.pth.tar --save-dir log-test-m --gpu-devices 1 --re-ranking --metadata-model metadatamodel
`
Optionally, run on previously saved feature without redoing inference <br />
`
python  main_video_person_reid.py --dataset aictrack2 --save-dir log --re-ranking --metadata-model metadatamodel --load-feature --feature-dir feature_dir
`
`feature_dir` can be point to previously saved feature directory, e.g. `log/feature_ep0300`.
