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

- label.txt is the categories.
- data.txt is the training data path and labels. Noted that the example here is to used for training vehicle type and brand, weighted for the losses need to be revised in the code (`multi_label_classifier.py` line 68-74). The traning data should follow this format.
- for other training/testing/visualization options, please refer to option.py.

Use the following command to run training code.

python multi_label_classifier.py --dir "./YOUR_DIRPATH_OF_data.txt_and_label.txt/" --mode "Train" --model "LightenB" --name "YOURMODELNAME" --batch_size 8 --gpu_ids 0 --input_channel 3 --load_size 512 --input_size 512 --ratio "[0.7, 0.1, 0.2]" --load_thread 4 --sum_epoch 500 --lr_decay_in_epoch 1 --display_port 8900 --validate_ratio 0.5 --top_k "(1,)" --score_thres 0.1 --display_train_freq 1000 --display_validate_freq 1000 --save_epoch_freq 2000  --display_image_ratio 0.1 --shuffle 


## Testing

To test the model, make sure you have the image id and paths under ./your_model_name/Data/Test/data.txt. Then run, 

python multi_label_classifier.py --dir "./YOUR_DIRPATH_OF_data.txt_and_label.txt/" --mode "Test" --model "LightenB" --name "YOURMODELNAME"  --checkpoint_name "/path_to_model.pth"


The probabilities of each label will be saved in `test.log`.


