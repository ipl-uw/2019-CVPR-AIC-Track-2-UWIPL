import os
import cv2
import sys
import numpy as np
import os.path as osp

FILE_LEN = 10

lev1s = ["./S02/", "./S05/"]

OUT_DIR = "./track1_test_img/"


for lev1 in lev1s:
    lev2s = os.listdir(lev1)
    for lev2 in lev2s:
        camera_path = osp.join(lev1, lev2)
        path_to_vid = osp.join(camera_path, "vdo.avi")

        vid = cv2.VideoCapture(path_to_vid)
        
        suc = True
        img = None

        count = 1

        out_path = osp.join(OUT_DIR, lev2)
        if not osp.isdir(out_path):
        	os.makedirs(out_path)

        while suc:
            suc, img = vid.read()
            if img is None:
                break

            f_name = osp.join(out_path, str(count).zfill(10) + ".jpg")

            cv2.imwrite(f_name, img)            
            count += 1
