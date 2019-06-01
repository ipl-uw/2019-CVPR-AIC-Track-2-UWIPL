import cv2
import os
import os.path as osp

IMG_DIR = "./track1_test_img/"
OUT_DIR = "./track1_sct_img_test_big/"

for res_f in os.listdir("./txt_GPS_new/"):
    camid = res_f.split(".")[0]
    cam_img_path = osp.join(IMG_DIR, camid)
    out_cam_path = osp.join(OUT_DIR, camid)

    if not osp.isdir(out_cam_path):
        os.makedirs(out_cam_path)

    for line in open(osp.join("./txt_GPS_new/", res_f)).readlines():
        tmp = line.strip("\n").split(",")
        f_id = tmp[0]
        obj_id = tmp[1]

        img_f = osp.join(cam_img_path, f_id.zfill(10) + ".jpg")
        img = cv2.imread(img_f)
	
        height, width = img.shape[:2]

		
        left = int(tmp[2])-20
        top = int(tmp[3])-20
        w = int(tmp[4])+40
        h = int(tmp[5])+40
	
        right = left + w
        bot = top + h

        if left<0:
            left = 0
        if top<0:
            top=0

        if right>width:
            right = width
        if bot>height:
            bot=height

	

        crop_img = img[top: bot, left:right]

        out_obj_path = osp.join(out_cam_path, obj_id)
        if not osp.isdir(out_obj_path):
            os.makedirs(out_obj_path)

        out_path = osp.join(out_obj_path, f_id.zfill(10) + ".jpg")
        cv2.imwrite(out_path, crop_img)

