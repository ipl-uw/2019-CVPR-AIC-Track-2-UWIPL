# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 00:16:00 2019

@author: hungminhsu
"""

import os
import shutil
def copy_rename(src_dir,old_file_name,dst_dir ,new_file_name):
    src_file = os.path.join(src_dir, old_file_name)
    #print("src_file:"+src_file)
    shutil.copy(src_file,dst_dir)
    
    dst_file = os.path.join(dst_dir, old_file_name)
    #print("dst_file:"+dst_file)
    new_dst_file_name = os.path.join(dst_dir, new_file_name)
    #print("new_dst_file_name:"+new_dst_file_name)
    os.rename(dst_file, new_dst_file_name)

###########################################################################

import xml.etree.ElementTree as ET
xmlp = ET.XMLParser(encoding="utf-8")
tree = ET.parse('train_label.xml', parser=xmlp)
root = tree.getroot()


source_path_query = "/media/twhuang/NewVolume1/aic19/aic19-track2-reid/image_query/"
path_query = "/media/twhuang/NewVolume1/aic19/aic19-track2-reid/image_query_deepreid/"
os.mkdir(path_query)

q_img_camID={}
q_img_carID={}

q_imgs = [f for f in os.listdir(source_path_query)]
q_imgs.sort()
for i, img in enumerate(q_imgs):
    q_img_camID[img] = 'c901'  # camID for query starts from 901
    q_img_carID[img] = '%04d'%(i+1)
for i, img in enumerate(q_imgs):
    print(i)
    #print(s)
    carID = q_img_carID[img]
    camID = q_img_camID[img]
    
    
    if not os.path.isdir(path_query+"/"+carID+"/"):
        os.mkdir(path_query+"/"+carID+"/")
    if not os.path.isdir(path_query+"/"+carID+"/"+camID+"/"):
        os.mkdir(path_query+"/"+carID+"/"+camID+"/")
    copy_rename(source_path_query,img,path_query+"/"+carID+"/"+camID+"/",'%s'%(img))

source_path_test = "/media/twhuang/NewVolume1/aic19/aic19-track2-reid/image_test/"
path_test = "/media/twhuang/NewVolume1/aic19/aic19-track2-reid/image_test_deepreid/"
os.mkdir(path_test)

g_img_camID={}
g_img_carID={}
g_imgs = []
with open('test_track.txt', 'r') as f:
    for i, line in enumerate(f):
        s = line.replace('\n', '').strip().split(' ')
        g_imgs.append(s)
        for img in s:
            g_img_camID[img] = 'c001'
            g_img_carID[img] = '%04d'%(i+1)
for l, s in enumerate(g_imgs):
    print(l)
    #print(s)
    for i in range(0,len(s)):
        
        carID = g_img_carID[s[i]]
        camID = g_img_camID[s[i]]
        
        
        if not os.path.isdir(path_test+"/"+carID+"/"):
            os.mkdir(path_test+"/"+carID+"/")
        if not os.path.isdir(path_test+"/"+carID+"/"+camID+"/"):
            os.mkdir(path_test+"/"+carID+"/"+camID+"/")
        copy_rename(source_path_test,s[i],path_test+"/"+carID+"/"+camID+"/",'%04d_%s'%(i, s[i]))

