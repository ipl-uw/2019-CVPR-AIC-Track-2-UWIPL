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


img_camID={}
img_carID={}

for neighbor in root.iter('Item'):
    #print(neighbor.attrib)
    #print(neighbor.get('imageName'))
    #print(neighbor.get('vehicleID'))
    #print(neighbor.get('cameraID'))
    img_camID[neighbor.get('imageName')] = neighbor.get('cameraID')
    img_carID[neighbor.get('imageName')] = neighbor.get('vehicleID')

carID_num={}

aic_track2_dir = '/path_to_aic19-track2-reid/'

source_path = aic_track2_dir + "image_train/"
path_train = aic_track2_dir + "image_train_deepreid/"
path_query = aic_track2_dir + "image_train_deepreid_query/"
path_query_single = aic_track2_dir + "image_train_deepreid_query_single/"
path_gallery = aic_track2_dir + "image_train_deepreid_gallery/"
os.mkdir(path_train)
os.mkdir(path_query)
os.mkdir(path_query_single)
os.mkdir(path_gallery)
file = open("train_track.txt","r")
for line in file:
    #print(line)
    s = line.replace(" \n","").split(" ")
    #print(s)
    # find single query i as the minimum image number in s
    tmp = [int(c[:-4]) for c in s]
    sq = tmp.index(min(tmp))
    for i in range(0,len(s)):
        
        carID = img_carID[s[i]]
        camID = img_camID[s[i]]
        

        if len(carID_num)<160 or carID in carID_num:
            if carID in carID_num:
                #if len(carID_num[carID])==1 and carID_num[carID][0]!=camID:
                #    carID_num[carID].append(camID) #ccc
                if not camID in carID_num[carID]:
                    carID_num[carID].append(camID)
            else:
                carID_num[carID]=[]
                carID_num[carID].append(camID)
        
        #print(carID_num[carID])
        #print(len(carID_num))
        #if len(carID_num)<160:
        if carID in carID_num and False:
            if len(carID_num[carID])==1:
                ###camID = carID_num[carID][0] #ccc
                if not os.path.isdir(path_query+"/"+carID+"/"):
                    os.mkdir(path_query+"/"+carID+"/")
                if not os.path.isdir(path_query+"/"+carID+"/"+camID+"/"):
                    os.mkdir(path_query+"/"+carID+"/"+camID+"/")
                copy_rename(source_path,s[i],path_query+"/"+carID+"/"+camID+"/",'%04d_%s'%(i,s[i]))
                if i == sq: # single query
                    if not os.path.isdir(path_query_single+"/"+carID+"/"):
                        os.mkdir(path_query_single+"/"+carID+"/")
                    if not os.path.isdir(path_query_single+"/"+carID+"/"+camID+"/"):
                        os.mkdir(path_query_single+"/"+carID+"/"+camID+"/")
                    copy_rename(source_path,s[i],path_query_single+"/"+carID+"/"+camID+"/",'%04d_%s'%(i,s[i]))

            #elif len(carID_num[carID])==2: #ccc
            else:
                #print("111111111")
                ###camID = carID_num[carID][1] #ccc
                if not os.path.isdir(path_gallery+"/"+carID+"/"):
                    os.mkdir(path_gallery+"/"+carID+"/")
                if not os.path.isdir(path_gallery+"/"+carID+"/"+camID+"/"):
                    os.mkdir(path_gallery+"/"+carID+"/"+camID+"/")
                copy_rename(source_path,s[i],path_gallery+"/"+carID+"/"+camID+"/",'%04d_%s'%(i,s[i]))
        else:
            #if carID not in carID_num:
            if not os.path.isdir(path_train+"/"+carID+"/"):
                os.mkdir(path_train+"/"+carID+"/")
            if not os.path.isdir(path_train+"/"+carID+"/"+camID+"/"):
                os.mkdir(path_train+"/"+carID+"/"+camID+"/")
            copy_rename(source_path,s[i],path_train+"/"+carID+"/"+camID+"/",'%04d_%s'%(i,s[i]))
            
                
