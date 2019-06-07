from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np

from utils import mkdir_if_missing, write_json, read_json
from bases import BaseVideoDataset
"""Dataset classes"""


class AICityTrack2(BaseVideoDataset):

    def __init__(self, root, min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = root
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.min_seq_len = min_seq_len
        
        print("Note: if root path is changed, the previously generated json files need to be re-generated (so delete them first)")

        query = self._process_dir3(self.dataset_dir, self.split_query_json_path, relabel=False)


        self.query = query
        self.num_query_pids, _, self.num_query_cams = self.get_videodata_info(self.query)

    def _process_dir3(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        camids = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing '{}' with {} cameras".format(dir_path, len(camids)))


        tracklets = []
        for camid in camids:
            ss = camid.split("/")
            cam = camid
            
            camid = int(osp.basename(ss[7].replace("c","")))
            print(camid)

            pidrs = glob.glob(osp.join(cam, '*'))
            for pdir in pidrs:
                raw_img_paths = glob.glob(osp.join(pdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                imgfiles = os.listdir(pdir)
                img_paths = []

                for imgfile in imgfiles:
                    img_idx_name = imgfile
                    img_paths.append(pdir+"/"+imgfile)

                ############### keep N largest images
                N_largest = 32 
                if N_largest > 0 and len(img_paths) > N_largest:
                    from PIL import Image
                    w = 4 # window for average size
                    area_first = 0
                    area_last = 0
                    for img_path in img_paths[:w]:
                        img = Image.open(img_path)
                        width, height = img.size
                        area_first += width*height
                    for img_path in img_paths[-w:]:
                        img = Image.open(img_path)
                        width, height = img.size
                        area_last += width*height
                    if area_first > area_last:
                        img_paths = img_paths[:N_largest]
                    else:
                        img_paths = img_paths[-N_largest:]
                ##############################################
                
                img_name = osp.basename(img_paths[0])

                ss = pdir.split("/")
                pid = int(ss[8])
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
        }
        write_json(split_dict, json_path)

        return tracklets



"""Create dataset"""

__factory = {
    'aictrack2': AICityTrack2,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

if __name__ == '__main__':
    dataset = AICityTrack2()






