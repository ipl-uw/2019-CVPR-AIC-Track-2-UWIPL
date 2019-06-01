from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random

from math import exp, atan2
#import cv2

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    #print(img_path)
    return img

def read_metadata(img_path, metadata_model, verbose=True):
    """Read sruface from file"""
    if metadata_model[:2] == 'v1':
        metadata_dim = 26 # 6, 12, 8 for type, brand, color
    elif metadata_model[:2] == 'v2':
        metadata_dim = 46 # 7, 30, 9 for type, brand, color
    else: # the oldest version
        metadata_dim = 26 # 6, 12, 8 for type, brand, color
    metadata_path = img_path.replace('image', 'metadata_%s'%metadata_model).replace('.jpg', '.txt')
    if os.path.isfile(metadata_path):
        #print(metadata_path)
        with open(metadata_path, 'r') as f:
            metadata = []
            for line in f:
                #print(line)
                if ',' in line:
                    line = line.strip().replace(' ', '').split(',')
                    line = [s for s in line if len(s) > 0]
                else:
                    line = line.strip().split(' ')
                    line = [s for s in line if len(s) > 0]
                #print(line)
                metadata.append(np.array(line, dtype=np.float32))
        metadata = np.concatenate(metadata) ### concat all probability vector
        assert metadata.shape[0] == metadata_dim
        return metadata
    else:
        if verbose:
            print('warning: metadata not exist: ' + str(metadata_path))
        return np.zeros(metadata_dim, dtype=np.float32) ### if no metadata

def PolyArea(pts):
    return -0.5*(np.dot(pts[:,0],np.roll(pts[:,1],1))-np.dot(pts[:,1],np.roll(pts[:,0],1)))

def keypointsArea(keypoints, ids):
    pts = np.array([(keypoints[i][0], keypoints[i][1]) for i in ids])
    #return cv2.contourArea(pts, oriented=True)
    return PolyArea(pts)

def keypointsSymmetry(keypoints):
    area0 = abs(keypointsArea(keypoints, [i for i in range(2, 18)])) + 1
    area1 = abs(keypointsArea(keypoints, [i for i in range(2+18, 18+18)])) + 1
    ratio = area1 / area0 if area1 < area0 else area0 / area1
    #print('area0: %f, area1: %f' % (area0, area1))
    return ratio

def keypointsParallel(keypoints):
    NUM_PAIRS = 18
    vecs = np.zeros((NUM_PAIRS, 2), dtype=np.float32)
    for i in range(NUM_PAIRS):
        vecs[i][0] = keypoints[i+18][0] - keypoints[i][0]
        vecs[i][1] = keypoints[i+18][1] - keypoints[i][1]
    vec_mean = np.mean(vecs, axis=0)
    vec_diff = np.subtract(vecs, vec_mean)
    vec_err = np.linalg.norm(vec_diff, axis=1) / np.linalg.norm(vec_mean)
    vec_errmean = np.mean(vec_err)
    return exp(-vec_errmean)

def keypointsConfidence(keypoints):
    parallel_conf = keypointsParallel(keypoints)
    symmetry_conf = keypointsSymmetry(keypoints)
    keypoint_conf = pow(parallel_conf**2 + symmetry_conf**2, 0.5) / pow(2, 0.5)
    return keypoint_conf

def keypointsSurface(keypoints):
    surfaces = []
    idss = []
    idss.append([i for i in range(2, 18)])
    idss.append([i for i in range(20, 36)][::-1])
    for i in range(16):
        idss.append([i%16+2, i%16+2+18, (i+1)%16+2+18, (i+1)%16+2])
    for ids in idss:
        surfaces.append(keypointsArea(keypoints, ids))
    surfaces = np.array(surfaces, dtype=np.float32)
    surfaces /= np.linalg.norm(surfaces)
    #surfaces *= 999
    #print(surfaces)
    return surfaces

def surfacesAngle(surfaces):
    x = surfaces[0] - surfaces[1]
    y = surfaces[16] + surfaces[15] + surfaces[14] - surfaces[10] - surfaces[11] - surfaces[12]
    return atan2(y, x) # between -pi and pi

def read_keypoint(img_path):
    """Read keypoint from file"""
    keypoint_path = img_path.replace('image', 'keypoint').replace('.jpg', '.txt')
    with open(keypoint_path, 'r') as f:
        keypoints = np.loadtxt(f, dtype=np.float32).flatten()
        keypoints = np.reshape(keypoints, (-1,3))
    #print(keypoints)
    return keypoints    

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, metadata_model, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.metadata_model = metadata_model
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs = []
            metadatas = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
                metadata = read_metadata(img_path, self.metadata_model, False)
                metadata = torch.from_numpy(metadata)
                metadata = metadata.unsqueeze(0)
                metadatas.append(metadata)
            imgs = torch.cat(imgs, dim=0)
            # imgs=imgs.permute(1,0,2,3)
            metadatas = torch.cat(metadatas, dim=0)
            return imgs, pid, camid, metadatas

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            # frame_indices = range(num)
            frame_indices = list(range(num))
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            metadatas_list = []
            for indices in indices_list:
                imgs = []
                metadatas = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    metadata = read_metadata(img_path, self.metadata_model, False) ####################
                    metadata = torch.from_numpy(metadata)
                    metadata = metadata.unsqueeze(0)
                    metadatas.append(metadata)
                imgs = torch.cat(imgs, dim=0)
                # imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
                metadatas = torch.cat(metadatas, dim=0)
                metadatas_list.append(metadatas)
            imgs_array = torch.stack(imgs_list)
            metadatas_array = torch.stack(metadatas_list)

            return imgs_array, pid, camid, metadatas_array, img_paths

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_SURFACE(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, metadata_model, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.metadata_model = metadata_model
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        keypoint_conf_thresh = 0.6#999
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs = []
            surfaces = []
            metadatas = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
                # TH  surface
                keypoints = read_keypoint(img_path)
                surface = keypointsSurface(keypoints)
                keypoint_conf = keypointsConfidence(keypoints)
                if keypoint_conf < keypoint_conf_thresh:
                    surface = surface * 0
                #print('surface = ' + str(surface))
                surface = torch.from_numpy(surface)
                surface = surface.unsqueeze(0)
                surfaces.append(surface)
                metadata = read_metadata(img_path, self.metadata_model, False)
                metadata = torch.from_numpy(metadata)
                metadata = metadata.unsqueeze(0)
                metadatas.append(metadata)
            imgs = torch.cat(imgs, dim=0)
            # imgs=imgs.permute(1,0,2,3)
            surfaces = torch.cat(surfaces, dim=0)
            metadatas = torch.cat(metadatas, dim=0)
            return imgs, surfaces, pid, camid, metadatas

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            # frame_indices = range(num)
            frame_indices = list(range(num))
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            surfaces_list = []
            metadatas_list = []
            for indices in indices_list:
                imgs = []
                surfaces = []
                metadatas = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    # TH  surface
                    keypoints = read_keypoint(img_path)
                    surface = keypointsSurface(keypoints)
                    keypoint_conf = keypointsConfidence(keypoints)
                    if keypoint_conf < keypoint_conf_thresh:
                        surface = surface * 0
                    #print('surface = ' + str(surface))
                    surface = torch.from_numpy(surface)
                    surface = surface.unsqueeze(0)
                    surfaces.append(surface)
                    metadata = read_metadata(img_path, self.metadata_model)
                    metadata = torch.from_numpy(metadata)
                    metadata = metadata.unsqueeze(0)
                    metadatas.append(metadata)
                imgs = torch.cat(imgs, dim=0)
                # imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
                surfaces = torch.cat(surfaces, dim=0)
                surfaces_list.append(surfaces)
                metadatas = torch.cat(metadatas, dim=0)
                metadatas_list.append(metadatas)
            imgs_array = torch.stack(imgs_list)
            surfaces_array = torch.stack(surfaces_list)
            metadatas_array = torch.stack(metadatas_list)

            return imgs_array, surfaces_array, pid, camid, metadatas_array, img_paths

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
