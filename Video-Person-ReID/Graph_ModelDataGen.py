from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import Graph_data_manager
from Graph_video_loader import VideoDataset
import transforms as T
import models
from models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from reidtools import visualize_ranked_results  # TH




def testseq(dataset_name, use_gpu):
    
    dataset_root = './video2img/track1_sct_img_test_big/'
    dataset = Graph_data_manager.AICityTrack2(root=dataset_root)


    width = 224
    height = 224
    transform_train = T.Compose([
        T.Random2DTranslation(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False
    seq_len = 4
    num_instance = 4
    train_batch = 32
    test_batch = 1

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=transform_test),
        batch_size=test_batch, shuffle=False, num_workers=4,
        pin_memory=pin_memory, drop_last=False,
    )

    arch = "resnet50ta"
    pretrained_model = "./log/track12_ta224_checkpoint_ep500.pth.tar"


    start_epoch = 0
    print("Initializing model: {}".format(arch))
    dataset.num_train_pids = 517
    if arch=='resnet503d':
        model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=width, sample_height=height, sample_duration=seq_len)
        if not os.path.exists(pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(pretrained_model))
        print("Loading checkpoint from '{}'".format(pretrained_model))
        checkpoint = torch.load(pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key: continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    else:
        if not os.path.exists(pretrained_model):
            model = models.init_model(name=arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
        else:
            model = models.init_model(name=arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
            checkpoint = torch.load(pretrained_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print("Loaded checkpoint from '{}'".format(pretrained_model))
            print("- start_epoch: {}\n- rank1: {}".format(start_epoch, checkpoint['rank1']))

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=0.3)

    lr = 0.0003
    gamma = 0.1
    stepsize = 200
    weight_decay = 5e-04

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
    start_epoch = start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    test(model, queryloader, 'avg', use_gpu, dataset, -1, meta_data_tab=None)

def test(model, queryloader, pool, use_gpu, dataset, epoch, ranks=[1, 5, 10, 20], meta_data_tab = None):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    if False:
        for batch_idx, (imgs, surfaces, pids, camids) in enumerate(queryloader):
            torch.cuda.empty_cache()
            if use_gpu:
                imgs = imgs.cuda()
                surfaces = surfaces.cuda()
            imgs = Variable(imgs, volatile=True)
            surfaces = Variable(surfaces, volatile=True)
            b, n, s, c, h, w = imgs.size()
            b_s, n_s, s_s, d_s = surfaces.size()
            assert(b == b_s and n == n_s and s == s_s)
            if n < 100:
                assert(b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
                surfaces = surfaces.view(b * n, s, -1)
                features = model(imgs, surfaces)
                features = features.view(n, -1)

            else:
                imgs = imgs.data
                imgs.resize_(50, s, c, h, w)
                imgs = imgs.view(50, s, c, h, w)
                imgs = Variable(imgs, volatile=True)
                surfaces = surfaces.data
                surfaces.resize_(50, s, d_s)
                surfaces = surfaces.view(50, s, -1)
                surfaces = Variable(surfaces, volatile=True)
                features = model(imgs, surfaces)
                features = features.view(50, -1)

            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
    else:
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            torch.cuda.empty_cache()
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs, volatile=True)
            b, n, s, c, h, w = imgs.size()
            if n < 100:
                assert(b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
                features = model(imgs)
                features = features.view(n, -1)

            else:
                imgs = imgs.data
                imgs.resize_(50, s, c, h, w)
                imgs = imgs.view(50, s, c, h, w)
                imgs = Variable(imgs, volatile=True)
                features = model(imgs)
                features = features.view(50, -1)

            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features.numpy())
            q_pids.extend(pids.numpy())
            q_camids.extend(camids.numpy())

    qf = np.array(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    np.save("qf3_no_nms_big0510.npy", qf)
    np.save("q_pids3_no_nms_big0510.npy", q_pids)
    np.save("q_camids3_no_nms_big0510.npy", q_camids)


def main():
    seed = 1
    gpu_devices = '0'
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    use_gpu = torch.cuda.is_available()
    use_gpu = True

    if not True:
        sys.stdout = Logger(osp.join('track1_log', 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join('track1_log', 'log_test.txt'))
    print("==========\nArgs:{}\n==========")

    if use_gpu:
        print("Currently using GPU {}".format(gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    dataset = "aictrack2"
    print("Initializing dataset {}".format(dataset))
    testseq(dataset, use_gpu)


if __name__ == '__main__':
    
    main()
