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

import data_manager
from video_loader import VideoDataset, VideoDataset_SURFACE
import transforms as T
import models
from models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss, BatchSoft
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate, evaluate_imgids, dump_matches_imgids, dump_query_result
from samplers import RandomIdentitySampler

from reidtools import visualize_ranked_results  # TH
from re_ranking_metadata import cluster_gallery_soft, re_ranking_metadata_soft, re_ranking_metadata_soft_v2, re_ranking_metadata_soft_v3, compute_metadata_distance_hard, compute_KL_divergence, compute_pred, compute_confusion_weight_old, compute_confusion_weight
from scipy.spatial.distance import cdist
import csv # TH for aggregate_log

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224,
                    help="width of an image (default: 224)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=800, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--bstri', action='store_true', default=False,
                    help="if this is True, bstri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=40, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='/home/jiyang/Workspace/Works/video-person-reid/3dconv-person-reid/pretrained_models/resnet-50-kinetics.pth', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-step', type=int, default=50,
                    help="save model for every N epochs (set to -1 to test after training)") #TH
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')


########################### training

# Additional feature
parser.add_argument('--use-surface', action='store_true', help='use surface feature')

# sample replace
parser.add_argument('--sample-replace', action='store_true', default=False,
                    help="if this is True, sample tracks with replace in training")

# augmentation rotation
parser.add_argument('--aug-rot', action='store_true', default=False,
                    help="if this is True, augmentation rotation is used in training")

# augmentation rotation
parser.add_argument('--augf-surface', action='store_true', default=False,
                    help="if this is True, augmentation in feature space using surface feature is used in training")

########################### testing

# metadata model
parser.add_argument('--metadata-model', type=str, default='', help='metadata model to use')

# meta data
parser.add_argument('--metas', type=str, default='0,1,2', help='metadatas to use, 0) type, 1) brand, 2) color')

# re-ranking
parser.add_argument('--re-ranking', action='store_true', help='perform re-ranking')

# cluster gallery
parser.add_argument('--cluster-gallery', action='store_true', help='cluster gallery features')

########################### misc

# using small query and gallery for debug
parser.add_argument('--use-small-dataset', action='store_true', help='use small query and gallery')

# Visualization
parser.add_argument('--visualize-ranks', action='store_true', help='visualize ranked results, only available in evaluation mode')

# save feature only
parser.add_argument('--feature-only', action='store_true', default=False, help="save feature only")

# evaluate multiple model in a folder
parser.add_argument('--evaluate-multiple', action='store_true', help="evaluation only for all model in the folder ")
parser.add_argument('--pretrained-model-folder', type=str, default='log/', help='folder of the models to evaluate, used with --evaluate-multiple')

########################### read feature

# read from pre-saved feature
parser.add_argument('--load-feature', action='store_true', default=False,
                    help="if this is True, run reid from pre-saved feature")
parser.add_argument('--feature-dir', type=str, default='./feature')


###########################



args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if not (args.evaluate or args.evaluate_multiple):
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    aggregate_log = open(osp.join(args.save_dir, 'aggregate_log.csv'), 'w')
    aggregate_fieldnames = ['epoch', 'metadata_prob_ranges', 'k', 'lr', 'niter', 'r_metadata', 'k1', 'k2', 'lambda_value', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20']
    aggregate_writer = csv.DictWriter(aggregate_log, fieldnames=aggregate_fieldnames)
    aggregate_writer.writeheader()

    print("==========\nArgs:{}\n==========".format(args))

    # load previously inferenced feature
    if args.load_feature:
        print('Load feature')
        return test_feature(aggregate_writer)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)
    if args.use_small_dataset: # TH
        dataset.train = dataset.train_small
        dataset.query = dataset.query_small
        dataset.gallery = dataset.gallery_small

    if args.aug_rot:
        transform_train = T.Compose([
            T.RandomRotation(degrees=(-30, 30)),
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    if args.use_surface:
        trainloader = DataLoader(
            VideoDataset_SURFACE(dataset.train, args.metadata_model, seq_len=args.seq_len, sample='random', transform=transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

        queryloader = DataLoader(
            VideoDataset_SURFACE(dataset.query, args.metadata_model, seq_len=args.seq_len, sample='dense', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        galleryloader = DataLoader(
            VideoDataset_SURFACE(dataset.gallery, args.metadata_model, seq_len=args.seq_len, sample='dense', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )
    else:
        trainloader = DataLoader(
            VideoDataset(dataset.train, args.metadata_model, seq_len=args.seq_len, sample='random', transform=transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

        queryloader = DataLoader(
            VideoDataset(dataset.query, args.metadata_model, seq_len=args.seq_len, sample='dense', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, args.metadata_model, seq_len=args.seq_len, sample='dense', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    print("Initializing model: {}".format(args.arch))
    #dataset.num_train_pids = 666 ############## !!!!!!!!!!!!!!!!!!!!!!!!!! tmp
    if args.arch == 'resnet503d':
        model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key:
                continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    else:
        if not os.path.exists(args.pretrained_model):
            if not args.augf_surface:
                model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
            else:
                model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'}, augf_surface=args.augf_surface)
        else:
            if not args.augf_surface:
                model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
            else:
                model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'}, augf_surface=args.augf_surface)
            
            '''
            print("Loading checkpoint from '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            print("Loaded pretrained weights from '{}'".format(args.pretrained_model))
            '''


            checkpoint = torch.load(args.pretrained_model)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']# + 1
            print("Loaded checkpoint from '{}'".format(args.pretrained_model))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, checkpoint['rank1']))

            '''
            state_dict = {}
            for key in checkpoint['state_dict']:
                if 'fc' in key: continue
                state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict, strict=False)
            '''
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)
    criterion_bstri = BatchSoft(args.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, args.pool, use_gpu, dataset, args.start_epoch, [1, 5, 10, 20], aggregate_writer)
        return

    if args.evaluate_multiple:
        print("Evaluate only")
        print("Evaluate multiple: " + args.pretrained_model_folder)

        pretrained_models = [osp.join(args.pretrained_model_folder, f) for f in os.listdir(args.pretrained_model_folder) if f[:13] == 'checkpoint_ep' and f[-8:] == '.pth.tar']
        pretrained_models = [f for f in pretrained_models if osp.isfile(f)]
        pretrained_models.sort()
        print('Number of pretrained models: %d' % len(pretrained_models))
        for pretrained_model in pretrained_models:
            print('model: ' + pretrained_model)
            if not os.path.exists(pretrained_model):
                raise IOError("Can't find pretrained model: {}".format(pretrained_model))
            epoch = int(osp.split(pretrained_model)[1][13:-8]) - 1
            model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
            checkpoint = torch.load(pretrained_model)
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint from '{}'".format(pretrained_model))
            if use_gpu:
                model = nn.DataParallel(model).cuda()
            test(model, queryloader, galleryloader, args.pool, use_gpu, dataset, epoch, [1, 5, 10, 20], aggregate_writer)
        return 

    if args.feature_only:
        print('Feature only')

    start_time = time.time()
    best_rank1 = -np.inf
    if args.arch == 'resnet503d':
        torch.backends.cudnn.benchmark = False
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))

        train(model, criterion_xent, criterion_htri, criterion_bstri, optimizer, trainloader, use_gpu, epoch)

        if args.stepsize > 0:
            scheduler.step()

        if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu, dataset, epoch, [1, 5, 10, 20], aggregate_writer)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            if args.save_step > 0 and (epoch + 1) % args.save_step == 0 or (epoch + 1) == args.max_epoch:
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    aggregate_log.close()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_htri, criterion_bstri, optimizer, trainloader, use_gpu, epoch):
    model.train()
    losses = AverageMeter()

    if args.use_surface:
        for batch_idx, (imgs, surfaces, pids, _, _metadatas) in enumerate(trainloader):
            if use_gpu:
                imgs, surfaces, pids = imgs.cuda(), surfaces.cuda(), pids.cuda()
            imgs, surfaces, pids = Variable(imgs), Variable(surfaces), Variable(pids)
            if not args.augf_surface:
                outputs, features = model(imgs, surfaces)
                surfaces = None
            else:
                outputs, features, surfaces = model(imgs, surfaces)
            if args.htri_only:
                # only use hard triplet loss to train the network
                loss = criterion_htri(features, pids)
            elif args.bstri:
                xent_loss = criterion_xent(outputs, pids)
                bstri_loss = criterion_bstri(features, pids)
                if epoch < 300 or True:
                    loss = xent_loss + bstri_loss
                else:
                    loss = bstri_loss
            else:
                # combine hard triplet loss with cross entropy loss
                xent_loss = criterion_xent(outputs, pids)
                htri_loss = criterion_htri(features, pids, surfaces)
                if epoch < 300 or True:
                    loss = xent_loss + htri_loss
                else:
                    loss = htri_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.data[0], pids.size(0))

            if (batch_idx + 1) % args.print_freq == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))
    else:
        for batch_idx, (imgs, pids, _, _metadatas) in enumerate(trainloader):
            if use_gpu:
                imgs, pids = imgs.cuda(), pids.cuda()
            imgs, pids = Variable(imgs), Variable(pids)
            outputs, features = model(imgs)
            if args.htri_only:
                # only use hard triplet loss to train the network
                loss = criterion_htri(features, pids)
            else:
                # combine hard triplet loss with cross entropy loss
                xent_loss = criterion_xent(outputs, pids)
                htri_loss = criterion_htri(features, pids)
                loss = xent_loss + htri_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.data[0], pids.size(0))

            if (batch_idx + 1) % args.print_freq == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))


def test(model, queryloader, galleryloader, pool, use_gpu, dataset, epoch, ranks=[1, 5, 10, 20], aggregate_writer=None):
    model.eval()

    qf, q_pids, q_camids, q_imgpaths = [], [], [], []
    q_metadatas = []
    if args.use_surface:
        for batch_idx, (imgs, surfaces, pids, camids, metadatas, img_paths) in enumerate(queryloader):
            torch.cuda.empty_cache()
            if use_gpu:
                imgs = imgs.cuda()
                surfaces = surfaces.cuda()
            imgs = Variable(imgs, volatile=True)
            surfaces = Variable(surfaces, volatile=True)
            # b=1, n=number of clips, s=16
            # print(imgs.size())
            # print(surfaces.size())
            b, n, s, c, h, w = imgs.size()
            b_s, n_s, s_s, d_s = surfaces.size()
            assert b == b_s and n == n_s and s == s_s
            b_m, n_m, s_m, d_m = metadatas.size()
            assert b == b_m and n == n_m and s == s_m
            if n < 100:
                # print(imgs.size())
                assert(b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
                surfaces = surfaces.view(b * n, s, -1)
                features = model(imgs, surfaces)
                features = features.view(n, -1)
                metadatas = metadatas.view(b * n * s, -1)

            else:
                imgs = imgs.data
                imgs.resize_(50, s, c, h, w)
                imgs = imgs.view(50, s, c, h, w)
                imgs = Variable(imgs, volatile=True)
                #imgs = imgs.view(b*n, s, c, h, w)
                surfaces = surfaces.data
                surfaces.resize_(50, s, d_s)
                surfaces = surfaces.view(50, s, -1)
                surfaces = Variable(surfaces, volatile=True)
                features = model(imgs, surfaces)
                features = features.view(50, -1)
                #metadatas = metadatas.data
                metadatas.resize_(50, s, d_m)
                metadatas = metadatas.view(50*s, -1)
                #metadatas = Variable(metadatas, volatile=True)
                

            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            #print(img_paths)
            img_paths = np.array(img_paths).reshape(1,-1).tolist()
            #print(img_paths)
            q_imgpaths.extend(img_paths)

            metadatas = torch.mean(metadatas, 0)
            q_metadatas.append(metadatas)
    else:
        for batch_idx, (imgs, pids, camids, metadatas, img_paths) in enumerate(queryloader):
            torch.cuda.empty_cache()
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs, volatile=True)
            # b=1, n=number of clips, s=16
            #print('imgs.size() = ' + str(imgs.size()))
            b, n, s, c, h, w = imgs.size()
            b_m, n_m, s_m, d_m = metadatas.size()
            assert b == b_m and n == n_m and s == s_m
            if n < 100:
                # print(imgs.size())
                assert(b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
                features = model(imgs)
                features = features.view(n, -1)
                metadatas = metadatas.view(b * n * s, -1)

            else:
                imgs = imgs.data
                imgs.resize_(50, s, c, h, w)
                imgs = imgs.view(50, s, c, h, w)
                imgs = Variable(imgs, volatile=True)
                #print('imgs.size() = ' + str(imgs.size()))
                #imgs = imgs.view(b*n, s, c, h, w)
                features = model(imgs)
                features = features.view(50, -1)
                #metadatas = metadatas.data
                metadatas.resize_(50, s, d_m)
                metadatas = metadatas.view(50*s, -1)
                #metadatas = Variable(metadatas, volatile=True)

            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            #print(img_paths)
            img_paths = np.array(img_paths).reshape(1,-1).tolist()
            #print(img_paths)
            q_imgpaths.extend(img_paths)
            #print('len(qf)')
            #print(len(qf))

            metadatas = torch.mean(metadatas, 0)
            q_metadatas.append(metadatas)

    #print('len(qf)')
    #print(len(qf))
    #print('qf = torch.stack(qf)')
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    #print('q_pids:')
    #for pid in q_pids:
    #    print(pid)
    q_camids = np.asarray(q_camids)
    #print('len(q_metadatas)')
    #print(len(q_metadatas))
    #print('q_metadatas = torch.stack(q_metadatas)')
    q_metadatas = torch.stack(q_metadatas)
    q_metadatas = q_metadatas.numpy()
    print('q_metadatas.shape = ' + str(q_metadatas.shape))

    # for debug
    #qf = qf.numpy()
    #for i in range(qf.shape[0]):
    #    if i % 10 == 0:
    #        print(i)
    #        print(qf[i])
    #return

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))


    gf, g_pids, g_camids, g_imgpaths = [], [], [], []
    g_metadatas = []
    if args.use_surface:
        for batch_idx, (imgs, surfaces, pids, camids, metadatas, img_paths) in enumerate(galleryloader):
            torch.cuda.empty_cache()
            if use_gpu:
                imgs = imgs.cuda()
                surfaces = surfaces.cuda()
            imgs = Variable(imgs, volatile=True)
            surfaces = Variable(surfaces, volatile=True)
            b, n, s, c, h, w = imgs.size()
            b_s, n_s, s_s, d_s = surfaces.size()
            assert b == b_s and n == n_s and s == s_s
            b_m, n_m, s_m, d_m = metadatas.size()
            assert b == b_m and n == n_m and s == s_m
            if n < 100:
                # print(imgs.size())
                assert(b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
                surfaces = surfaces.view(b * n, s, -1)
                features = model(imgs, surfaces)
                features = features.view(n, -1)
                metadatas = metadatas.view(b * n * s, -1)

            else:
                imgs = imgs.data
                imgs.resize_(50, s, c, h, w)
                imgs = imgs.view(50, s, c, h, w)
                imgs = Variable(imgs, volatile=True)
                #imgs = imgs.view(b*n, s, c, h, w)
                surfaces = surfaces.data
                surfaces.resize_(50, s, d_s)
                surfaces = surfaces.view(50, s, -1)
                surfaces = Variable(surfaces, volatile=True)
                features = model(imgs, surfaces)
                features = features.view(50, -1)
                metadatas.resize_(50, s, d_m)
                metadatas = metadatas.view(50*s, -1)

            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            #print(img_paths)
            img_paths = np.array(img_paths).reshape(1,-1).tolist()
            #print(img_paths)
            g_imgpaths.extend(img_paths)
            #import pdb; pdb.set_trace()

            metadatas = torch.mean(metadatas, 0) # average pooling
            #metadatas, _ = torch.max(metadatas, 0) # max pooling
            g_metadatas.append(metadatas)
    else:
        for batch_idx, (imgs, pids, camids, metadatas, img_paths) in enumerate(galleryloader):
            torch.cuda.empty_cache()
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs, volatile=True)
            b, n, s, c, h, w = imgs.size()
            b_m, n_m, s_m, d_m = metadatas.size()
            assert b == b_m and n == n_m and s == s_m
            if n < 100:
                # print(imgs.size())
                assert(b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
                features = model(imgs)
                features = features.view(n, -1)
                metadatas = metadatas.view(b * n * s, -1)

            else:
                imgs = imgs.data
                imgs.resize_(50, s, c, h, w)
                imgs = imgs.view(50, s, c, h, w)
                imgs = Variable(imgs, volatile=True)
                #imgs = imgs.view(b*n, s, c, h, w)
                features = model(imgs)
                features = features.view(50, -1)
                metadatas.resize_(50, s, d_m)
                metadatas = metadatas.view(50*s, -1)

            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            #print(img_paths)
            img_paths = np.array(img_paths).reshape(1,-1).tolist()
            #print(img_paths)
            g_imgpaths.extend(img_paths)
            #print('len(gf)')
            #print(len(gf))

            metadatas = torch.mean(metadatas, 0) # average pooling
            #metadatas, _ = torch.max(metadatas, 0) # max pooling
            g_metadatas.append(metadatas)
    
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    #print('g_pids:')
    #for pid in g_pids:
    #    print(pid)
    g_camids = np.asarray(g_camids)
    g_metadatas = torch.stack(g_metadatas)
    g_metadatas = g_metadatas.numpy()
    print('g_metadatas.shape = ' + str(g_metadatas.shape))

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    #print('q_pids.shape = ' + str(q_pids.shape))
    #print('g_pids.shape = ' + str(g_pids.shape))
    #print('q_camids.shape = ' + str(q_camids.shape))
    #print('g_camids.shape = ' + str(g_camids.shape))
    #print('len(q_imgids) = ' + str(len(q_imgids)))
    #print('len(g_imgids) = ' + str(len(g_imgids)))
    #print('q_imgids[0]:')
    #print(q_imgids[0])
    #print('g_imgids[0]:')
    #print(g_imgids[0])

    # extract filename and imgid
    #q_imgids, _q_pids, _q_camid = zip(*dataset.query)
    #g_imgids, _g_pids, _g_camid = zip(*dataset.gallery)
    q_imgids = [[osp.split(img)[1][:-4].split('_')[-1] for img in q_imgid] for q_imgid in q_imgpaths]
    g_imgids = [[osp.split(img)[1][:-4].split('_')[-1] for img in g_imgid] for g_imgid in g_imgpaths]
    #print('q_imgids:')
    #print(q_imgids)
    #print('g_imgids:')
    #print(g_imgids)


    # save feature and additional information
    feature_dir = osp.join(args.save_dir, 'feature_ep%04d' % (epoch + 1))
    if not osp.isdir(feature_dir):
        os.mkdir(feature_dir)
    #qf_np = np.array(qf)
    #gf_np = np.array(gf)
    #np.save(osp.join(feature_dir, 'qf.npy'), qf_np)
    #np.save(osp.join(feature_dir, 'gf.npy'), gf_np)
    qf = np.array(qf)
    gf = np.array(gf)
    np.save(osp.join(feature_dir, 'qf.npy'), qf)
    np.save(osp.join(feature_dir, 'gf.npy'), gf)
    np.save(osp.join(feature_dir, 'q_pids.npy'), q_pids)
    np.save(osp.join(feature_dir, 'g_pids.npy'), g_pids)
    np.save(osp.join(feature_dir, 'q_camids.npy'), q_camids)
    np.save(osp.join(feature_dir, 'g_camids.npy'), g_camids)
    np.save(osp.join(feature_dir, 'q_metadatas_%s.npy'%args.metadata_model), q_metadatas)
    np.save(osp.join(feature_dir, 'g_metadatas_%s.npy'%args.metadata_model), g_metadatas)
    with open(osp.join(feature_dir, 'q_imgids.txt'), 'w') as f:
        for q_imgid in q_imgids:
            f.write(' '.join(q_imgid)+'\n')
    with open(osp.join(feature_dir, 'g_imgids.txt'), 'w') as f:
        for g_imgid in g_imgids:
            f.write(' '.join(g_imgid)+'\n')
    with open(osp.join(feature_dir, 'q_imgpaths.txt'), 'w') as f:
        for q_imgid in q_imgpaths:
            f.write(' '.join(q_imgid)+'\n')
    with open(osp.join(feature_dir, 'g_imgpaths.txt'), 'w') as f:
        for g_imgid in g_imgpaths:
            f.write(' '.join(g_imgid)+'\n')
    if args.feature_only:
        return 0

    return evaluate_feature(qf, gf, q_metadatas, g_metadatas, q_pids, g_pids, q_camids, g_camids, q_imgids, g_imgids, q_imgpaths, g_imgpaths, aggregate_writer, epoch, ranks)

def test_feature(aggregate_writer):
    # read pids, camids, imgids, feature and metadata from file
    q_imgids = []
    with open(osp.join(args.feature_dir, 'q_imgids.txt'), 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            q_imgids.append(line)
    g_imgids = []
    with open(osp.join(args.feature_dir, 'g_imgids.txt'), 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            g_imgids.append(line)
    q_imgpaths = []
    #with open(osp.join(args.feature_dir, 'q_imgpaths.txt'), 'r') as f:
    #    for line in f:
    #        line = line.strip().split(' ')
    #        q_imgpaths.append(line)
    g_imgpaths = []
    #with open(osp.join(args.feature_dir, 'g_imgpaths.txt'), 'r') as f:
    #    for line in f:
    #        line = line.strip().split(' ')
    #        g_imgpaths.append(line)
    num_q = len(q_imgids)
    num_g = len(g_imgids)
    print('(num_q, num_g) = (%d, %d)'%(num_q, num_g))

    q_pids = np.load(osp.join(args.feature_dir, 'q_pids.npy'))
    g_pids = np.load(osp.join(args.feature_dir, 'g_pids.npy'))
    q_camids = np.load(osp.join(args.feature_dir, 'q_camids.npy'))
    g_camids = np.load(osp.join(args.feature_dir, 'g_camids.npy'))
    qf = np.load(osp.join(args.feature_dir, 'qf.npy'))
    gf = np.load(osp.join(args.feature_dir, 'gf.npy'))   
    if osp.isfile('./metadata/q_metadatas_%s.npy'%args.metadata_model) and osp.isfile('./metadata/g_metadatas_%s.npy'%args.metadata_model):
        q_metadatas = np.load('./metadata/q_metadatas_%s.npy'%args.metadata_model)
        g_metadatas = np.load('./metadata/g_metadatas_%s.npy'%args.metadata_model)
    else:
        q_metadatas = np.load(osp.join(args.feature_dir, 'q_metadatas_%s.npy'%args.metadata_model))
        g_metadatas = np.load(osp.join(args.feature_dir, 'g_metadatas_%s.npy'%args.metadata_model))
    print('q_pids.shape = ' + str(q_pids.shape))
    print('g_pids.shape = ' + str(g_pids.shape))
    print('q_camids.shape = ' + str(q_camids.shape))
    print('g_camids.shape = ' + str(g_camids.shape))
    print('qf.shape = ' + str(qf.shape))
    print('gf.shape = ' + str(gf.shape))
    print('q_metadatas.shape = ' + str(q_metadatas.shape))
    print('g_metadatas.shape = ' + str(g_metadatas.shape))

    ranks=[1, 5, 10, 20]
    return evaluate_feature(qf, gf, q_metadatas, g_metadatas, q_pids, g_pids, q_camids, g_camids, q_imgids, g_imgids, q_imgpaths, g_imgpaths, aggregate_writer, -1, ranks)

def evaluate_feature(qf, gf, q_metadatas, g_metadatas, q_pids, g_pids, q_camids, g_camids, q_imgids, g_imgids, q_imgpaths, g_imgpaths, aggregate_writer, epoch, ranks):
    # metadata settings
    if args.metadata_model:
        #confusion_mats_obj = np.load('./metadata/cm_%s.npy'%args.metadata_model, encoding='latin1')
        confusion_file = './metadata/cm_%s_normalized.npy'%args.metadata_model
        if osp.isfile(confusion_file):
            confusion_mats_obj = np.load(confusion_file, encoding='latin1')
        else:
            print('confusion matrix file not found: %s'%confusion_file)
            print('use identity matrix as confusion matrix')
            confusion_mats_obj = np.load('./metadata/cm_%seye.npy'%args.metadata_model[:2], encoding='latin1')
        if args.metadata_model[:2] == 'v1':
            default_metadata_prob_ranges = [(0,6), (6,18), (18,26)]
            confusion_mats = dict()
            for i, prob_range in enumerate(default_metadata_prob_ranges):
                confusion_mats[prob_range] = confusion_mats_obj[i]
        elif args.metadata_model[:2] == 'v2':
            default_metadata_prob_ranges = [(0,7), (7,37), (37,46)]
            confusion_mats = dict()
            for i, prob_range in enumerate(default_metadata_prob_ranges):
                confusion_mats[prob_range] = confusion_mats_obj[i]
        else:
            print('invalid metadata_model, should be v1xx or v2xx but found %s'%args.metadata_model)
            import sys
            sys.exit()

        metas = args.metas.split(',')
        metas = [int(m) for m in metas if len(m) > 0]
        print('metas: ' + str(metas))
        metadata_prob_ranges = []
        for im in range(len(default_metadata_prob_ranges)):
            if im in metas:
                metadata_prob_ranges.append(default_metadata_prob_ranges[im])
        print('metadata_prob_ranges: ' + str(metadata_prob_ranges))
        

    if args.cluster_gallery and args.metadata_model:
        k, lr, niter = 5, 0.5, 9
        print('apply cluster gallery')
        print('(k, lr, niter): (%d, %f, %d)' % (k, lr, niter))
        gf = cluster_gallery_soft(gf, g_metadatas, metadata_prob_ranges, k, lr, niter)
    else:
        k, lr, niter = -1, -1, -1


    # compute original distance
    print('computing original distance')
    query_num = qf.shape[0]
    gallery_num = gf.shape[0]
    all_num = query_num + gallery_num
    feat = np.append(qf, gf, axis=0)
    all_metadatas = np.append(q_metadatas, g_metadatas, axis=0)
    feat = feat.astype(np.float16)
    MemorySave = False
    if MemorySave:
        original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it, ] = np.power(cdist(feat[i:it, ], feat), 2).astype(np.float16)
            else:
                original_dist[i:, :] = np.power(cdist(feat[i:, ], feat), 2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat, feat).astype(np.float16)
        original_dist = np.power(original_dist, 2).astype(np.float16)
    del feat

    # apply meta data
    m_num = len(metadata_prob_ranges)
    for p_begin, p_end in metadata_prob_ranges:
        assert (p_begin, p_end) in confusion_mats

    conf_pred = np.ones((all_num, all_num, m_num), dtype=np.float32)
    use_KL = False
    if use_KL:
        print('computing KL divergence')
        KL_div_U = compute_KL_divergence(all_metadatas, np.ones(all_metadatas.shape, dtype=np.float32), metadata_prob_ranges)
        for im, (p_begin, p_end) in enumerate(metadata_prob_ranges):
            conf_pred[:,:,im] = KL_div_U[:,:,im] * np.transpose(KL_div_U[:,:,im]) / (np.log(p_end - p_begin)*np.log(p_end - p_begin))
    pred = compute_pred(all_metadatas, metadata_prob_ranges)
    confusion_dist = np.zeros((all_num, all_num, m_num), dtype=np.float32)
    for im, (p_begin, p_end) in enumerate(metadata_prob_ranges):
        confusion_weight = compute_confusion_weight(pred[:,im], pred[:,im], confusion_mats[(p_begin, p_end)])
        confusion_dist[:,:,im] = -np.log(confusion_weight + 1e-4) / np.log(p_end-p_begin)
    metadata_dist = conf_pred * confusion_dist
    metadata_dist = np.sum(metadata_dist, axis=2)


    ###################################################3
    #distmat += 1000 * compute_metadata_distance_hard(qf, gf, metadata_prob_ranges)
    ##########################################


    print("Computing CMC, mAP and matches_imgids - top 100")
    distmat = original_dist[0:query_num, query_num:all_num]
    cmc, mAP, matches_imgids, matches_imgids_FP, matches_gt_pred = evaluate_imgids(distmat, q_pids, g_pids, q_camids, g_camids, q_imgids, g_imgids, 50, 100)
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    if aggregate_writer is not None:
        aggregate_writer.writerow({'epoch': str(epoch+1), 'metadata_prob_ranges': '', 'k': str(k), 'lr': str(lr), 'niter': str(niter), 'r_metadata': '', 'k1': '', 'k2': '', 'lambda_value': '', 'mAP': str(mAP), 'Rank-1': str(cmc[0]), 'Rank-5': str(cmc[4]), 'Rank-10': str(cmc[9]), 'Rank-20': str(cmc[19])})

    # dump txt result
    dump_matches_imgids(osp.join(args.save_dir, 'dist_%04d' % (epoch + 1)), matches_imgids)
    dump_matches_imgids(osp.join(args.save_dir, 'dist_%04d_FP' % (epoch + 1)), matches_imgids_FP)
    dump_query_result(osp.join(args.save_dir, 'track2.txt_%04d' % (epoch + 1)), matches_imgids)

    # visualization
    if args.visualize_ranks:
        tracklets_query = []
        for i in range(len(q_imgpaths)):
            img_paths = q_imgpaths[i]
            pid = q_pids[i]
            camid = q_camids[i]
            tracklets_query.append((img_paths, pid, camid))
        tracklets_gallery = []
        for i in range(len(g_imgpaths)):
            img_paths = g_imgpaths[i]
            pid = g_pids[i]
            camid = g_camids[i]
            tracklets_query.append((img_paths, pid, camid))
        visualize_ranked_results(
            distmat, (tracklets_query, tracklets_gallery),
            save_dir=osp.join(args.save_dir, 'ranked_results_%04d' % (epoch + 1)), topk=20)  # TH

    # re-ranking
    if args.re_ranking and args.metadata_model:
        # zhangping doing re-rank

        print('Run re-ranking')

        max_mAP = 0

        fMAP = open(osp.join(args.save_dir, 'myMAPLog'), 'w')
        fMAP.close()

        f = open(osp.join(args.save_dir, 'myLog'), 'w')

        #for k1 in range(4,32,2):
        #for k1 in range(2,32,1):
        for k1 in range(4,5,1):
            for k2 in range(4,k1+1,1):
                #for lambda_value in np.arange(0.2,1.1,0.1):
                for lambda_value in np.arange(0.5,0.6,0.1):
                    print('now try different k1, k2 and lambda_value')
                    print('k1 = ', k1)
                    print('k2 = ', k2)
                    print('lambda_value = ', lambda_value)
                    #r_metadata = 10.0
                    #r_metadatas = [0.001, 0.002, 0.005, 0.01, 0.02]
                    #for r_metadata in np.arange(0.5, 20, 0.5):
                    for r_metadata in np.arange(0.01, 20, 0.01):
                    #for r_metadata in np.arange(0.01, 0.51, 0.01):
                    #for r_metadata in r_metadatas:
                        print('r_metadata = ', r_metadata)

                        #final_dist = re_ranking_metadata_soft(qf, gf, q_metadatas, g_metadatas, metadata_prob_ranges, k1, k2, lambda_value)
                        #final_dist = re_ranking_metadata_soft_v2(qf, gf, q_metadatas, g_metadatas, confusion_mats, metadata_prob_ranges, k1, k2, lambda_value)
                        final_dist = re_ranking_metadata_soft_v3(original_dist, metadata_dist, query_num, all_num, r_metadata, k1, k2, lambda_value)
                        
                        
                        print('after re-ranking, the final_dist.shape is: ', final_dist.shape)
                        # ping evaluate after re-ranking
                        #print("Computing CMC and mAP after re-ranking")
                        # final_dist = [qg], instead of [[qq,qg],[gq,gg]]
                        #cmc, mAP = evaluate(final_dist, q_pids, g_pids, q_camids, g_camids)
                        print("Computing CMC, mAP and matches_imgids after re-ranking - top 100")
                        cmc, mAP, matches_imgids, matches_imgids_FP, matches_gt_pred = evaluate_imgids(final_dist, q_pids, g_pids, q_camids, g_camids, q_imgids, g_imgids, 50, 100)
                        print("after re-ranking Results ----------")
                        print("after re-ranking: mAP: {:.1%}".format(mAP))
                        print("after re-ranking: CMC curve")
                        for r in ranks:
                            print("after re-ranking: Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
                        print("------------------")
                        if aggregate_writer is not None:
                            aggregate_writer.writerow({'epoch': str(epoch+1), 'metadata_prob_ranges': ';'.join([(str(rb)+'-'+str(re)) for rb,re in metadata_prob_ranges]), 'k': str(k), 'lr': str(lr), 'niter': str(niter), 'r_metadata': str(r_metadata), 'k1': str(k1), 'k2': str(k2), 'lambda_value': str(lambda_value), 'mAP': str(mAP), 'Rank-1': str(cmc[0]), 'Rank-5': str(cmc[4]), 'Rank-10': str(cmc[9]), 'Rank-20': str(cmc[19])})
                        if mAP - max_mAP > 0:
                            max_mAP = mAP
                            print('new max_mAP')
                            print('\n')
                            print('right now the max_mAP is: ', max_mAP)
                            print('\n')
                            print('and the correspondding cmc[0] is: ', cmc[0])
                            with open(osp.join(args.save_dir, 'myMAPLog'), 'a') as fMAP:
                                fMAP.write('max_mAP is: ' + str(max_mAP) + ' correspondding cmc[0] is: ' + str(cmc[0]) + 
                                        ' the parameters are: [k1 = ' + str(k1) + ' k2 = ' + str(k2) + ' lambda_value = ' + str(lambda_value) + ']' )
                                fMAP.write('\n')
                        f.write('k1 = '+str(k1) + ' k2 = ' + str(k2) + ' lambda_value = ' + str(lambda_value))
                        f.write('\n')
                        f.write('mAP = '+str(mAP) + ' CMC[0] = ' + str(cmc[0]))
                        f.write('\n')

                        # dump txt result
                        dump_matches_imgids(osp.join(args.save_dir, 'dist_rerank-%d-%d-%.2f_%04d' % (k1, k2, lambda_value, epoch + 1)), matches_imgids)
                        dump_matches_imgids(osp.join(args.save_dir, 'dist_rerank-%d-%d-%.2f_%04d_FP' % (k1, k2, lambda_value, epoch + 1)), matches_imgids_FP)
                        dump_query_result(osp.join(args.save_dir, 'track2.txt-%d-%d-%.2f_%04d' % (k1, k2, lambda_value, epoch + 1)), matches_imgids)
        
        print("------------------")
        print('\n')
        print('\n')
        print('overall the max_mAP is: ', max_mAP)
        print('\n')

        f.close()   

    return cmc[0]




if __name__ == '__main__':
    main()
