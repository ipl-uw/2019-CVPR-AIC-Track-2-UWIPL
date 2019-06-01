from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
#from models.lightcnn import * # JR
#import torch.nn.parallel.data_parallel as DataParallel # JR
from collections import OrderedDict # JR

__all__ = ['ResNet50TP', 'ResNet50TA', 'myResNet50TA', 'ResNet50RNN', 'ResNet50TP_ORIENTATION', 'ResNet50TP_ORIENTATION_IOU', 'ResNet50TA_ORIENTATION', 'ResNet50TA_ORIENTATION_IOU', 'ResNet50TA_SURFACE', 'ResNet50TA_SURFACE_NU', 'ResNet50TA_SURFACE_NU4', 'ResNet50TA_SURFACE_NU2', 'ResNet50TA_SURFACE_NU2F1', 'ResNet50TA_SURFACE_N1', 'ResNet50TA_SURFACE_N2']


class ResNet50TP_ORIENTATION_IOU(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP_ORIENTATION_IOU, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim + 2, num_classes)

    def forward(self, x, orientation, frame_IOU):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.permute(0, 2, 1)
        f = F.avg_pool1d(x, t)
        f = f.view(b, self.feat_dim)

        orientation = orientation.view(f.size(0), -1)
        orientation = orientation.float()
        frame_IOU = frame_IOU.view(f.size(0), -1)
        frame_IOU = frame_IOU.float()
        f = torch.cat((f, orientation, frame_IOU), 1)

        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TP_ORIENTATION(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP_ORIENTATION, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim + 1, num_classes)

    def forward(self, x, orientation):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.permute(0, 2, 1)
        f = F.avg_pool1d(x, t)
        f = f.view(b, self.feat_dim)
        #f = torch.LongTensor(f)
        #f = f.long()
        #import pdb; pdb.set_trace()
        orientation = orientation.view(f.size(0), -1)
        #import pdb; pdb.set_trace()
        orientation = orientation.float()
        #import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        f = torch.cat((f, orientation), 1)
        #import pdb; pdb.set_trace()

        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.permute(0, 2, 1)
        f = F.avg_pool1d(x, t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA_ORIENTATION_IOU1(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_ORIENTATION_IOU, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim+8*8, num_classes) #seq leng=4 is +8
        self.classifier = nn.Linear(self.feat_dim + 8, num_classes)  # seq leng=4 is +8
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5, 3])  # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x, orientation, frame_IOU):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        # print("X2",x.shape)
        #import pdb; pdb.set_trace()
        x = self.base(x)

        # print("X3",x.shape)
        # print("Ori",orientation.shape)

        #import pdb; pdb.set_trace()
        a = F.relu(self.attention_conv(x))
        # print("a1",a.shape)
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        #import pdb; pdb.set_trace()
        a = F.relu(self.attention_tconv(a))
        # print("a2",a.shape)
        #import pdb; pdb.set_trace()
        a = a.view(b, t)
        # print("a3",a.shape)
        x = F.avg_pool2d(x, x.size()[2:])
        # print("x1",x.shape)
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        #import pdb; pdb.set_trace()
        #x = x.view(b,-1)

        #import pdb; pdb.set_trace()
        #x = torch.cat((x,orientation,frame_IOU),1)
        # print("a4",a.shape)
        x = x.view(b, t, -1)
        # print("x2",x.shape)
        orientation = orientation.view(x.size(0), -1)
        # print("ori2",orientation.shape)
        #import pdb; pdb.set_trace()
        orientation = orientation.float()
        frame_IOU = frame_IOU.view(x.size(0), -1)
        frame_IOU = frame_IOU.float()
        for i in range(1, t + 1):
            orientation = torch.stack((orientation, orientation))
            frame_IOU = torch.stack((frame_IOU, frame_IOU))
        orientation = orientation.view(b, t, -1)
        frame_IOU = frame_IOU.view(b, t, -1)
        #import pdb; pdb.set_trace()
        x = torch.cat((x, orientation, frame_IOU), 2)

        #import pdb; pdb.set_trace()

        a = torch.unsqueeze(a, -1)
        #a = a.expand(b, t, self.feat_dim)
        a = a.expand(b, t, x.size(2))
        #import pdb; pdb.set_trace()
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        #f = att_x.view(b,self.feat_dim)
        f = att_x.view(b, x.size(2))
        #import pdb; pdb.set_trace()
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA_ORIENTATION_IOU(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_ORIENTATION_IOU, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim + 8, num_classes)  # seq leng=4 is +8   =8 is 8*8
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5, 3])  # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x, orientation, frame_IOU):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        #import pdb; pdb.set_trace()
        x = self.base(x)
        #import pdb; pdb.set_trace()
        a = F.relu(self.attention_conv(x))
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        #import pdb; pdb.set_trace()
        a = F.relu(self.attention_tconv(a))
        #import pdb; pdb.set_trace()
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        #import pdb; pdb.set_trace()
        #x = x.view(b,-1)

        #import pdb; pdb.set_trace()
        #x = torch.cat((x,orientation,frame_IOU),1)
        x = x.view(b, t, -1)
        orientation = orientation.view(x.size(0), -1)
        orientation = orientation.float()
        frame_IOU = frame_IOU.view(x.size(0), -1)
        frame_IOU = frame_IOU.float()
        for i in range(1, t + 1):
            orientation = torch.stack((orientation, orientation))
            frame_IOU = torch.stack((frame_IOU, frame_IOU))
        orientation = orientation.view(b, t, -1)
        frame_IOU = frame_IOU.view(b, t, -1)
        #import pdb; pdb.set_trace()
        x = torch.cat((x, orientation, frame_IOU), 2)

        #import pdb; pdb.set_trace()

        a = torch.unsqueeze(a, -1)
        #a = a.expand(b, t, self.feat_dim)
        a = a.expand(b, t, x.size(2))
        #import pdb; pdb.set_trace()
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        #f = att_x.view(b,self.feat_dim)
        f = att_x.view(b, x.size(2))
        #import pdb; pdb.set_trace()
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA_ORIENTATION(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_ORIENTATION, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim + 1, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 4])  # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x, orientation):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)

        orientation = orientation.view(x.size(0), -1)
        orientation = orientation.float()
        orientation = torch.unsqueeze(orientation, -1)
        x = torch.cat((x, orientation), 1)
        #import pdb; pdb.set_trace()
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA_SURFACE(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, augf_surface=False, **kwargs):
        super(ResNet50TA_SURFACE, self).__init__()
        self.loss = loss
        self.augf_surface = augf_surface
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 7])  # 7,7 cooresponds to 224, 224 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim, 1, 3, padding=1)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #m = nn.ReLU6() ####### !!!!! optional scale
        #a = m(self.attention_conv(x)) ####### !!!!! optional scale
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        #surface = torch.mul(surface, 6) ####### !!!!! optional scale
        a = torch.cat((a, surface), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.augf_surface:
            surface = surface.permute(0, 2, 1)
            surface = F.avg_pool1d(surface, kernel_size=t)
            surface = surface.view(b, self.surface_dim)
        
        if self.loss == {'xent'}:
            return y if not self.augf_surface else (y, surface)
        elif self.loss == {'xent', 'htri'}:
            return (y, f) if not self.augf_surface else (y, f, surface)
        elif self.loss == {'cent'}:
            return (y, f) if not self.augf_surface else (y, f, surface)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TA_SURFACE_NU(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_SURFACE_NU, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 7])  # 7,7 cooresponds to 224, 224 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim, 1, 3, padding=1)
        self.nu_surface = nn.Linear(self.surface_dim, self.feat_dim, bias=True)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #m = nn.ReLU6() ####### !!!!! optional scale
        #a = m(self.attention_conv(x)) ####### !!!!! optional scale
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        #surface = torch.mul(surface, 6) ####### !!!!! optional scale
        a = torch.cat((a, surface), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        ## surface_nu
        surface = surface.view(b*t, self.surface_dim)
        surface_nu = self.nu_surface(surface)
        surface_nu = surface_nu.view(b, t, -1)
        #surface_nu = torch.mul(surface_nu, 100) # optional scaling
        x = x.add(surface_nu)

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TA_SURFACE_NU4(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_SURFACE_NU4, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 7])  # 7,7 cooresponds to 224, 224 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim, 1, 3, padding=1)
        self.nu_surface = nn.Linear(self.surface_dim, self.feat_dim, bias=True)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #m = nn.ReLU6() ####### !!!!! optional scale
        #a = m(self.attention_conv(x)) ####### !!!!! optional scale
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        #surface = torch.mul(surface, 6) ####### !!!!! optional scale
        a = torch.cat((a, surface), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        ## surface_nu
        surface = surface.view(b*t, self.surface_dim)
        surface_nu = F.relu(self.nu_surface(surface)) ### the only difference
        surface_nu = surface_nu.view(b, t, -1)
        #surface_nu = torch.mul(surface_nu, 100) # optional scaling
        x = x.add(surface_nu)

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TA_SURFACE_NU2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_SURFACE_NU2, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.surface_dim_middle = 256  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 7])  # 7,7 cooresponds to 224, 224 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim, 1, 3, padding=1)
        self.nu_surface_1 = nn.Linear(self.surface_dim, self.surface_dim_middle, bias=True)
        self.nu_surface_2 = nn.Linear(self.surface_dim_middle, self.feat_dim, bias=True)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #m = nn.ReLU6() ####### !!!!! optional scale
        #a = m(self.attention_conv(x)) ####### !!!!! optional scale
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        #surface = torch.mul(surface, 6) ####### !!!!! optional scale
        a = torch.cat((a, surface), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        ## surface_nu
        surface = surface.view(b*t, self.surface_dim)
        surface_nu = F.relu(self.nu_surface_1(surface))
        surface_nu = F.relu(self.nu_surface_2(surface_nu))
        surface_nu = surface_nu.view(b, t, -1)
        #surface_nu = torch.mul(surface_nu, 100) # optional scaling
        x = x.add(surface_nu)

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TA_SURFACE_NU2F1(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_SURFACE_NU2F1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.surface_dim_middle = 256  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 7])  # 7,7 cooresponds to 224, 224 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim, 1, 3, padding=1)
        self.nu_surface_1 = nn.Linear(self.surface_dim, self.surface_dim_middle, bias=True)
        self.nu_surface_2 = nn.Linear(self.surface_dim_middle, self.feat_dim, bias=True)
        self.nu_surface_f1 = nn.Linear(self.feat_dim, self.feat_dim, bias=True)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #m = nn.ReLU6() ####### !!!!! optional scale
        #a = m(self.attention_conv(x)) ####### !!!!! optional scale
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        #surface = torch.mul(surface, 6) ####### !!!!! optional scale
        a = torch.cat((a, surface), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        ## surface_nu
        surface = surface.view(b*t, self.surface_dim)
        surface_nu = F.relu(self.nu_surface_1(surface))
        surface_nu = F.relu(self.nu_surface_2(surface_nu))
        surface_nu = surface_nu.view(b, t, -1)
        #surface_nu = torch.mul(surface_nu, 100) # optional scaling
        x = x.add(surface_nu)
        x = F.relu(self.nu_surface_f1(x))

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TA_SURFACE_N1(nn.Module): # add surface by 1
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_SURFACE_N1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim, 1, 3, padding=1)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        #surface = torch.add(surface, 1) # add surface by 1 # obsolete
        m = torch.nn.Threshold(0, 0) # take positive
        surface = m(surface)
        surface = torch.mul(surface, pow(2, 0.5)) # multiply by 2^0.5
        
        a = torch.cat((a, surface), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50TA_SURFACE_N2(nn.Module): # take abs and expand size
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA_SURFACE_N2, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.surface_dim = 18  # surface feature dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim + self.surface_dim*2, 1, 3, padding=1)

    def forward(self, x, surface):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)

        # append surface feature
        surface = surface.view(b, t, self.surface_dim)
        surface_n = torch.mul(surface, -1) # multiply by -1
        surface_c = torch.cat((surface, surface_n), 2)
        m = torch.nn.Threshold(0, 0) # take positive
        surface_c = m(surface_c)
        a = torch.cat((a, surface_c), 2)

        a = a.permute(0, 2, 1)

        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)

        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax'  # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048  # feature dimension
        self.middle_dim = 256  # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        # self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [5,3]) # 5,3 cooresponds to 150, 75 input image size
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 5])  # 3,5 cooresponds to 75, 150 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7, 7])  # 7,7 cooresponds to 224, 224 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        # print(x)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)

        f = att_x.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50r, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

# JR
class myResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(myResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        #state_dict=torch.load('/hd/Jiarui/pytorch-multi-label-classifier-master/aic/trainer_aic0225/Train/epoch_25_snapshot.pth')
        state_dict=torch.load('/media/twhuang/NewVolume1/0219AIC_attentionReID/Video-Person-ReID-master2/epoch_25_snapshot.pth')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'base' in k:
                name = k[10:] # remove `module.
                new_state_dict[name] = v
            else:              
                pass
        resnet50.load_state_dict(new_state_dict)

        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        print(self.base)
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        #self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,7]) # 7,4 cooresponds to 150, 75 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        #import pdb; pdb.set_trace()
        #print(1,x.shape)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        #print(2,x.shape)
        x = self.base(x)
       # print(3,x.shape)
        a = F.relu(self.attention_conv(x))
        #print(4,a.shape)
        #import pdb; pdb.set_trace()
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        #print(5,a.shape)
        a = F.relu(self.attention_tconv(a))
        #print(6,a.shape)
        a = a.view(b, t)
        #print(7,a.shape)
        x = F.avg_pool2d(x, x.size()[2:])
        #print(8,x.shape)
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
            #print(9,a.shape)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
            #print(10,a.shape)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        #print(11,x.shape)
        a = torch.unsqueeze(a, -1)
        #print(12,a.shape)
        a = a.expand(b, t, self.feat_dim)
        #print(13,a.shape)
        att_x = torch.mul(x,a)
        #print(14,att_x.shape)
        att_x = torch.sum(att_x,1)
        #print(15,att_x.shape)
        
        f = att_x.view(b,self.feat_dim)
        #print(16,f.shape)
        if not self.training:
            return f
        y = self.classifier(f)
        #print(17,y.shape)
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))