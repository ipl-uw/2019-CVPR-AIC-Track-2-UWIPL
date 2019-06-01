from __future__ import absolute_import

from .ResNet import *

__factory = {
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TA,
    'myresnet50ta': myResNet50TA,
    'resnet50rnn': ResNet50RNN,
    'resnet50tp_ori': ResNet50TP_ORIENTATION,
    'resnet50tp_ori_iou': ResNet50TP_ORIENTATION_IOU,
    'resnet50ta_ori': ResNet50TA_ORIENTATION,
    'resnet50ta_ori_iou': ResNet50TA_ORIENTATION_IOU,
    'resnet50ta_surface': ResNet50TA_SURFACE,
    'resnet50ta_surface_nu': ResNet50TA_SURFACE_NU,
    'resnet50ta_surface_nu4': ResNet50TA_SURFACE_NU4,
    'resnet50ta_surface_nu2': ResNet50TA_SURFACE_NU2,
    'resnet50ta_surface_nu2f1': ResNet50TA_SURFACE_NU2F1,
    'resnet50ta_surface_n1': ResNet50TA_SURFACE_N1,
    'resnet50ta_surface_n2': ResNet50TA_SURFACE_N2,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
