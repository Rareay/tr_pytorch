import sys

sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
import torchvision
from modules import *
import pytorch_to_caffe
from module_resnet import ModeResnet18, ModeResnet10
#from jiang.nb_pytorch_2_caffe import NBNet
from jiang.NBNet import NBNet
from jiang.utils.training_util import load_checkpoint
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

if __name__ == '__main__':
    name = 'nb'
    #Module = ModeResnet10()
    #Module = tlcMode()
    Module = NBNet().cpu()
    #Module.load_state_dict(torch.load("Module5.pth"))
    checkpoint = torch.load("jiang/model_best.pth.tar", map_location='cpu')
    Module.load_state_dict(checkpoint['state_dict'])
    #checkpoint = load_checkpoint("/home/tanrui/tanrui/trpytorch/jiang", False, 'best')
    #Module.load_state_dict(checkpoint['state_dict'])
    Module.eval()
    #input = torch.ones([1, 3, 32, 32])
    #input = torch.ones([1, 3, 128, 128])
    input=Variable(torch.ones([1,3,128,128]))
    pytorch_to_caffe.trans_net(Module, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
