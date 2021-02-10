import sys

sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
import torchvision
from modules import *
import pytorch_to_caffe
from module_resnet import ModeResnet18

if __name__ == '__main__':
    name = 'resnet18'
    Module = ModeResnet18()
    Module.load_state_dict(torch.load("Module20.pth"))
    Module.eval()
    input = torch.ones([1, 3, 224, 224])
    pytorch_to_caffe.trans_net(Module, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))