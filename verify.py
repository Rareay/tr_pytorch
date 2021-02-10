# 2018.09.06 by Shining 
import sys
sys.path.insert(0,'/home/shining/Projects/github-projects/caffe-project/caffe/python')
import caffe
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.models import resnet
import time
from module_resnet import ModeResnet18

import os
import cv2

#caffe load formate
def load_image_caffe(imgfile):
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def load_image_pytorch(image_path):
    name_list = []
    images = torch.tensor([])
    # transforms.ToTensor()
    transform1 = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (args.height, args.width))
        img = transform1(img)  # 归一化到 [0.0,1.0]
        img = img.view(-1, 3, args.height, args.width)
        images = torch.cat((images, img), 0)
        name_list.append(image_path)
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                full_path = os.path.join(image_path, file)
                img = cv2.imread(full_path)
                if img is None:
                    continue
                img = cv2.resize(img, (args.height, args.width))
                img = transform1(img)  # 归一化到 [0.0,1.0]
                img = img.view(-1, 3, args.height, args.width)
                images = torch.cat((images, img), 0)
                name_list.append(file)

    return name_list, images



def forward_pytorch(net, image):
    if args.cuda:
        net.cuda()
    #print(net)
    net.eval()
    #image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    t0 = time.time()
    blobs = net.forward(image)
    #print(blobs.data.numpy().flatten())
    t1 = time.time()
    return t1-t0, blobs, net.parameters()

# Reference from:
def forward_caffe(protofile, weightfile, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['blob1'].reshape(image.size()[0],
                               image.size()[1],
                               image.size()[2],
                               image.size()[3])
    net.blobs['blob1'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='resnet18.prototxt', type=str)
    parser.add_argument('--weightfile', default='resnet18.caffemodel', type=str)
    parser.add_argument('--model', default="Module20.pth", type=str)
    parser.add_argument('--imgfile', default='./data/temp', type=str)
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()

    protofile = args.protofile
    weightfile = args.weightfile
    imgfile = args.imgfile

    net = ModeResnet18()
    net.load_state_dict(torch.load("Module20.pth"))
    net.eval()

    namelist, image = load_image_pytorch(imgfile)

    time_pytorch, pytorch_blobs, pytorch_models = forward_pytorch(net, image)
    time_caffe, caffe_blobs, caffe_params = forward_caffe(protofile, weightfile, image)

    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)
    
    print('------------ Output Difference ------------')
    blob_name = "softmax_blob1"
    if args.cuda:
        pytorch_data = pytorch_blobs.data.cpu().numpy()
    else:
        pytorch_data = pytorch_blobs.data.numpy()
    caffe_data = caffe_blobs[blob_name].data

    for i in range(len(caffe_data)):
        print("[%.4f %.4f %.4f] %s" % (caffe_data[i][0],
                                       caffe_data[i][1],
                                       caffe_data[i][2],
                                       namelist[i]))
    diff = abs(pytorch_data - caffe_data).sum()
    print('\n %-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' % (blob_name, pytorch_data.shape, caffe_data.shape, diff/pytorch_data.size))
