import caffe
import numpy as np
import time
import argparse
import torchvision.transforms as transforms
from module_resnet import ModeResnet18

import os
import cv2
import sys



def read_imagelist(image_path):
    name_list = []
    if os.path.isfile(image_path):
        name_list.append(image_path)
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                full_path = os.path.join(image_path, file)
                name_list.append(full_path)
    return name_list


def load_image(path, w, h):
    transform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = cv2.imread(path) ## BRG
    if img is None:
        print("image is empty!")
        return None
    img = cv2.resize(img, (w, h))
    img = transform(img)  # 归一化到 [0.0,1.0]
    return img


def load_caffemod(protofile, weightfile, input_blob, input_size):
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs[input_blob].reshape(1, input_size[0],
                                  input_size[1],
                                  input_size[2])
    return net


def caffe_forward(net, image, input_blob):
    net.blobs[input_blob].data[...] = image
    t0 = time.time()
    net.forward()
    t1 = time.time()
    return t1-t0, net.blobs


def predict(net, namelist, input_blob, output_blob, input_size):
    for name in namelist:
        image = load_image(name, input_size[1], input_size[2])
        if image is None:
            continue
        t, output = caffe_forward(net, image, input_blob)
        output = output[output_blob].data
        print("use: %.4f ms [" %(t), end=" ")
        for i in range(len(output[0])):
            print("%.4f" %(output[0][i]), end=" ")
        output_list = output[0].tolist()
        index = output_list.index(max(output_list))
        print("] %d %s" %(index, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test caffemode')
    parser.add_argument('--protofile',  default='resnet18.prototxt', type=str)
    parser.add_argument('--weightfile', default='resnet18.caffemodel', type=str)
    parser.add_argument('--input_blob',  default='blob1',    type=str)
    parser.add_argument('--output_blob', default='softmax_blob1', type=str)
    parser.add_argument('--imgfile', default='./data/raw-img/test/pecora', type=str)
    parser.add_argument('--input_size', default=[3, 224, 224], type=list)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)

    args = parser.parse_args()

    protofile = args.protofile
    weightfile = args.weightfile
    input_blob = args.input_blob
    input_size = args.input_size
    output_blob = args.output_blob
    imgfile = args.imgfile

    namelist = read_imagelist(imgfile)

    net = load_caffemod(protofile, weightfile, input_blob, input_size)

    predict(net, namelist, input_blob, output_blob, input_size)


