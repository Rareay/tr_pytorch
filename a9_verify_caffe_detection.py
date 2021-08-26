import caffe
import numpy as np
import time
import argparse
import torchvision.transforms as transforms
from module_resnet import ModeResnet18

import os
import cv2
import sys


Biases = [10,17,  12,23,  16,27,  18,34,  23,39,  26,49,  33,59,  42,73,  55,100]
mask1 = [6, 7, 8]
mask2 = [3, 4, 5]
mask3 = [0, 1, 2]

def read_imagelist(image_path):
    name_list = []
    if os.path.isfile(image_path):
        name_list.append(image_path)
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                full_path = os.path.join(root, file)
                name_list.append(full_path)
    return name_list


def load_image(path, w, h):
    transform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = cv2.imread(path) ## BRG
    if img is None:
        print("image is empty!")
        return None
    img = cv2.resize(img, (h, w))
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


def doCount(counts):
    for index, count in enumerate(counts):
        dia = np.diagonal(count)      # 取对角线元素
        sum_h = np.sum(count, axis=0) # 纵向求和
        sum_v = np.sum(count, axis=1) # 横向求和
        accurate = sum_h
        for i, s in enumerate(sum_h):
            accurate[i] = dia[i] / s if s != 0 else 0
        recall = sum_v
        for i, s in enumerate(sum_v):
            recall[i] = dia[i] / s if s != 0 else 0
        print("")
        print(count)
        print("")
        print("[label %d] sum_h   :" % (index), end=' ')
        print(sum_h)
        print("[label %d] sum_v   :" % (index), end=' ')
        print(sum_v)
        print("[label %d] accurate:" % (index), end=' ')
        print(accurate)
        print("[label %d] recall  :" % (index), end=' ')
        print(recall)



def predictDetection(net, namelist, input_blob, output_blob, input_size):
    print(input_size)
    for name in namelist:
        image = load_image(name, input_size[1], input_size[2])
        if image is None:
            continue
        t, outputs = caffe_forward(net, image, input_blob)
        #print(len(outputs))
        for i in range(len(output_blob)):
            out_name = output_blob[i]
            output = outputs[out_name].data
            heigh = len(output[0][0])
            width = len(output[0][0][0])
            for box_index in range(3):
                for h in range(heigh):
                    for w in range(width):
                        deep = box_index * 6
                        if output[0][deep + 4][h][w] * output[0][deep + 5][h][w]> 0.2:
                            print("(%d,%d,%d,%d) [" %(i, box_index, h, w), end=" ")
                            for d in range(6):
                                print("%.2f" % (output[0][deep + d][h][w]), end=" ")
                            print("]")


def predictClass(net, namelist, input_blob, output_blob, input_size):
    for name in namelist:
        image = load_image(name[0], input_size[1], input_size[2])
        if image is None:
            continue
        t, outputs = caffe_forward(net, image, input_blob)
        print("[ %.1f%c  %.3fms ]" % (total * 100 / len(namelist), '%', t), end=" ")
        for i in range(len(output_blob)):
            print("[", end=" ")
            out_name = output_blob[i]
            output = outputs[out_name].data
            for j in range(len(output[0])):
                print("%.4f" %(output[0][j]), end=" ")
            output_list = output[0].tolist()
            index = output_list.index(max(output_list))
            print("] ", end=" ")
            if name[1] >= len(counts[i]):
                continue
            counts[i][name[1]][index] += 1
        total += 1
        print("%s" %(name[0]))
    #doCount(counts)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test caffemode')
    parser.add_argument('--protofile',  default='caffemodel/quan.prototxt', type=str)
    parser.add_argument('--weightfile', default='caffemodel/quan.caffemodel', type=str)
    parser.add_argument('--input_blob',  default='data',    type=str)
    #parser.add_argument('--input_blob',  default='blob1',    type=str)
    parser.add_argument('--output_blob', default=['prob'], type=list)
    #parser.add_argument('--output_blob', default=['layer54-conv', 'layer61-conv', 'layer69-conv'], type=list)
    parser.add_argument('--imgfile', default='./data/tlc/color/test/e', type=str)
    parser.add_argument('--input_size', default=[3, 32, 32], type=list)
    parser.add_argument('--meanB', default=102, type=float)
    parser.add_argument('--meanG', default=102, type=float)
    parser.add_argument('--meanR', default=102, type=float)
    parser.add_argument('--scale', default=255, type=float)

    args = parser.parse_args()

    protofile = args.protofile
    weightfile = args.weightfile
    input_blob = args.input_blob
    input_size = args.input_size
    output_blob = args.output_blob
    imgfile = args.imgfile
    

    namelist = []
    #namelist = read_imagelist(imgfile)
    namelist += read_imagelist("data/tlc/color/test/e", 0)
    namelist += read_imagelist("data/tlc/color/test/g", 1)
    namelist += read_imagelist("data/tlc/color/test/r", 2)
    namelist += read_imagelist("data/tlc/color/test/y", 3)


    net = load_caffemod(protofile, weightfile, input_blob, input_size)

    #predictDetection(net, namelist, input_blob, output_blob, input_size)
    predictClass(net, namelist, input_blob, output_blob, input_size)


