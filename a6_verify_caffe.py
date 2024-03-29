import caffe
import numpy as np
import time
import argparse
import torchvision.transforms as transforms
from module_resnet import ModeResnet18

import os
import cv2
import sys



def read_imagelist(image_path, label):
    name_list = []
    if os.path.isfile(image_path):
        name_list.append(image_path)
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for file in files:
                full_path = os.path.join(root, file)
                name_list.append([full_path, label])
    return name_list


def load_image(path, w, h):
    transform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #transforms.Normalize(mean=[104.0/255, 104/255, 104/255], std=[1.0/255, 1.0/255, 1.0/255])
        #transforms.Normalize(mean=[104.0 / 255, 117 / 255, 123 / 255], std=[1.0 / 255, 1.0 / 255, 1.0 / 255])
    ])
    img = cv2.imread(path) ## BRG
    if img is None:
        print("image is empty!")
        return None
    img = cv2.resize(img, (h, w))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
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



def predictClass(net, namelist, input_blob, output_blob, input_size):
    counts = []
    image = load_image(namelist[0][0], input_size[1], input_size[2])
    t, outputs = caffe_forward(net, image, input_blob)
    for i in range(len(output_blob)):
        #count = [0 for _ in range(len(outputs[output_blob[i]].data[0]))]
        #counts.append(np.array(count))
        l = len(outputs[output_blob[i]].data[0])
        counts.append(np.zeros([l, l]))
    total = 0

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
    #counts = np.array(counts)
    #print(counts)
    #print(counts / total)
    doCount(counts)

def changeMaskTOImage(mask):
    deep = len(mask[0]) # 16
    high = len(mask[0][0]) # 512
    weight = len(mask[0][0][0]) # 928
    img = np.zeros([high, weight, 3], np.uint8)
    for h in range(high):
        for w in range(weight):
            max_id = 0
            max_value = 0
            for d in range(deep):
                if mask[0][d][h][w] > max_value:
                    max_id = d
                    max_value = mask[0][d][h][w]
            if max_id > 0:
                img[h, w, :] = [255, 255, 255]
    return img


def predictMask(net, namelist, input_blob, output_blob, input_size):
    if not os.path.exists('output'):
        os.makedirs('output')
    for name in namelist:
        image = load_image(name[0], input_size[1], input_size[2])
        if image is None:
            continue
        save_path = os.path.join('output', os.path.basename(name[0]))
        t, outputs = caffe_forward(net, image, input_blob)
        for i in range(len(output_blob)):
            out_name = output_blob[i]
            output = outputs[out_name].data # 16 * 512 * 928
            img = changeMaskTOImage(output)
            cv2.imwrite(save_path, img)
            print("saved in %s" % (save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test caffemode')
    parser.add_argument('--protofile',  default='caffemodel/tlc-29.prototxt', type=str)
    parser.add_argument('--weightfile', default='caffemodel/tlc-29.caffemodel', type=str)
    parser.add_argument('--input_blob',  default='blob1',    type=str)
    #parser.add_argument('--input_blob',  default='data',    type=str)
    parser.add_argument('--output_blob', default=['softmax_blob1'], type=list)
    #parser.add_argument('--output_blob', default=['prob'], type=list)
    parser.add_argument('--imgfile', default='./data/nmac/Bicycle', type=str)
    parser.add_argument('--imgclass', default=0, type=int)
    parser.add_argument('--input_size', default=[3, 32, 32], type=list)
    #parser.add_argument('--input_size', default=[3, 928, 512], type=list)
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
    imgclass = args.imgclass
    

    #namelist = read_imagelist(imgfile, imgclass)
    namelist = []
    namelist += read_imagelist("data/tlc/tlc-test/e", 0)
    namelist += read_imagelist("data/tlc/tlc-test/r", 1)
    namelist += read_imagelist("data/tlc/tlc-test/g", 2)
    namelist += read_imagelist("data/tlc/tlc-test/y", 3)
    #namelist += read_imagelist("data/nmac/Bicycle", 1)
    #namelist += read_imagelist("data/nmac/Tricycle2", 2)
    #namelist += read_imagelist(imgfile, 0)

    net = load_caffemod(protofile, weightfile, input_blob, input_size)

    predictClass(net, namelist, input_blob, output_blob, input_size)
    #predictMask(net, namelist, input_blob, output_blob, input_size)


