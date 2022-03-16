from torch import is_tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from a3_getImageList import getClass
import cv2
import os

# 输出图像的函数
def imshowDatesetBatch(img):
    npimg = img.numpy()
    npimg = npimg / 2 + 0.5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def default_loader(path):
    return Image.open(path).convert('RGB')

image_transform = transforms.Compose([
    transforms.Resize(512),               # 把图片resize为256*256
    transforms.RandomCrop(512),           # 随机裁剪224*224
    #transforms.RandomHorizontalFlip(),    # 水平翻转
    transforms.ToTensor(),                # 将图像转为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # image=(image-mean)/std
    #transforms.Normalize(mean=[0, 0, 0], std=[1., 1., 1.])   # image=(image-mean)/std
])


class getDataset(Dataset):
    #def __init__(self, txt, transform=image_transform):
    def __init__(self, txt, w, h):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        #self.transform = transform
        self.transform = transforms.Compose([
            transforms.Resize([h, w]),                       # 把图片resize为 h*w
            #transforms.RandomCrop([int(h*0.8), int(w*0.8)]), # 随机裁剪为指定大小
            #transforms.RandomHorizontalFlip(),    # 水平翻转
            transforms.ToTensor(),                # 将图像转为Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # image=(image-mean)/std
        ])

    def __getitem__(self, item):
        #print(self.imgs[item])
        fn, label = self.imgs[item]
        #img = cv2.imread(fn)
        #img = Image.fromarray(img)
        img = Image.open(fn)
        if img.mode is "RGBA" \
                or img.mode is "PNG"\
                or img.mode is "L" \
                or img.mode is "CMYK":
            img = img.convert("RGB")
        #print(img.mode)
        if self.transform:
            img = self.transform(img)
        #print(img.size())
        return img, label

    def __len__(self):
        return len(self.imgs)


def download_CIFAR10(is_train=True):
    if is_train == True:
        t = "train"
    else:
        t = "test"
    if not os.path.exists("download"):
        os.system("mkdir download")
    if not os.path.exists("download/%s" % (t)):
        os.system("mkdir download/%s" % (t))
    dataset = torchvision.datasets.CIFAR10(root="download/CIFAR10",\
        train=is_train, download=True, transform=None)
    labels = np.zeros(len(dataset))
    for i in range(len(dataset)):
        labels[i] = dataset[i][1]
        img_path = 'download/CIFAR10/%s/%d_%d.png' % (t, labels[i], i)
        sample = dataset[i][0]
        sample.save(img_path)




if __name__ == '__main__':
    download_CIFAR10(is_train=True)
    download_CIFAR10(is_train=False)
    exit(0)
    train_data = getDataset(txt="./imagelist/train.txt", w=300, h=200)
    #train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=2)

    # 随机获取训练图片
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # 打印图片标签
    classes = getClass()
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

    # 显示图片
    imshowDatesetBatch(torchvision.utils.make_grid(images))
