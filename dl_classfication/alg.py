import argparse
from ast import parse
from email.utils import localtime
from fileinput import filename
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from a3_getImageList import getClass
from getdataset import getDataset
import os
import time
import numpy as np


class ModelClassfication():
    __use_cuda = None
    __device   = None
    use_type = "test" # train  test  continue 三种模式
    input_w = 512     # 模型输入尺寸
    input_h = 512     # 模型输入尺寸
    batch_size = 8    # 
    epoch = 0         #
    model = None      # 模型
    optimizer = None  # 优化器
    lr = 0.005        # 初始学习率
    class_name = None # 类别名称
    model_name = None # 模型名称
    model_save_path = "Models" # 模型保存的文件夹名称
    train_loader = None # 训练数据加载器
    val_loader  = None  # 测试数据加载器
    save_each_epoch = 0 # 多少个epoch保存一次模型

    def __init__(self) -> None:
        self.__use_cuda = torch.cuda.is_available()
        self.__device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.model_save_path):
            os.system('mkdir %s' % (self.model_save_path))
        np.set_printoptions(precision=4) # numpy打印保留4位小数

    def load_data(self, train_txt=None, val_txt=None):
        if train_txt != None: # 获取训练数据加载器
            train_data = getDataset(txt=train_txt, w=self.input_w, h=self.input_h)
            self.train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        if val_txt != None:   # 获取测试数据加载器
            val_data = getDataset(txt=val_txt, w=self.input_w, h=self.input_h)
            self.val_loader = DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=False, num_workers=2)
        if train_txt == None and val_txt == None:
            print("Please intput train dataset or valide dataset!")
            exit(0)

    def load_model(self, model_name=None):
        self.model_name = model_name
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=True) # 保持训练过的参数
        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=True) # 保持训练过的参数
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=True) # 保持训练过的参数
        elif model_name == "resnet101":
            self.model = models.resnet101(pretrained=True) # 保持训练过的参数
        elif model_name == "resnet152":
            self.model = models.resnet152(pretrained=True) # 保持训练过的参数
        else:
            print("Please input mode name form:")
            print("  resnet18 / resnet34 / resnet50 / resnet101 / resnet152")
            exit(0)
        
        if self.use_type == "test":
            for param in self.model.parameters():
                param.requires_grad = False # True：可以更新梯度 False：不可更新梯度
        

    def change_class_number(self, num): # 改变模型的类别数量，即要分类的个数
        if self.model == None:
            print("Please input mode name!")
            exit(0)
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num),
            nn.LogSoftmax(dim=1)
        )
    
    def load_optimizer(self): # 加载优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def save_dict(self, filename): # 保存模型和其他参数，同时保存模型的onnx
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename + ".pth")
        print("Save Path:" + filename + ".pth")
        input = torch.randn(1, 3, self.input_h, self.input_w).to(self.__device) # 设置输入尺寸
        torch.onnx.export(self.model, input, filename + ".onnx", verbose=False)
        #print("Save Path:" + filename + ".onnx")
    
    def load_dict(self, filename): # 加载模型和其他参数
        if None == filename:
            print("The dict file not exists!")
            exit(0)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
    
    def train(self, epoch_max):
        if self.__use_cuda:
            self.model = self.model.cuda(0)
            if self.use_type == "continue": # 把optimivzer加载到gpu
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda(0)
        criterion = nn.CrossEntropyLoss()
        batch_num = 0
        self.model.train()
        cur_loss = 0.
        cur_acc  = 0.
        while self.epoch < epoch_max:
            self.epoch += 1
            train_loss = 0.
            train_acc  = 0.
            total_img  = 0.
            tab = np.zeros([len(self.class_name), len(self.class_name)])
            tab_diag = np.diag_indices(len(self.class_name))
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                total_img += labels.size()[0]
                if self.__use_cuda:
                    inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # 统计各类别正确数量和总数量
                p = predicted.tolist()
                l = labels.tolist()
                for j in range(len(p)):  # 表头：  实际值\测量值
                    tab[l[j], p[j]] += 1

                # 打印
                p_num = 10
                if i != 0 and i % p_num == 0:
                    sum_0 = tab.sum(axis=0) # 每一列求和
                    sum_0[sum_0 == 0] = 1
                    sum_1 = tab.sum(axis=1) # 每一行求和
                    sum_1[sum_1 == 0] = 1
                    diag = tab[tab_diag]    # 对角线
                    precision = diag / sum_0          # 精确率
                    recall = diag / sum_1             # 召回率
                    accuracy = diag.sum() / tab.sum() # 准确率
                    print('Epoch: %d  Batch: %d,  Iamges: %d,  Loss: %.4f,  Tacc: %.4f'
                            % (self.epoch,
                               batch_num,
                               total_img,
                               train_loss / i,
                               accuracy))
                    print("             Precision recall", end="\n")
                    for jj in range(len(self.class_name)):
                        print('%12s %.4f    %.4f' % (self.class_name[jj], precision[jj], recall[jj]))
                    print("")
                    cur_acc = accuracy

                batch_num += 1
                if i != 0:
                    cur_loss = train_loss / i

            if self.save_each_epoch != 0 and self.epoch % self.save_each_epoch == 0:
                log_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                log_acc  = "%.4f-%.4f" % (cur_loss, cur_acc)
                name = "%s-h%d-w%d-c%d-%s-%s" % (self.model_name, len(self.class_name), \
                                                 self.input_h, self.input_w, log_time, log_acc)
                save_path = os.path.join(self.model_save_path, name)
                self.save_dict(save_path)
        print("Finished Training.")

    def test(self):
        if self.__use_cuda:
            self.model = self.model.cuda(0)
        correct = 0
        total = 0
        flag = 0
        tab = np.zeros([len(self.class_name), len(self.class_name)])
        tab_diag = np.diag_indices(len(self.class_name))
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                if self.__use_cuda:
                    inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if (total - flag) > 500:
                    flag = total
                    print("Have been test %d images..." % (total))
                p = predicted.tolist()
                l = labels.tolist()
                for j in range(len(p)):  # 表头：  实际值\测量值
                    tab[l[j], p[j]] += 1
        print("Have been test %d images... Done." % (total))

        sum_0 = tab.sum(axis=0) # 每一列求和
        sum_0[sum_0 == 0] = 1
        sum_1 = tab.sum(axis=1) # 每一行求和
        sum_1[sum_1 == 0] = 1
        diag = tab[tab_diag]    # 对角线
        # 精确率
        precision = diag / sum_0
        # 召回率
        recall = diag / sum_1
        # 准确率
        accuracy = diag.sum() / tab.sum()
        print('Test Total Acc: %.4f' % (accuracy))
        print("             Precision recall", end="\n")
        for jj in range(len(self.class_name)):
            print('%12s %.4f    %.4f' % (self.class_name[jj], precision[jj], recall[jj]))

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default=None)
    parser.add_argument('-t', '--type', default=None)
    parser.add_argument('-c', '--class_num', default=2)
    parser.add_argument('-W', '--width', default=None)
    parser.add_argument('-H', '--height', default=None)
    parser.add_argument('-b', '--batch_size', default=8)
    parser.add_argument('-r', '--lr', default=0.005)
    parser.add_argument('-e', '--epoch_max', default=100)
    parser.add_argument('-s', '--save_each_epoch', default=1)

    parser.add_argument('-d', '--dict', default=None)

    parser.add_argument('-tp', '--train_txt', default="./imagelist/train.txt")
    parser.add_argument('-vp', '--val_txt',   default="./imagelist/val.txt")

    args = parser.parse_args()
    if None == args.model_name or None == args.type or None == args.width \
        or None == args.height:
        print("Usage:")
        print("  python %s -n resnet34 -t test -c 2 -W 512 -H 512 -b 16 -r 0.005" % (sys.argv[0]))
        print("  -m --mode_name   resnet18/resnet34/resnet50/resnet101/resnet152")
        print("  -t --type        test/train/continue")
        print("  -c --class_num   2")
        print("  -W --width       512")
        print("  -H --height      512")
        print("  -b --batch_size  16")
        print("  -r --lr          0.005")
        print("  -e --epoch_max   100")
        print("  -s --save_each_epoch 1")
        print("  -d --dict        xxx.pth")
        exit(0)

    classfication = ModelClassfication()
    classfication.use_type   = args.type
    classfication.input_w    = int(args.width)
    classfication.input_h    = int(args.height)
    classfication.batch_size = int(args.batch_size)
    classfication.save_each_epoch = int(args.save_each_epoch)
    classfication.class_name = getClass()

    # 记录脚本的参数，保存在文件中

    if 'train' == args.type:
        classfication.lr = float(args.lr)
        classfication.load_data(train_txt=args.train_txt)
        classfication.load_model(model_name=args.model_name)
        classfication.change_class_number(num=int(args.class_num))
        classfication.load_optimizer()
        if args.dict != None: # （可选）指定模型文件
            classfication.load_dict(args.dict)
            classfication.load_optimizer()
        classfication.train(epoch_max=int(args.epoch_max))
    elif 'test' == args.type:
        classfication.load_data(val_txt=args.val_txt)
        classfication.load_model(model_name=args.model_name)
        classfication.change_class_number(num=int(args.class_num))
        classfication.load_optimizer()
        classfication.load_dict(args.dict)
        classfication.test()
    elif 'continue' == args.type:
        classfication.load_data(train_txt=args.train_txt)
        classfication.load_model(model_name=args.model_name)
        classfication.change_class_number(num=int(args.class_num))
        classfication.load_optimizer()
        classfication.load_dict(args.dict)
        classfication.train(epoch_max=int(args.epoch_max))
        
