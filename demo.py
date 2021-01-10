import torch
import torchvision
import torchvision.transforms as transforms

from getDataSet import getDataset, imshowDatesetBatch
from torchvision import transforms
from torch.utils.data import DataLoader
from getImageList import getClass
from modules import *

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils




class Demo():
    __useCuda = None
    __Device = None
    ModuleSavePath = "./Module"
    TrainLoader = None
    TestLoader = None
    Module = None
    Epoch = 0
    EpochSaveModule = 0
    EpochDoTest = 0
    Writer = None
    BatchNum = 0
    Classes = None

    def __init__(self, WriterPath = "./runs/log"):
        self.Writer = SummaryWriter(WriterPath)
        self.__useCuda = torch.cuda.is_available()
        self.__Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def loadData(self, TrainLoader, TestLoader):
        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader

    def loadModule(self, Module):
        self.Module = Module
        if self.__useCuda:
            self.Module = self.Module.cuda()

    def train(self, lr=0.5, momentum=0.9):
        Criterion = nn.CrossEntropyLoss()
        Optimizer = optim.SGD(self.Module.parameters(), lr=lr, momentum=momentum)
        self.Module.train()
        self.BatchNum = 0
        for epoch in range(1, self.Epoch+1):
            train_loss = 0.
            train_acc = 0.
            total_img = 0
            for i, data in enumerate(self.TrainLoader, 0):
                inputs, labels = data
                total_img += labels.size()[0]
                if self.__useCuda:
                    inputs, labels = inputs.to(self.__Device), labels.to(self.__Device)

                outputs = self.Module(inputs)
                loss = Criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()
                train_loss += loss.item()
                acc = torch.sum(predicted == labels)
                train_acc += acc.item()

                p_num = 10
                if i != 0 and i % p_num == 0:
                    print('Epoch: %d  Batch: %d,  Loss: %.4f,  Acc: %.4f'
                            % (epoch,
                               self.BatchNum,
                               train_loss / i,
                               train_acc / total_img))
                    self.Writer.add_scalar('train/loss', train_loss / i, self.BatchNum)
                    self.Writer.add_scalar('train/acc', train_acc / total_img, self.BatchNum)
                self.BatchNum += 1
            if self.EpochSaveModule != 0 and epoch % self.EpochSaveModule == 0:
                torch.save(self.Module.state_dict(), self.ModuleSavePath + str(epoch) + ".pth")
            if self.EpochDoTest != 0 and epoch % self.EpochDoTest == 0:
                self.test()
        self.Writer.close()
        print('Finished Training')

    def test(self):
        correct = 0
        total = 0
        class_correct = list(0. for i in range(len(self.Classes)))
        class_total = list(0. for i in range(len(self.Classes)))
        self.Module.eval()
        with torch.no_grad():
            for data in self.TestLoader:
                inputs, labels = data
                if self.__useCuda:
                    inputs, labels = inputs.to(self.__Device), labels.to(self.__Device)
                outputs = self.Module(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total % 1000 == 0:
                    print("Have been test %d images..." % (total))
                c = (predicted == labels).squeeze()
                for i in range(labels.size()[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        self.Writer.add_scalar('test/loss tatal', correct / total, self.BatchNum)
        print('Test Total Acc: %.4f' % (correct / total))

        for i in range(len(self.Classes)):
            self.Writer.add_scalar('testc ' + self.Classes[i],
                                   class_correct[i] / class_total[i],
                                   self.BatchNum)
            print('Test ', self.Classes[i], ' Acc: %.4f' % (class_correct[i] / class_total[i]))

if __name__ == "__main__":
    demo = Demo()
    demo.Classes = getClass()

    TrainData = getDataset(txt="./imagelist/train.txt")
    TestData = getDataset(txt="./imagelist/test.txt")
    TrainLoader = DataLoader(dataset=TrainData, batch_size=256, shuffle=True, num_workers=2)
    TestLoader = DataLoader(dataset=TestData, batch_size=256, shuffle=False, num_workers=2)
    demo.loadData(TrainLoader, TestLoader)

    Module = torchvision.models.resnet18(pretrained=True)
    for param in Module.parameters():
        param.require_grad = False  # 不改变卷积网络部分的参数
    dim = Module.fc.in_features
    Module.fc = nn.Linear(dim, 10)
    demo.loadModule(Module)

    demo.Epoch = 20
    demo.EpochSaveModule = 10
    demo.EpochDoTest = 5
    demo.train(lr=0.0001)

