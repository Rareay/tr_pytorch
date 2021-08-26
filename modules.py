
import torch.nn as nn
import torch.nn.functional as F



#class tlcMode(nn.Module):
#    def __init__(self):
#        super(tlcMode, self).__init__()
#        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
#        self.bn1 = nn.BatchNorm2d(32)
#        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
#        self.bn2 = nn.BatchNorm2d(64)
#        self.pool = nn.AvgPool2d(8)
#        self.conv3 = nn.Conv2d(64, 4, 1, bias=False)
#
#    def forward(self, x):
#        x = F.relu(self.bn1(self.conv1(x)))
#        x = F.relu(self.bn2(self.conv2(x)))
#        x = self.pool(x)
#        x = self.conv3(x)
#        x = x.view(-1, 4)
#        return F.softmax(x, dim=1)

class tlcMode(nn.Module):
    def __init__(self):
        super(tlcMode, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 3, 1, 1, 0, bias=False)
        self.fc = nn.Linear(3*4*4, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.maxpool(x)
        x = self.conv3(x)
        x = x.view(-1, 3 * 4 * 4)
        x = F.softmax(self.fc(x), dim=1)
        return x



class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)
        # softmax 1 * 1 * 10

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out

class Maogou(nn.Module):
    def __init__(self):
        super(Maogou, self).__init__()
        # input 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x





