import torch.nn as nn
import torch.nn.functional as F
import torch


# CBR
class block_CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(block_CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


# CB
class block_CB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(block_CB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Res_one
class block_Res_one(nn.Module):
    def __init__(self, in_channels, out_channels, first=None):
        super(block_Res_one, self).__init__()

        self.res = nn.Sequential(
            block_CBR(in_channels, out_channels, 3),
            block_CB(out_channels, out_channels, 3),
        )

    def forward(self, x):
        return F.relu(x + self.res(x))


# Res_two
class block_Res_two(nn.Module):
    def __init__(self, in_channels, out_channels, first=None):
        super(block_Res_two, self).__init__()

        if first == None:
            self.res = nn.Sequential(
                block_CBR(in_channels, out_channels, 3, 2),
                block_CB(out_channels, out_channels, 3),
            )
            self.CB = block_CB(in_channels, out_channels, 1, 2)
        else:
            self.res = nn.Sequential(
                block_CBR(in_channels, out_channels, 3),
                block_CB(out_channels, out_channels, 3),
            )
            self.CB = block_CB(in_channels, out_channels, 1)

    def forward(self, x):
        return F.relu(self.CB(x) + self.res(x))




class ModeResnet18(nn.Module):
    def __init__(self):
        super(ModeResnet18, self).__init__()
        ## input 3 x 224 x 224
        self.CBR1 = block_CBR(3, 64, 7, 2)
        self.POOL1 = nn.MaxPool2d(3, 2)
        self.RES = nn.Sequential(
            block_Res_two(64, 64, first=True),
            block_Res_one(64, 64),  # 64 6 x 56
            block_Res_two(64, 128),
            block_Res_one(128, 128),  # 128 x 28 x 28
            block_Res_two(128, 256),
            block_Res_one(256, 256),  # 256 x 14 x 14
            block_Res_two(256, 512),
            block_Res_one(512, 512),  # 512 x 7 x 7
        )
        self.POOL2 = nn.MaxPool2d(7, 1)
        self.fc = nn.Linear(512 * 1 * 1, 3)


    def forward(self, x):
        x = self.CBR1(x)  # 64  x 112 x 112
        x = self.POOL1(x) # 64  x 56  x 56
        x = self.RES(x)   # 512 x 7   x 7
        x = self.POOL2(x) # 512 x 7   x 7
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1) # 3
        return x




if __name__ == "__main__":
    #a = torch.tensor([[[1,2],[2,3]],[[5,6],[7,8]]])
    #b = torch.tensor([[[1,2],[2,3]],[[5,6],[7,8]]])
    #c = torch.stack((a, b, b), 0)
    #print(a.size())
    #print(c.size())
    #a = "asdfdf32as.jpg\n"
    #a = a.strip('\n')
    #a = a.rstrip()
    #a = a.split()
    #print(a[0])
    a = torch.tensor([])
    b = torch.tensor([[[[1,2],[2,3]],[[5,6],[7,8]]]])
    c = torch.tensor([[[[1,2],[2,3]],[[5,6],[7,9]]]])
    a = torch.cat((a, b), 0)
    a = torch.cat((a, c), 0)
    print(b.size())
    print(a.size())
    print(a)
    pass
