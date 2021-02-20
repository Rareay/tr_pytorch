import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np


class block_Mish(nn.Module):
    def __init__(self):
        super(block_Mish, self).__init__()

    def forward(self, x):
        return x + torch.tanh(F.softplus(x))


class block_CBM(nn.Module):
    def __init__(self, in_channal, out_channal, kernel, stride=1):
        super(block_CBM, self).__init__()
        self.conv = nn.Conv2d(in_channal, out_channal, kernel, stride, kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channal)
        self.activate = block_Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class block_Res(nn.Module):
    def __init__(self, channels, hidden_channnels=None):
        super(block_Res, self).__init__()
        if hidden_channnels is None:
            dden_channnels = channels

        self.block = nn.Sequential(
            block_CBM(channels, hidden_channnels, 1),
            block_CBM(hidden_channnels, channels, 3),
        )

    def forward(self, x):
        return x + self.block(x)



class block_Resbody(nn.Module):
    def __init__(self, in_channel, out_channel, num_res, first):
        super(block_Resbody, self).__init__()
        self.downsample_conv = block_CBM(in_channel, out_channel, 3, stride=2)
        if first:
            self.split_conv0 = block_CBM(out_channel, out_channel, 1)
            self.split_conv1 = block_CBM(out_channel, out_channel, 1)
            self.blocks_conv = nn.Sequential(
                block_Res(channels=out_channel, hidden_channels=out_channel//2),
                block_CBM(out_channel, out_channel, 1)
            )
            self.concat_conv = block_CBM(out_channel * 2, out_channel, 1)
        else:
            self.split_conv0 = block_CBM(out_channel, out_channel//2, 1)
            self.split_conv1 = block_CBM(out_channel, out_channel//2, 1)
            self.blocks_conv = nn.Sequential(
                *[block_CBM(channels=out_channel//2) for _ in range(num_res)],
                block_CBM(out_channel//2, out_channel//2 ,1)
            )
            self.concat_conv = block_CBM(out_channel, out_channel, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = block_CBM(3, self.inplanes, kernel=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            block_Resbody(self.inplanes, self.feature_channels[0], layers[0], first=True),
            block_Resbody(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            block_Resbody(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            block_Resbody(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            block_Resbody(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5



def darknet53(pretrained):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    #if pretrained:
    #    load_model_pth(model, pretrained)
    return model


if __name__ == '__main__':
    backbone = darknet53()

