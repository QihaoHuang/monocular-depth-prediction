import torch
import torch.nn as nn
from torch.autograd import Variable
from graphviz import Digraph
from visualize import  make_dot

class Scale12(nn.Module):
    def __init__(self):
        super(Scale12, self).__init__()
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_6 = nn.Sequential(
            nn.Linear(512 * 9 * 7, 4096),
            nn.ReLU(),
            nn.Dropout()
        )

        self.layer1_7 = nn.Sequential(
            nn.Linear(4096, 64 * 19 * 14),
            nn.ReLU(),
            nn.Dropout()
        )

        self.layer2_1 = nn.Sequential(
            # TODO padding=1????
            nn.Conv2d(3, 96, kernel_size=9, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2_2 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.layer2_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.layer2_4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        # x = Nx3x304x228
        out1 = self.layer1_1(x)
        # print(out1.size())
        # out1 = Nx64x150x112
        out1 = self.layer1_2(out1)
        # print(out1.size())
        # out1 = Nx128x75x56
        out1 = self.layer1_3(out1)
        # print(out1.size())
        # out1 = Nx256x37x28
        out1 = self.layer1_4(out1)
        # print(out1.size())
        # out1 = Nx512x18x14
        out1 = self.layer1_5(out1)
        # print(out1.size())
        # out1 = Nx512x9x7
        _features = out1.view(out1.size(0), -1)
        # _features = Nx32256(512*9*7)
        # print(_features.size())
        out1 = self.layer1_6(_features)
        # out1 = 4096
        out1 = self.layer1_7(out1)
        # out1 = 17024(64*19*14)
        out1 = out1.view(N, 64, 19, 14)
        # reshape Nx17024 to Nx64x19x14
        m1 = nn.Upsample(scale_factor=4, mode='bilinear')
        out1 = m1(out1)
        # print(out1.size())
        # unsample Nx64x19x14 -> Nx64x76x56
        out1 = out1[:, :, :-2, :-1]
        # print(out1.size())
        # crop Nx64x76x56 -> Nx64x74x55
        # out1 = Nx64x74x55
        out2 = self.layer2_1(x)
        # out2 = Nx96x74x55
        # print(out2.size())
        out2 = torch.cat([out2, out1], 1)
        # out2 = Nx160x74x55
        out2 = self.layer2_2(out2)
        # print(out2.size())
        # out2 = Nx64x74x55
        out2 = self.layer2_3(out2)
        # print(out2.size())
        # out2 = Nx64x74x55
        out2 = self.layer2_4(out2)
        # print(out2.size())
        # out2 = Nx1x74x55
        m2 = nn.Upsample(scale_factor=2, mode='bilinear')
        out2 = m2(out2)
        # print(out2.size())
        # unsample Nx1x74x55 -> Nx1x148x110
        out2 = out2[:, :, :-1, :-1]
        # print(out2.size())
        # crop Nx1x148x110 -> Nx1x147x109

        return out2


class Scale3(nn.Module):
    def __init__(self):
        super(Scale3, self).__init__()
        self.layer3_1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
            # pool k=3 -> 96x146x108
            # pool k=2 -> 96x147x109
        )

        self.layer3_2 = nn.Sequential(
            nn.Conv2d(97, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.layer3_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.layer3_4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        # x = Nx3x304x228
        out3 = self.layer3_1(x)
        # print(out3.size())
        # out3 = Nx96x147x109
        # print(net12.forward(x).size())
        out3 = torch.cat([out3, net12.forward(x)], 1)
        # net12.forward(x) = Nx1x147x109
        # out3 = Nx97x147x109
        out3 = self.layer3_2(out3)
        # out3 = Nx64x147x109
        out3 = self.layer3_3(out3)
        # out3 = Nx64x147x109
        out3 = self.layer3_4(out3)
        # out3 = Nx1x147x109
        return out3

N = 16
input = Variable(torch.randn(N, 3, 304, 228))
net12 = Scale12()
# output = net12(input)
net3 = Scale3()
output = net3(input)

o = make_dot(output)
o.view()