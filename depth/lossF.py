# -*- coding:utf8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from net import Scale12
from torch.autograd import Variable


depth_eps = 1e-6

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, estimated_depth, ground_truth):
        if estimated_depth.size() == ground_truth.size():
            # x = torch.log(torch.abs(estimated_depth)) - torch.log(torch.abs(ground_truth))
            x = torch.abs(torch.log(F.relu(estimated_depth) + depth_eps) - torch.log(F.relu(ground_truth) + depth_eps))

            h_x = x.size()[2]
            w_x = x.size()[3]
            img_r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
            img_l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
            img_t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
            img_b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
            #print(img_r.size(), img_l.size(), img_t.size(), img_b.size())
            xgrad = torch.sum(torch.pow(torch.pow((img_r - img_l) * 0.5, 2) + torch.pow((img_t - img_b) * 0.5, 2), 0.5))
            x_shape = self._tensor_size(x)
            xavag = xgrad / x_shape
            loss_result = torch.sum(x * x) / self._tensor_shape(x) - torch.pow(torch.sum(x) / self._tensor_shape(x), 2) / (2*pow(x_shape, 2)) + xavag
        else:
            print("Error: dimension of estimated_depth and ground_truth doesn't match")
            loss_result = 0
        return loss_result

    def _tensor_size(self, t):
        return t.size()[2] * t.size()[3]
    def _tensor_shape(self, t):
        return t.size()[0] * t.size()[1] * t.size()[2] * t.size()[3]

def count_acc(output, label, delta):
    gd_log = torch.log(label + 1)
    gd_log = gd_log - torch.mean(gd_log)

    out_log = torch.log(output + 1)
    out_log = out_log - torch.mean(out_log)

    error_log = torch.abs(gd_log - out_log)

    running_corrects = np.sum(error_log.detach().cpu().numpy() < np.log(delta))

    return running_corrects


if __name__ == "__main__":
    N = 1
    input = Variable(torch.randn(N, 3, 304, 228))
    # 随机生成数据
    # ---------------Scale12------------
    print("-" * 20)
    net12 = Scale12(N)
    net12.eval()
    # print(net12)

    # 测试net2
    out = net12(input)
    # print(out.size())
    print("-" * 20)

    gd = Variable(torch.randn(N, 1, 74, 55) + 10)

    criterion = MyLoss()
    loss = criterion(out + 10, gd + 10)
    print(loss.data[0].numpy())
    print(loss)

    delta = 1.25
    acc = count_acc(out, gd, delta)
    print(acc)

    # net12.zero_grad()
    # loss.backward()

    # ---------------Scale3------------
    # print("-" * 20)
    # net3 = Scale3()
    # net3.eval()
    # print(net3)

    # print("-" * 20)
    # 测试net3
    # output = net3(input)
    # print(output)
