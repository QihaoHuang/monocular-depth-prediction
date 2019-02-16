import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from net import Scale12, Scale3
from param_utils import BATCH_SIZE, NUM_WORKERS
from data import MyDataset

if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: MyDataset(x, data_transforms[x], "trainNet12") for x in ['train', 'val']}

    data_loaders = {x: DataLoader(image_datasets[x],
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("train size:", dataset_sizes['train'])
    print("val size:", dataset_sizes['val'])

    inputs, labels = next(iter(data_loaders['train']))
    print(inputs.shape)
    print(labels.shape)

    out1 = torchvision.utils.make_grid(inputs)
    inp1 = torch.transpose(out1, 0, 2)
    plt.imshow(inp1/255)
    plt.title("image")
    # plt.savefig("1.jpg")
    plt.show()

    out2 = torchvision.utils.make_grid(labels)
    inp2 = torch.transpose(out2, 0, 2)
    plt.imshow(inp2/200)
    plt.title("depth")
    # plt.savefig("2.jpg")
    plt.show()

    path = os.getcwd()

    # 加载模型参数
    net12 = Scale12(BATCH_SIZE)
    net12_model_wts = path + "/checkpoints/net12.pth"
    net12.load_state_dict(torch.load(net12_model_wts))
    net12.eval()

    inputs = Variable(inputs)
    outputs = net12(inputs)
    img = torchvision.utils.make_grid(outputs)
    img = torch.transpose(img, 0, 2)
    t = img.cpu().detach().numpy()
    # print(img.max())
    # outputs_cpu = img.cpu().detach().numpy()
    plt.imshow(t)
    plt.title("depth-predict")
    # plt.savefig("depth-predict.jpg")
    plt.show()
    #
    # # # 加载模型参数
    # net3 = Scale3(net12)
    # net3_model_wts = path + "/checkpoints/net3.pth"
    # net3.load_state_dict(torch.load(net3_model_wts))
    # net3.eval()
    #
    # # 预测
    # outputs = net3(inputs)
    #
    # # 结果输出 可视化
    # img = torchvision.utils.make_grid(outputs)
    # img = torch.transpose(img, 0, 2)
    # t = img.cpu().detach().numpy()
    # plt.imshow(t)
    #
    # plt.title("depth-predict")
    # plt.savefig("depth-predict.jpg")
    # plt.show()