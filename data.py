# -*- coding:utf8 -*-
import os
import scipy.io as scio
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from lossF import MyLoss, count_acc
from net import Scale12, Scale3
from param_utils import BATCH_SIZE, NUM_WORKERS


def readData(test_ratio=0.1, val_ratio=0.1):
    datapath = os.getcwd()
    image_mat_data = scio.loadmat(os.path.join(datapath, 'data/images_test.mat'))
    depth_mat_data = scio.loadmat(os.path.join(datapath, 'data/depths_test.mat'))
    # label_mat_data = scio.loadmat(os.path.join(datapath, 'data/labels_test.mat'))
    # discard unnecassary information
    raw_images = np.array(image_mat_data['images'])  # shape = 480 * 640 * 3* 1449
    raw_depths = np.array(depth_mat_data['depths'])  # shape = 480 * 640 * 1449
    # raw_labels = np.array(label_mat_data['labels'])# shape = 480 * 640 * 1449
    # =========================divide data into different types============================
    # train and test
    np.random.seed(0)
    indices = np.random.permutation(raw_images.shape[3])  # 1449
    test_numbers = 160  # round (raw_images.shape[3]*test_ratio)
    images_train = raw_images[:, :, :, indices[:-test_numbers]]
    depths_train = raw_depths[:, :, indices[:-test_numbers]]
    images_test = raw_images[:, :, :, indices[-test_numbers:]]
    depths_test = raw_depths[:, :, indices[-test_numbers:]]
    # test and val
    np.random.seed(0)
    new_indices = np.random.permutation(images_train.shape[3])  # 1299
    test_numbers = 160  # round (raw_images.shape[3]*test_ratio)
    images_val = images_train[:, :, :, new_indices[-test_numbers:]]
    depths_val = depths_train[:, :, new_indices[-test_numbers:]]
    images_train = images_train[:, :, :, new_indices[:-test_numbers]]
    depths_train = depths_train[:, :, new_indices[:-test_numbers]]
    # discard the 9 items
    np.random.seed(0)
    new_indices = np.random.permutation(images_train.shape[3])  # 1129
    test_numbers = 9
    images_train = images_train[:, :, :, new_indices[:-test_numbers]]
    depths_train = depths_train[:, :, new_indices[:-test_numbers]]
    # images_train, images_test,depths_train,depths_test = train_test_split(raw_images, raw_depths, test_size=test_ratio+val_ratio,random_state=42)
    # images_test, images_val,depths_test,depths_val = train_test_split(raw_images, raw_depths, test_size=0.5,random_state=42)
    # ===========================save data=======================
    np.save('data/image_train.npy', images_train)
    np.save('data/image_test.npy', images_test)
    np.save('data/image_val.npy', images_val)
    np.save('data/depth_train.npy', depths_train)
    np.save('data/depth_test.npy', depths_test)
    np.save('data/depth_val.npy', depths_val)
    # np.save('data/raw_labels.npy', raw_labels)
    return None


class MyDataset(Dataset):
    def __init__(self, type, image_transform, stage):
        # ====================read data====================
        if type == 'train':
            image_data = np.load('data/image_train.npy')
            depth_data = np.load('data/depth_train.npy')
        elif type == 'test':
            image_data = np.load('data/image_test.npy')
            depth_data = np.load('data/depth_test.npy')
        elif type == 'val':
            image_data = np.load('data/image_val.npy')
            depth_data = np.load('data/depth_val.npy')
        else:
            print('wrong input')

        # ================to tensor_torch================
        self.image = image_data
        self.depth = depth_data

        # 张数
        if self.image.shape[-1] == self.depth.shape[-1]:
            self.length = self.image.shape[-1]

        # print(self.length)

        # 数据变换
        # 480x640 -> 228x304
        self.rescale = Rescale(228)
        # 228x304 -> 220x296
        self.randomCrop = RandomCrop((220, 296))
        # 220x296 -> 55x74

        if stage == "trainNet3":
            self.resize = transforms.Resize((109, 147))
            # Resize((109, 147)) when trainNet3
        elif stage == "trainNet12":
            self.resize = transforms.Resize((55, 74))
            # Resize((55, 74)) when trainNet12

        self.toTensor = ToTensor()
        self.DtoTensor = DToTensor()
        self.transform = image_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # print("index:", index)
        # print("image shape:", self.image[:, :, :, index].shape)
        # 480x640x3
        # print("depth shape:", self.depth[:, :, index].shape)
        # 480x640

        # 1.resclae 480x640x3 -> 228x304x3 -> 2.toTenosr -> 3x304x228
        img = self.transform(self.toTensor(self.rescale(self.image[:, :, :, index])))

        # print("*-" * 20)

        # 1.rescale 480x640-> 228x304 -> 2.randomCrop -> 220x296 ->
        # 3.resize -> 55x74 -> 4. DtoTensor -> 1x74x55
        depths = self.DtoTensor(self.resize(self.randomCrop(self.rescale(self.depth[:, :, index]))))

        return img, depths


class Rescale(object):
    """Rescale the image in a sample to a given size.
       480 * 640 -> 228x304
       depths are changed at the same time 
    """

    def __init__(self, output_size):
        # output_size=228
        self.output_size = output_size
        self.resizeF = transforms.Resize((228, 304))

    def __call__(self, sample):
        # h, w = sample.shape[:2]
        # if h > w:
        #     new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #     new_h, new_w = self.output_size, self.output_size * w / h
        #
        # new_h, new_w = int(new_h), int(new_w)
        # print(sample.shape)
        # print("new_h: ", new_h)
        # print("new_w: ", new_w)
        # sample = transforms.Resize(sample, (new_h, new_w))

        # 将numpy ndarray转换为PIL Image
        t = Image.fromarray(sample)
        # 做resize变换
        sample = self.resizeF(t)
        # 将PIL 转换为 PIL Image
        sample = np.array(sample)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, depths):
        h, w = depths.shape[:2]
        new_h, new_w = self.output_size
        # =================random crop=================
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # if you want to crop at a certain point , just replace top or left with your desired position
        depths = depths[top: top + new_h, left: left + new_w]
        # 将numpy ndarray转换为PIL Image
        depths = Image.fromarray(depths)

        return depths


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(type(sample))

        sample = np.array(sample)
        sample = np.transpose(sample, (2, 1, 0))
        # 将 228x304x3 转换为 3x304x228

        # 添加了转换为float
        t = torch.from_numpy(sample).float()
        # outputs_cpu = t.cpu().detach().numpy()

        return t


class DToTensor(object):
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample = np.array(sample)
        sample = np.transpose(sample, (1, 0))

        x = sample.shape[0]
        y = sample.shape[1]
        sample = sample.reshape(1, x, y)
        # 将 55x74 转换为 1x74x55

        t = torch.from_numpy(sample).float()
        # 添加了转换为float
        return t


if __name__ == "__main__":
    # readData()

    # 测试数据图像变换
    data_transforms = {
        'train': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: MyDataset(x, data_transforms[x], "trainNet3") for x in ['train', 'val']}

    data_loaders = {x: DataLoader(image_datasets[x],
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("train size:", dataset_sizes['train'])
    print("val size:", dataset_sizes['val'])

    inputs, labels = next(iter(data_loaders['train']))

    outputs_cpu = labels.cpu().detach().numpy()

    # print(inputs.shape)
    # print(labels.shape)

    out1 = torchvision.utils.make_grid(inputs)
    inp1 = torch.transpose(out1, 0, 2)
    plt.imshow(inp1 / 255)
    plt.title("image")
    # plt.savefig("1.jpg")
    plt.show()

    out2 = torchvision.utils.make_grid(labels)
    inp2 = torch.transpose(out2, 0, 2)
    plt.imshow(inp2)
    plt.title("depth")
    # plt.savefig("2.jpg")
    plt.show()

    inputs = Variable(inputs)
    labels = Variable(labels)

    # inputs = Variable(inputs.cuda())
    # labels = Variable(labels.cuda())

    # model = Scale12(BATCH_SIZE)
    #
    path = os.getcwd()
    net12 = Scale12(BATCH_SIZE)
    # net12仅做预测不训练
    net12_model_wts = path + "/checkpoints/net12.pth"
    net12.load_state_dict(torch.load(net12_model_wts))
    net12.eval()

    model = Scale3(net12)
    model.eval()

    criterion = MyLoss()

    print("____________")
    outputs = model(inputs)
    print(outputs.shape)

    print("____________")
    loss = criterion(outputs, labels)
    print(loss)
    print(loss / (109 * 147))

    print("____________")
    acc = count_acc(outputs, labels, 1.25) / (109 * 147 * 16)
    print("acc: ", acc)
