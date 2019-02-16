# -*- coding:utf8 -*-

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from data import MyDataset
from lossF import MyLoss, count_acc
from net import Scale12, Scale3
from param_utils import BATCH_SIZE, LR, EPOCHS, USE_GPU, NUM_WORKERS, MOMENTUM, GAMMA, STEP_SIZE, DELTA, net3_H, \
    net3_W



from tensorboardX import SummaryWriter


def init_weight(modules_dict, init_method='kaiming-normal'):
    for m in modules_dict:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                init.uniform(m.bias)
            if init_method == 'kaiming_normal':
                init.kaiming_normal(m.weight)
            elif init_method == 'xaiver_uniform':
                init.xavier_uniform(m.weight)
            elif init_method == 'xaiver_normal':
                init.xavier_normal(m.weight)
            else:
                raise NotImplementedError

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def train_model(model, criterion, optimizer, scheduler, num_epochs, delta):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    writer = SummaryWriter('runsNet3')

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            if phase == 'val':
                continue
                # model.train(False)
            print('current phase:' + phase)
            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(data_loaders[phase]):
                inputs, labels = data
                print("getting inputs and labels : %d" % i)
                # if i < 91:
                #    continue

                if gpu_status:
                    inputs = Variable(inputs.cuda())
                    # print(type(inputs))
                    # print(inputs.shape)
                    labels = Variable(labels.cuda())
                    # print(type(labels))
                    # print(labels.shape)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                # optimizer.zero_grad()

                outputs = model(inputs)
                # x = vutils.make_grid(outputs.cpu())
                # print(outputs.max())

                outputs_cpu = outputs.cpu().detach().numpy()
                # writer.add_image('output image', x, epoch)

                # print(outputs.shape)
                loss = criterion(outputs, labels)
                # print(loss)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss
                running_corrects += count_acc(outputs, labels, delta)
                # print(running_corrects)

                x = outputs
                del inputs, labels

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / (dataset_sizes[phase] * net3_H * net3_W)
            writer.add_scalar('Train_loss', epoch_loss, epoch)
            writer.add_scalar('Train_acc', epoch_acc, epoch)

            print('{} Loss : {:.4f} ACC : {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    writer.close()
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val ACC: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    path = os.getcwd()
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

    net12 = Scale12(BATCH_SIZE)
    # net12仅做预测不训练
    net12_model_wts = path + "/checkpoints/net12.pth"
    net12.load_state_dict(torch.load(net12_model_wts))
    net12.eval()

    model = Scale3(net12)
    init_weight(model.state_dict())

    if USE_GPU:
        gpu_status = torch.cuda.is_available()
    else:
        gpu_status = False

    if gpu_status:
        model = model.cuda()
        print("using gpu")
    else:
        print("using cpu")

    criterion = MyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)

    model_final = train_model(model=model,
                              criterion=criterion,
                              optimizer=optimizer_ft,
                              scheduler=exp_lr_scheduler,
                              num_epochs=EPOCHS,
                              delta=DELTA)

    torch.save(model_final.state_dict(), path + "/checkpoints/net3.pth")
