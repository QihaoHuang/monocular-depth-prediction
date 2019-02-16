import random
import os
import time
from model import *
from torch.autograd import Variable
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import flow_transforms
import torch
from nyu_dataset_loader import ListDataset

color = np.array([(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255),  # magenta
                  (192, 192, 192),  # silver
                  (128, 128, 128),  # gray
                  (128, 0, 0),  # maroon
                  (128, 128, 0),  # olive
                  (0, 128, 0),  # green
                  (128, 0, 128),  # purple
                  (0, 128, 128),  # teal
                  (65, 105, 225),  # royal blue
                  (255, 250, 205),  # lemon chiffon
                  (255, 20, 147),  # deep pink
                  (218, 112, 214),  # orchid]
                  (135, 206, 250),  # light sky blue
                  (127, 255, 212),  # aqua marine
                  (0, 255, 127),  # spring green
                  (255, 215, 0),  # gold
                  (165, 42, 42),  # brown
                  (148, 0, 211),  # violet
                  (210, 105, 30),  # chocolate
                  (244, 164, 96),  # sandy brown
                  (240, 255, 240),  # honeydew
                  (112, 128, 144), (64, 224, 208), (100, 149,
                                                    237), (30, 144, 255), (221, 160, 221),
                  (205, 133, 63), (255, 240, 245), (255, 255, 240), (255, 165, 0), (255, 160, 122), (205, 92, 92),
                  (240, 248, 255)])


def load_data(type, path):
    input_rgb_images_dir = os.path.join(path, 'input/')
    target_depth_images_dir = os.path.join(path, 'target_depths/')
    target_labels_images_dir = os.path.join(path, 'labels_38/')

    train_on = 1000
    val_on = 100
    test_on = 50

    print('Loading images...')

    NUM_TRAIN = 1000
    NUM_VAL = 300
    NUM_TEST = 149

    listing = random.sample(os.listdir(input_rgb_images_dir), 1449)
    train_listing = listing[:min(NUM_TRAIN, train_on)]
    val_listing = listing[NUM_TRAIN:min(NUM_VAL + NUM_TRAIN, val_on + NUM_TRAIN)]
    test_listing = listing[NUM_VAL +
                           NUM_TRAIN:min(NUM_VAL + NUM_TRAIN + NUM_TEST, test_on + NUM_VAL + NUM_TRAIN)]
    data_dir = (input_rgb_images_dir, target_depth_images_dir,
                target_labels_images_dir)

    input_transform = transforms.Compose(
        [flow_transforms.Scale(228), flow_transforms.ArrayToTensor()])
    target_depth_transform = transforms.Compose(
        [flow_transforms.Scale_Single(228), flow_transforms.ArrayToTensor()])
    target_labels_transform = transforms.Compose(
        [flow_transforms.ArrayToTensor()])

    co_transform = flow_transforms.Compose([
        flow_transforms.RandomCrop((480, 640)),
        flow_transforms.RandomHorizontalFlip()
    ])

    train_dataset = ListDataset(data_dir, train_listing, input_transform, target_depth_transform,
                                target_labels_transform, co_transform)

    val_dataset = ListDataset(data_dir, val_listing, input_transform, target_depth_transform,
                              target_labels_transform)

    test_dataset = ListDataset(data_dir, test_listing, input_transform, target_depth_transform,
                               target_labels_transform)

    print("Loading data...")
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(
        val_dataset, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size, shuffle=True, drop_last=True)

    if type == "train":
        return train_loader
    elif type == "test":
        return test_loader
    else:
        return val_loader


def count_acc(model, x_var, z_var, y, delta):

    with torch.no_grad():
        pred_depth, pred_labels = model(x_var, z_var)
        _, preds = pred_labels.data.cpu().max(1)

        # Save the input RGB image, Ground truth depth map, Ground Truth Coloured Semantic Segmentation Map,
        # Predicted Coloured Semantic Segmentation Map, Predicted Depth Map for one image in the current batch
        plt.ion()
        input_rgb_image = x_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        plt.imsave('input_rgb.png', input_rgb_image)
        plt.imshow(input_rgb_image)

        input_gt_depth_image = z_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        plt.imsave('input_gt_depth.png', input_gt_depth_image)
        plt.imshow(input_gt_depth_image)

        colored_gt_label = color[y[0].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
        plt.imsave('input_gt_label.png', colored_gt_label)

        colored_pred_label = color[preds[0].squeeze().cpu().numpy().astype(int)].astype(np.uint8)
        plt.imsave('pred_label.png', colored_pred_label)

        pred_depth_image = pred_depth[0].data.squeeze().cpu().numpy().astype(np.float)

        max_value = pred_depth_image.max()
        t = pred_depth_image / max_value
        plt.imsave('pred_depth.png', t)
        # plt.imshow(t)

        plt.imsave('pred_depth_gray.png', t, cmap='gray')
        plt.imshow(t)
        plt.show()
        plt.pause(3)

        # Computing pixel-wise accuracy

    #     depth_eps = 1e-7
    #     print(z_var[0].shape)
    #     print(pred_depth.shape)
    #     print(pred_labels.shape)
    #     print(input_gt_depth_image.shape)
    #     print(pred_depth_image.shape)
    #     gd_log = np.log(input_gt_depth_image + depth_eps)
    #     gd_log = gd_log - np.mean(gd_log)
    #
    #     out_log = np.log(pred_depth_image + depth_eps)
    #     out_log = out_log - np.mean(out_log)
    #
    #     error_log = np.abs(gd_log - out_log)
    #
    #     running_corrects = np.sum(error_log < np.log(delta))
    #
    # acc = float(running_corrects) / input_gt_depth_image.shape[2]*input_gt_depth_image.shape[3]
    acc = 0

    return acc


def test_predict_time(model, x_var, z_var, y):
    since = time.time()
    for i in range(50):
        count_acc(model, x_var, z_var, y, delta)
        print(str(i))

    end = time.time()
    print("-------------")
    print("average time: ", (end-since)/50)


if __name__ == "__main__":
    batch_size = 16
    delta = 1.25
    path = os.getcwd()

    print("current working path: ", path)
    # choose test data loader
    type = "test"
    loader = load_data(type, os.path.join(path, "data/nyu_datasets_changed/"))

    dtype = torch.cuda.FloatTensor
    model = Model(ResidualBlock, UpProj_Block, batch_size)
    model.type(dtype)


    model_best_param = os.path.join(path, 'model_best.pth.tar')
    print(model_best_param)
    checkpoint = torch.load(model_best_param)
    print("load checkpoint ok")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("load parameters ok")

    x, z, y = next(iter(loader))
    print("input RGB shape: ", x.shape)
    print("input depth shape: ", z.shape)
    print("input label shape: ", y.shape)




    x_var = Variable(x.type(dtype)).contiguous()
    z_var = Variable(z.type(dtype)).contiguous()
    y_var = Variable(y.type(dtype).long()).contiguous()
    m = nn.LogSoftmax()

    # pred_depth, pred_labels = model(x_var, z_var)
    # y_var = y_var.squeeze()
    # loss_fn = torch.nn.NLLLoss2d().type(dtype)
    # loss = loss_fn(m(pred_labels), y_var)
    # running_loss = loss.data.cpu().numpy()
    # print("loss between predict label and grouth truth label is:", running_loss)

    x_var = Variable(x.type(dtype))
    z_var = Variable(z.type(dtype))

    acc = count_acc(model, x_var, z_var, y, delta)
    print("ok!!!!!!!!!!")
