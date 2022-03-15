from model import UNet
from dataloader import Cell_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt

import os

#import any other libraries you need below this line
import gc
import time
from optparse import OptionParser
import Preprocess

def main(epoch_n, lr, data_path, patch_path, result_path, batch_size, preprocess):
    # Paramteres

    # learning rate
    # lr = 1e-4
    # number of training epochs
    # epoch_n = 1
    # input image-mask size
    image_size = 572
    # root directory of project
    root_dir = os.getcwd()
    # training batch size
    # batch_size = 1
    # use checkpoint model for training
    load = False
    # use GPU for training
    gpu = True

    data_dir = os.path.join(root_dir, data_path)
    new_data_dir = os.path.join(root_dir, patch_path)
    save_dir = os.path.join(root_dir, result_path)

    if not os.path.exists(data_dir):
        print('data folder not existed')
    if not os.path.exists(new_data_dir):
        if preprocess:
            os.makedirs(new_data_dir)
        else:
            print("patch images folder not existed")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print('preproc: ' + data_dir)
    print('preproc: ' + new_data_dir)

    if preprocess:
        Preprocess.Preprocess(data_dir, new_data_dir)
    # data_dir = "./data/cells/"

    trainset = Cell_data(data_dir=new_data_dir, size=image_size)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = Cell_data(data_dir=new_data_dir, size=image_size, train=False)
    testloader = DataLoader(testset, batch_size=batch_size)

    device = torch.device('cuda:0' if gpu else 'cpu')
    print(device)

    channels = 3
    classes = 3
    # model = UNet(channels, classes).to('cuda:0').to(device)
    model = UNet(channels, classes).to(device)

    if load:
        print('loading model')
        model.load_state_dict(torch.load('checkpoint.pt'))

    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

    start = time.perf_counter()
    print("train start time: ", start)
    model.train()
    train_loss = []
    test_loss = []
    for e in range(epoch_n):
        epoch_loss = 0
        model.train()
        for i, data in enumerate(trainloader):
            gc.collect()
            torch.cuda.empty_cache()

            image, label = data

            image = image.to(device).float()
            label = label.to(device).float()

            pred = model(image)

            # crop_x = (label.shape[1] - pred.shape[2]) // 2
            # crop_y = (label.shape[2] - pred.shape[3]) // 2
            #
            # label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
        print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
        train_loss.append(epoch_loss / trainset.__len__())

        torch.save(model.state_dict(), 'checkpoint.pt')

        model.eval()

        total = 0
        correct = 0
        total_loss = 0

        with torch.no_grad():
            for i, data in enumerate(testloader):
                image, label = data

                image = image.to(device).float()
                label = label.to(device).float()

                pred = model(image)
                # crop_x = (label.shape[1] - pred.shape[2]) // 2
                # crop_y = (label.shape[2] - pred.shape[3]) // 2
                #
                # label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

                loss = criterion(pred, label)
                total_loss += loss.item()

                # _, pred_labels = torch.max(pred, dim=1)

                # total += label.shape[0] * label.shape[1] * label.shape[2]
                # correct += (pred_labels == label).sum().item()

            # print('Accuracy: %.4f ---- Loss: %.4f' % (correct / total, total_loss / testset.__len__()))
            test_loss.append(total_loss / testset.__len__())

    end = time.perf_counter()
    print("train time: ", end - start)
    # testing and visualization
    print("testing")
    model.eval()

    output_masks = []
    # output_labels = []

    with torch.no_grad():
        # for i in range(testset.__len__()):
        for i, data in enumerate(testloader):
            image, label = data

            image = image.to(device).float()
            pred = model(image)

            output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

            # crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
            # crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
            # labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()

            output_masks.append(output_mask)
            # output_labels.append(labels)

    Preprocess.Output(save_dir, output_masks)

    plt.show()

    plt.figure()
    plt.plot(range(1, epoch_n + 1), train_loss, 'bo', label="training")
    plt.plot(range(1, epoch_n + 1), test_loss, 'go', label="testing")
    plt.legend()
    plt.show()

def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=10, type='int')
    parser.add_option('--learning_rate', default=0.0001)
    parser.add_option('--batch_size', default=1, type='int')
    parser.add_option('--data_path', default='data')
    parser.add_option('--patch_path', default='patch_data')
    parser.add_option('--result_path', default='results')
    parser.add_option('--preprocess', default=False, type='bool')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    main(epoch_n=args.epochs, lr=args.learning_rate, data_path=args.data_path, patch_path=args.patch_path, result_path=args.result_path, batch_size=args.batch_size, preprocess=args.preprocess)