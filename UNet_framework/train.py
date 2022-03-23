from model import UNet
from model_A1 import UNet_A1
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
from metrics import PSNR, SSIM

def main(epoch_n, lr, data_path, patch_path, result_path, batch_size, preprocess, filetype, eval_mode):

    # Paramteres
    start = time.perf_counter()
    print("start time: ", start)
    # learning rate
    # lr = 1e-4
    # number of training epochs
    # epoch_n = 5
    # input image-mask size
    image_size = 572
    # root directory of project
    root_dir = os.getcwd()
    # training batch size
    # batch_size = 6
    # use checkpoint model for training
    if eval_mode == 1:
        load = True
    else:
        load = False
    # use GPU for training
    gpu = True

    psnr = PSNR(data_range=1., reduction='sum')
    ssim = SSIM(channels=3, data_range=1., reduction='sum')

    data_dir = os.path.join(root_dir, data_path)
    new_data_dir = os.path.join(root_dir, patch_path)
    save_dir = os.path.join(root_dir, result_path)

    if not os.path.exists(data_dir):
        print('data folder not existed')
    if not os.path.exists(new_data_dir):
        if preprocess == 1:
            os.makedirs(new_data_dir)
        else:
            print("patch images folder not existed")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print('preproc: ' + data_dir)
    print('preproc: ' + new_data_dir)

    if preprocess:
        Preprocess.Preprocess(data_dir, new_data_dir, filetype)
    # data_dir = "./data/cells/"

    trainset = Cell_data(data_dir=new_data_dir, size=image_size, filetype=filetype)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = Cell_data(data_dir=new_data_dir, size=image_size, train=False, filetype=filetype)
    testloader = DataLoader(testset, batch_size=batch_size)

    device = torch.device('cuda:0' if gpu else 'cpu')
    print(device)

    channels = 3
    classes = 3
    # model = UNet(channels, classes).to('cuda:0').to(device)
    model = UNet(channels, classes).to(device)
    # model = UNet_A1(channels, classes).to(device)


    if load:
        print('loading model')
        model.load_state_dict(torch.load('checkpoint.pt'))

    criterion = nn.L1Loss()

    # optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)


    if eval_mode != 1:
        model.train()
        train_loss = []
        test_loss = []
        best_loss = 999999
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

                # #####################  only for A1 ##################### #
                # crop_x = (label.shape[2] - pred.shape[2]) // 2
                # crop_y = (label.shape[3] - pred.shape[3]) // 2
                # label = label[:, :, crop_x: label.shape[2] - crop_x, crop_y: label.shape[3] - crop_y]
                # #####################  only for A1 ##################### #

                loss = criterion(pred, label)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

                print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
            print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
            train_loss.append(epoch_loss / trainset.__len__())

            model.eval()

            total_loss = 0

            with torch.no_grad():
                for i, data in enumerate(testloader):
                    image, label = data

                    image = image.to(device).float()
                    label = label.to(device).float()

                    pred = model(image)
                    # #####################  only for A1 ##################### #
                    # crop_x = (label.shape[2] - pred.shape[2]) // 2
                    # crop_y = (label.shape[3] - pred.shape[3]) // 2
                    # label = label[:, :, crop_x: label.shape[2] - crop_x, crop_y: label.shape[3] - crop_y]
                    # #####################  only for A1 ##################### #

                    loss = criterion(pred, label)
                    total_loss += loss.item()

                    # calculate PSNR and SSIM
                    psnr_batch = psnr(pred, label)
                    ssim_batch = ssim(pred, label)
                    print('validation batch %d --- PSNR: %.4f' % (i, psnr_batch))
                    print('validation batch %d --- SSIM: %.4f' % (i, ssim_batch))

                test_loss.append(total_loss / testset.__len__())
            if epoch_loss < best_loss:
                torch.save(model.state_dict(), 'checkpoint.pt')
            print("--------------------------------------------------------------------------")
            print(" ")

        end = time.perf_counter()
        print("train time: ", end - start)


    #eval mode

    # testing and visualization
    print("testing...")
    model.eval()

    inputs = []
    output_masks = []
    output_labels = []

    with torch.no_grad():
        for i in range(testset.__len__()):
            image, label = testset.__getitem__(i)

            image = image.to(device).float().unsqueeze(0)
            label = label.to(device).float()
            pred = model(image)
            # #####################  only for A1 ##################### #
            # crop_x = (label.shape[1] - pred.shape[2]) // 2
            # crop_y = (label.shape[2] - pred.shape[3]) // 2
            # label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]
            # #####################  only for A1 ##################### #

            psnr_batch = psnr(pred, label)
            ssim_batch = ssim(pred, label)
            print('evaluation image %d --- PSNR: %.4f' % (i, psnr_batch))
            print('evaluation image %d --- SSIM: %.4f' % (i, ssim_batch))
            print("============================================================")


            inputs.append(image.squeeze(0))
            output_masks.append(pred.squeeze(0))
            output_labels.append(label.squeeze(0))

    Preprocess.Output(save_dir, output_masks, output_labels, inputs, filetype)

    if eval_mode != 1:
        plt.show()

        plt.figure()
        plt.plot(range(1, epoch_n + 1), train_loss, 'bo', label="training")
        plt.plot(range(1, epoch_n + 1), test_loss, 'go', label="testing")
        plt.legend()
        plt.show()

def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=10, type=int)
    parser.add_option('--learning_rate', default=0.0001)
    parser.add_option('--batch_size', default=1, type=int)
    parser.add_option('--data_path', default='data')
    parser.add_option('--patch_path', default='patch_data')
    parser.add_option('--result_path', default='results')
    parser.add_option('--preprocess', default=0, type=int)
    parser.add_option('--filetype', default='exr')
    parser.add_option('--eval_mode', default=0, type=int)


    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    main(epoch_n=args.epochs, lr=args.learning_rate, data_path=args.data_path, patch_path=args.patch_path,
         result_path=args.result_path, batch_size=args.batch_size, preprocess=args.preprocess, filetype=args.filetype, eval_mode=args.eval_mode)