import torch
import torch.nn as nn
import torch.nn.functional as F

#import any other libraries you need below this line

class twoConvBlock_Down(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock_Down, self).__init__()
    #initialize the block
    self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3)
    self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3)
    self.relu = nn.ReLU(True)
    self.batch_norm = nn.BatchNorm2d(output_channel)

  def forward(self, x):
    # implement the forward path
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm(x)
    x = self.relu(x)
    return x

class twoConvBlock_Mid(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock_Mid, self).__init__()
    #initialize the block
    self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
    self.relu = nn.ReLU(True)
    self.batch_norm = nn.BatchNorm2d(output_channel)

  def forward(self, x):
    # implement the forward path
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm(x)
    x = self.relu(x)
    return x

class twoConvBlock_Up(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock_Up, self).__init__()
    #initialize the block
    self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=2)
    self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=2)
    self.relu = nn.ReLU(True)
    self.batch_norm = nn.BatchNorm2d(output_channel)

  def forward(self, x):
    # implement the forward path
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch_norm(x)
    x = self.relu(x)
    return x


class downStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(downStep, self).__init__()
    #initialize the down path
    self.max_pooling = nn.MaxPool2d(2)
    self.conv = twoConvBlock_Down(input_channel, output_channel)

  def forward(self,x):
    # implement the forward path
    x = self.max_pooling(x)
    x = self.conv(x)
    return x


class upStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(upStep, self).__init__()
    #initialize the up path
    self.upsampling = nn.ConvTranspose2d(input_channel, input_channel // 2, kernel_size=2, stride=2)
    self.conv = twoConvBlock_Up(input_channel, output_channel)

  def forward(self, x1, x2):
    #implement the forward path
    x1 = self.upsampling(x1)
    # print("y1: " + str(x1.size()) + " y2: " + str(x2.size()))
    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x

class Mid(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(Mid, self).__init__()
    #initialize the down path
    self.max_pooling = nn.MaxPool2d(2)
    self.conv = twoConvBlock_Mid(input_channel, output_channel)

  def forward(self, x):
    # implement the forward path
    x = self.max_pooling(x)
    x = self.conv(x)
    return x


class Out(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(Out, self).__init__()
    self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=1)

  def forward(self, x):
    return self.conv(x)

class UNet(nn.Module):
  def __init__(self, channels, classes):
    super(UNet, self).__init__()
    # initialize the complete model
    self.input = twoConvBlock_Down(channels, 64)
    self.down1 = downStep(64, 128)
    self.down2 = downStep(128, 256)
    self.down3 = downStep(256, 512)
    self.mid = Mid(512, 1024)
    self.up1 = upStep(1024, 512)
    self.up2 = upStep(512, 256)
    self.up3 = upStep(256, 128)
    self.up4 = upStep(128, 64)
    self.output = Out(64, classes)

  def forward(self, x):
    #implement the forward path
    x1 = self.input(x)
    # print("x1: " + str(x1.size()))
    x2 = self.down1(x1)
    # print("x2: " + str(x2.size()))
    x3 = self.down2(x2)
    # print("x3: " + str(x3.size()))
    x4 = self.down3(x3)
    # print("x4: " + str(x4.size()))
    x5 = self.mid(x4)
    # print("x5: " + str(x5.size()))
    x = self.up1(x5, x4)
    # print("x: " + str(x.size()))
    x = self.up2(x, x3)
    # print("x: " + str(x.size()))
    x = self.up3(x, x2)
    # print("x: " + str(x.size()))
    x = self.up4(x, x1)
    # print("x: " + str(x.size()))
    x = self.output(x)
    # print("x: " + str(x.size()))
    return x



