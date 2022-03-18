import torch
import torch.nn as nn
import torch.nn.functional as F

#import any other libraries you need below this line

class twoConvBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock, self).__init__()
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

class downStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(downStep, self).__init__()
    #initialize the down path
    self.max_pooling = nn.MaxPool2d(2)
    self.conv = twoConvBlock(input_channel, output_channel)

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
    self.conv = twoConvBlock(input_channel, output_channel)

  def forward(self, x1, x2):
    #implement the forward path
    x1 = self.upsampling(x1)

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x2 = F.pad(x2, [-diffX // 2, -diffX // 2, -diffY // 2, -diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x

class Out(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(Out, self).__init__()
    self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=1)

  def forward(self, x):
    return self.conv(x)

class UNet_A1(nn.Module):
  def __init__(self, channels, classes):
    super(UNet_A1, self).__init__()
    # initialize the complete model
    self.input = twoConvBlock(channels, 64)
    self.down1 = downStep(64, 128)
    self.down2 = downStep(128, 256)
    self.down3 = downStep(256, 512)
    self.down4 = downStep(512, 1024)
    self.up1 = upStep(1024, 512)
    self.up2 = upStep(512, 256)
    self.up3 = upStep(256, 128)
    self.up4 = upStep(128, 64)
    self.output = Out(64, classes)

  def forward(self, x):
    #implement the forward path
    x1 = self.input(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.output(x)
    return x



