## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # input image size: 224x224
        
        #Convlution 1
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        
        # maxpool layer 1
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        #Convlution 2
        # 32 feature-map images input(110x110), 64 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (64, 108, 108)
        self.conv2 = nn.Conv2d(32, 64, 3)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        
        # maxpool layer 2
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (64, 54, 54)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        #Convlution 3
        # 64 feature-map images input(54x54), 128 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for one image, will have the dimensions: (128, 52, 52)
        self.conv3 = nn.Conv2d(64, 128, 3)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        
        # maxpool layer 3
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (128, 26, 26)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        #Convlution 4
        # 128 feature-map images input(26x26), 256 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output Tensor for one image, will have the dimensions: (256, 24, 24)
        self.conv4 = nn.Conv2d(128, 256, 3)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        
        # maxpool layer 4
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (256, 12, 12)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        #Convlution 5
        # 256 feature-map images input(12x12), 512 output channels/feature maps
        # 1x1 square convolution kernel
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output Tensor for one image, will have the dimensions: (512, 10, 10)
        self.conv5 = nn.Conv2d(256, 512, 3)
        torch.nn.init.xavier_uniform(self.conv5.weight)
        
        # maxpool layer 5
        # pool with kernel_size=2, stride=2
        # after one pool layer, this becomes (512, 5, 5)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # 2 full-connect layer
        #128x31x31
        self.fc1 = nn.Linear(512*5*5 , 1024)
        self.fc2 = nn.Linear(1024,136)

        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
