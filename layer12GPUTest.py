#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os 
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter
Module = nn.Module
import collections
import os
import torch.backends.cudnn as cudnn
from itertools import repeat

unfold = F.unfold

SIZE = 321
DOWNSAMPLE_SIZE = 50
FILTERSIZE = 5
STRIDE = 1
BATCHES = 1449
COLOR_WEIGHT = 0.5
SPATIAL_WEIGHT = 0.5
FILTER_STRIDE = 2
BATCH_SIZE = 1

# In[11]:


os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
cudnn.enabled = True
cudnn.benchmark = True


# In[12]:


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# In[13]:


def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
 
    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)
 
    # N x [inC * kH * kW] x [outH * outW]
    cols = unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1).cuda()
 
    out = torch.matmul(cols, weight.reshape(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
 
    if bias is not None:
        out = out + bias.expand_as(out)
    return out


# In[14]:


class Conv2dLocal(Module):
 
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
 
    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


# In[27]:


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        #in_height, in_width, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1
        self.conv1 = Conv2dLocal(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 21, 21, FILTERSIZE, STRIDE, 2, 0, 1)


    def forward(self, x):
        x = self.conv1(x)
        A = torch.ones(x.size()[2], x.size()[3]).cuda()
        x = F.linear(x, A)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[16]:


def loadImages():

    test = (torch.randint(0, 10, (3, 3, SIZE, SIZE))) # color, height, width
    return test
    #print(test[1][0][0])
    #print(len(test))


# In[17]:


def calcDistance(color1, color2, color3, compColor1, compColor2, compColor3, imageX, imageY, x, y, middle):
    colorWeight = COLOR_WEIGHT
    spatialWeight = SPATIAL_WEIGHT
    distance = 0
    distance += colorWeight * math.sqrt((color1-compColor1)**2 + (color2-compColor2)**2 + (color3-compColor3)**2)
    distance += spatialWeight * math.sqrt((imageX-(imageX-middle+x))**2 + (imageY-(imageY-middle+y))**2)
    return distance


# In[18]:


def initializeFilters(image):
    imageHeight = DOWNSAMPLE_SIZE
    imageWidth = DOWNSAMPLE_SIZE
    filterHeight = FILTERSIZE
    filterWidth = FILTERSIZE
    filters = torch.zeros(imageHeight, imageWidth, filterHeight, filterWidth) #imageY, imageX, y, x
    filters = filters.cuda()
    middle = math.floor(filterHeight/2)
    for imageY in range(0, imageHeight, STRIDE):
        for imageX in range(0, imageWidth, STRIDE):
            for y in range(0, filterHeight, FILTER_STRIDE):
                for x in range(0, filterWidth, FILTER_STRIDE):
                    if not ((imageX + (x-middle)) < 0 or (imageX + x) > imageWidth or (imageY + (y-middle)) < 0 or (imageY + y) > imageHeight):
                        #print("test")
                        filters[imageY][imageX][y][x] = calcDistance(image[0][imageY][imageX], 
                                                                      image[1][imageY][imageX], 
                                                                      image[2][imageY][imageX], 
                                                                      image[0][imageY - middle + y][imageX - middle + x], 
                                                                      image[1][imageY - middle + y][imageX - middle + x], 
                                                                      image[2][imageY - middle + y][imageX - middle + x],
                                                                      imageX, imageY, x, y, middle)
                        #print(filters[imageY][imageX][y][x])
                    else:
                        filters[imageY][imageX][y][x] = 0
    return filters


# In[19]:

net = Net()
net.to("cuda:0")
net.eval()

for i in range(BATCHES):
    images = torch.load('/root/VOC12_After_Deeplab_Test/batch' + str(i)+ '.pth')# the 3 original images
    images = images[0]
    images = images.float()
    images = torch.nn.functional.interpolate(images, size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), mode="bilinear")
    images = images.cuda()

    predictions = torch.load('/root/VOC12_After_Deeplab_Test/prediction' + str(i)+ '.pth') # b11 3 predictions
    predictions = predictions.float()
    predictions = torch.nn.functional.interpolate(predictions, size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), mode="bilinear")
    predictions = predictions.cuda()

    outputs = []

    for j in range(BATCH_SIZE):
        image = images[j] # iterate every picture directly
        prediction = predictions[j]
        filters = initializeFilters(image)
        filters.unsqueeze_(-3)
        filters.unsqueeze_(-3)
        filters = filters.expand(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE, 21, 21, FILTERSIZE, FILTERSIZE)
        print(filters.size())
        net.conv1.weight = torch.nn.Parameter(filters)
        input = prediction.unsqueeze(0)

        outputs.append(net(input))

    output = outputs[0]
    torch.save(output, "/root/VOC12_After_b12/TrainBatch3TensorsGPU/predictions"+ str(i)+".pth")
    print(i)    

print("finished")
