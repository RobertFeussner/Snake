import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import collections
Module = nn.Module
from itertools import repeat
from torch.autograd import Variable
import argparse
import numpy as np
import argparse
from torch.utils import data
import pickle
import cv2
import scipy.misc
import sys
import os.path as osp
import matplotlib.pyplot as plt
import random
import timeit

unfold = F.unfold
SIZE = 321
DOWNSAMPLE_SIZE = 50
PATHb12 = "/root/VOC12_After_b12/TrainBatch3TensorsGPU/predictions"
PATHb11 = "/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels"
BATCHES = 3525


LEARNING_RATE = 1.8e-4 #2.5e-4
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
IGNORE_LABEL = 255

#arguments function - from DeepLab
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
cudnn.enabled = True
cudnn.benchmark = True

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # layer 13: filter size = 9 x 9 x 21 + Xavier initialization
        self.conv1 = nn.Conv2d(in_channels=21, out_channels=105, kernel_size=(9, 9), stride=1)

        torch.nn.init.xavier_uniform_(self.conv1.weight)

        # layer 14: block min pooling layer
        self.min_pool = nn.MaxPool3d((5, 1, 1), stride=(5, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        A = torch.ones(x.size()[2], x.size()[3]).cuda()
        x = F.linear(x,A)
        x = - self.min_pool(-x)
        return x

#loss function with rescalling of target, prediction
def loss_rescale(predict, target, ignore_label):
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != ignore_label)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss_final = F.cross_entropy(predict, target, size_average=SIZE)
    return loss_final

#function to calculate the loss - difference between the prediction and the ground truth labels
def loss_calc(prediction, target):
    target = Variable(target.long()).cuda()
    return loss_rescale(prediction, target, IGNORE_LABEL)

interp = nn.Upsample(size=(SIZE,SIZE), mode='bilinear', align_corners=True)

all_predictions = []
all_labels = []


for i in range(BATCHES):
    #load results from b12
    predictions = torch.load(PATHb12 + str(i)+ '.pth') # b12 3 predictions
    predictions = predictions.float()
    #predictions = predictions.cuda()

    #load labels = ground truth
    labels = torch.load(PATHb11 + str(i) + '.pth')
    labels = labels.float()
    #labels = labels.cuda()


    for j in range(3):
        prediction = predictions[j].unsqueeze(0)
        all_predictions.append(prediction)

        label = labels[j].unsqueeze(0)
        all_labels.append(label)


model = Net()
model.to("cuda:0")
model.train()
model.cuda()


#Stochastic Gradient Descent optimizer
optimizer = optim.SGD([model.conv1.weight], lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
optimizer.zero_grad()


#train & save intermediate models
for i_iter in range(BATCHES * 3):
    optimizer.zero_grad()
    pred = Variable(interp(all_predictions[i_iter])).cuda()
    label = Variable(all_labels[i_iter])
    output = interp(model(pred))
    loss = loss_calc(output, label)
    loss.backward()
    optimizer.step()

    if (i_iter + 1) % BATCHES == 0:
        print('[Iteration %d, loss = %f]' % (i_iter, loss))

print("evaluate output")

outputs = []
#save the output for the trained model
for i_iter in range(BATCHES * 3):
    #save output in batch of 3
    pred = Variable(interp(all_predictions[i_iter])).cuda()
    outputs.append(interp(model(pred)))
    if (i_iter + 1) %3 == 0:
        j = (i_iter + 1) / 3 - 1
        print(j)
        output = torch.cat((outputs[0], outputs[1], outputs[2]), 0)
        print(output)
        torch.save(output, "/root/VOC12_After_b14/TrainBatch3TensorsGPU/predictions" + str(j) + ".pth")
        output = []










