import torch
import torch.nn as nn
import os
import collections
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
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
num_epochs = 3
SIZE = 321
DOWNSAMPLE_SIZE = 50
PATHb12 = "/root/VOC12_After_b12/TrainBatch3TensorsGPU/predictions"
PATHb11 = "/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels"
BATCHES = 1 #should be changed to 3525 when I train


LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_STEPS = 10000
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
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
cudnn.enabled = True
cudnn.benchmark = True


def loss_rescale(predict, target, ignore_label):
    """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
    """
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != ignore_label)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss_final = F.cross_entropy(predict, target, size_average=SIZE)
    return loss_final

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

#function to calculate the loss - difference between the prediction and the ground truth labels
def loss_calc(prediction, target):
    target = Variable(target.long()).cuda()
    return loss_rescale(prediction, target, IGNORE_LABEL)

#functions to get the learning rate - from DeepLab
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    #optimizer.param_groups[1]['lr'] = lr * 10


model = Net()
model.to("cuda:0")
model.train()
model.cuda()

for i in range(BATCHES):
    #load results from b12
    predictions = torch.load(PATHb12 + str(i)+ '.pth') # b12 3 predictions
    predictions = predictions.float()
    predictions = torch.nn.functional.interpolate(predictions, size=(SIZE,SIZE), mode="bilinear") #upsample back to 321 x 321
    predictions = predictions.cuda()

    #load labels = ground truth
    labels = torch.load(PATHb11 + str(i) + '.pth')
    labels = labels.float()
    labels = labels.cuda()

    images = torch.load(PATHb11 + str(i) + '.pth')
    images = labels.float()
    images = labels.cuda()

    all_predictions = []
    all_labels = []
    all_images = []


    for j in range(3):
        prediction = predictions[j].unsqueeze(0)
        all_predictions.append(prediction)

        label = labels[j].unsqueeze(0)
        all_labels.append(label)

        image = images[j]
        all_images.append(image)

#Stochastic Gradient Descent optimizer
optimizer = optim.SGD([model.conv1.weight], lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
optimizer.zero_grad()

#get
train_loss_history = []
train_acc_history = []

interp = nn.Upsample(size=(SIZE,SIZE), mode='bilinear', align_corners=True)

for epoch in range(num_epochs):
    for i_iter in range(BATCHES):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        pred = Variable(all_predictions[i_iter]).cuda()
        label = Variable(labels[i_iter])

        output = interp(model(pred))

        print(output.size())
        print(label.size())

        loss = loss_calc(output, label)

        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.data.cpu().numpy())

        classif_targets = label >= 0
        train_accuracy = np.mean((pred == label)[classif_targets].data.cpu().numpy())
        train_acc_history.append(train_accuracy)

        if i_iter % 1:
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, train_accuracy, train_loss))







