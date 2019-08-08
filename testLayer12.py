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
NUM_CLASSES = 21
TEST_BATCHES = 1449 #1449


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

interp = nn.Upsample(size=(SIZE,SIZE), mode='bilinear', align_corners=True)

model = Net()
model.to("cuda:0")
model.cuda()

all_testdata = []

i = 11
while i < TEST_BATCHES:
    testdata = torch.load("/root/VOC12_After_b12/TrainBatch3TensorsGPUTest/predictions" + str(i) + '.pth')
    testdata = testdata[0].unsqueeze(0)
    all_testdata.append(testdata)
    i = i + 1


sys.path.append('Pytorch-Deeplab') # needed for the next 2 lines

from deeplab.model import Res_Deeplab
from collections import OrderedDict

def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')

data_list = []
for i_iter in range(len(all_testdata)):
    print(i_iter)

    pred = Variable(interp(all_testdata[i_iter])).cuda()
    output = interp(model(pred))

    test_batch_b11 = torch.load("/root/VOC12_After_Deeplab_Test/batch" + str(i_iter) + '.pth')
    image, label, size, name = test_batch_b11
    size = size[0].numpy()

    output = torch.nn.functional.softmax(output)

    output = output.cpu().data[0].numpy()

    output = output[:, :size[0], :size[1]]
    gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

    data_list.append([gt.flatten(), output.flatten()])

get_iou(data_list, NUM_CLASSES)


