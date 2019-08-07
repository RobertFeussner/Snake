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
TEST_BATCHES = 1449
LEARNING_RATE = 1.9e-4 #1.8e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0005
IGNORE_LABEL = 255
NUM_CLASSES = 21

#arguments function
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--betas", type=float, default=BETAS,
                        help="Momentum component of the optimiser.")
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
all_testdata = []

main_phase = 'eval'

for i in range(BATCHES):
    #load results from b12
    predictions = torch.load(PATHb12 + str(i)+ '.pth') # b12 3 predictions
    predictions = predictions.float()

    #load labels = ground truth
    labels = torch.load(PATHb11 + str(i) + '.pth')
    labels = labels.float()


    for j in range(3):
        prediction = predictions[j].unsqueeze(0)
        all_predictions.append(prediction)

        label = labels[j].unsqueeze(0)
        all_labels.append(label)

for i in range(TEST_BATCHES):
    testdata = torch.load("/root/VOC12_After_b12/TrainBatch3TensorsGPUTest/predictions" + str(i) + '.pth')
    testdata = testdata[0].unsqueeze(0)
    all_testdata.append(testdata)


index = int(0.8 * BATCHES * 3)
train_data = all_predictions[:index]
train_data_labels = all_labels[:index]

val_data = all_predictions[(index+1):]
val_data_labels = all_labels[(index+1):]

model = Net()
model.to("cuda:0")
model.cuda()

# Adam optimizer
optimizer = optim.Adam([model.conv1.weight], lr=args.learning_rate, betas=args.betas, eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

train_loss_history = []
val_loss_history = []


#train & save model
if main_phase == 'not_eval':
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        if phase == 'train':
            log_nth = len(train_data) - 1
            for i_iter in range(len(train_data)):
                optimizer.zero_grad()
                pred = Variable(interp(train_data[i_iter])).cuda()
                label = Variable(train_data_labels [i_iter])
                output = interp(model(pred))
                loss = loss_calc(output, label)
                loss.backward()
                optimizer.step()

                train_loss_history.append(loss.data.cpu().numpy())
                if i_iter % log_nth == 0:
                    last_log_nth = train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth)
                    print('train loss: %.3f' % train_loss)


        if phase == 'val':
            log_nth = len(val_data) - 1
            for i_iter in range(len(val_data)):
                pred = Variable(interp(val_data[i_iter])).cuda()
                label = Variable(val_data_labels[i_iter])
                output = interp(model(pred))
                loss = loss_calc(output, label)

                val_loss_history.append(loss.data.cpu().numpy())
                if i_iter % log_nth == 0:
                    last_log_nth = val_loss_history[-log_nth:]
                    validation_loss = np.mean(last_log_nth)
                    print('validation loss: %.3f' % validation_loss)

            torch.save(model, "/root/VOC12_After_b14/TrainBatch3TensorsGPU/big_lr/model")


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
if main_phase == 'eval':
    model = torch.load("/root/VOC12_After_b14/TrainBatch3TensorsGPU/big_lr/model")
    for i_iter in range(TEST_BATCHES):
        #save test output in batch of 1
        pred = Variable(interp(all_testdata[i_iter])).cuda()
        output = interp(model(pred))
        torch.save(output, "/root/VOC12_After_b14/TrainBatch3TensorsGPUTest/predictions" + str(i_iter) + ".pth")

        test_batch_b11 = torch.load("/root/VOC12_After_Deeplab_Test/batch" + str(i_iter) + '.pth')
        image, label, size, name = test_batch_b11
        size = size[0].numpy()

        output = output.cpu().data[0].numpy()

        output = output[:, :size[0], :size[1]]
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        data_list.append([gt.flatten(), output.flatten()])

    get_iou(data_list, NUM_CLASSES)











