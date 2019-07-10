import argparse #don't need it, actually...

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp

sys.path.append('Pytorch-Deeplab') # needed for the next 3 lines

from deeplab.model import Res_Deeplab
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import VOCDataSet
import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 3 # important, cause new!
DATA_DIRECTORY = '/root/VOCdevkit/VOC2012/'
DATA_LIST_PATH = '/root/Bio-snake-robot/Pytorch-Deeplab/dataset/list/train_aug.txt'
NUM_CLASSES = 21
NUM_STEPS = 3525 # all images
INPUT_SIZE = '321,321'

h, w = map(int, INPUT_SIZE.split(','))
input_size = (h, w)

gpu0 = 0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu0)
cudnn.enabled = True

model = Res_Deeplab(num_classes=21)

pathToTrainedModel = '/root/FirstTrain10000StepsDefaultParametersBatch6/VOC12_scenes_10000.pth'

# not the following, as we work on CPU (has to be reloaded! on CPU)
saved_state_dict = torch.load(pathToTrainedModel)

#saved_state_dict = torch.load(pathToTrainedModel, map_location=lambda storage, loc: storage)

model.load_state_dict(saved_state_dict)

model.eval()




#if we use cpu
model.cuda()

trainloader = data.DataLoader(VOCDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=10, crop_size=input_size, 
                    scale=False, mirror=False, mean=IMG_MEAN), 
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
for i_iter, batch in enumerate(trainloader):
    images, labels, _, _ = batch
    images = Variable(images).cuda() #gets and saves a gpu output, for cpu see evaluate.py
    
    torch.save(images, '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/images'+ str(i_iter) + '.pth')
    torch.save(labels, '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels'+ str(i_iter) + '.pth')

    pred = interp(model(images))
    torch.save(pred, '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions'+ str(i_iter) + '.pth')
    if i_iter > NUM_STEPS:
	    break

#i = 0
#t = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels'+ str(i) + '.pth')
#print(t.size())
#t = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/images'+ str(i) + '.pth')
#print(t.size())
#t = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions'+ str(i) + '.pth')
#print(t.size())





























