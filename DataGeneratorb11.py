# This script loads the state directory of the final version of the Deeplab V2 (from Pytorch Deeplab)
# and loads all the test data, processes it and saves its predictions for training the postprocessing
# layers.

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

# Image mean, calculated over the data set
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

# batch size could not be bigger than three in our case (error, GPU out of memory)
BATCH_SIZE = 3 # important, cause new!
DATA_DIRECTORY = '/root/VOCdevkit/VOC2012/'
DATA_LIST_PATH = '/root/Bio-snake-robot/Pytorch-Deeplab/dataset/list/train_aug.txt'
# Pascal Voc 12 has 21 classes
NUM_CLASSES = 21
# all images, 10575 single images (per batch one step)
NUM_STEPS = 3525 
# Defined size (gives better results due to higher consistency)
INPUT_SIZE = '321,321'

h, w = map(int, INPUT_SIZE.split(','))
input_size = (h, w)

# For GPU usage
gpu0 = 0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu0)
cudnn.enabled = True

# Initilization of model
model = Res_Deeplab(num_classes=21)

# load the state dictionary of the model
# IMPORTANT! HAS TO BE CHANGED HERE, USE PATH TO YOUR BEST TRAINED MODEL
pathToTrainedModel = '/root/20000StepsDefaultParametersBatch6/VOC12_scenes_20000.pth'
saved_state_dict = torch.load(pathToTrainedModel)
model.load_state_dict(saved_state_dict)

# no training/updating weights here
model.eval()

model.cuda()

# Trainloader from Pytorch deeplab, uses the pascal voc data set
trainloader = data.DataLoader(VOCDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=10, crop_size=input_size, 
                    scale=False, mirror=False, mean=IMG_MEAN), 
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

# Iterate over the trainloader for the above mentioned times
for i_iter, batch in enumerate(trainloader):
    # get the images and labels (ground truth) of each batch
    images, labels, _, _ = batch
    images = Variable(images).cuda() #gets and saves a gpu output
    
	# save them in these paths
    torch.save(images, '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/images'+ str(i_iter) + '.pth')
    torch.save(labels, '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels'+ str(i_iter) + '.pth')

	# put them through the net and save the prediction as well
    pred = interp(model(images))
    torch.save(pred, '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions'+ str(i_iter) + '.pth')
    if i_iter > NUM_STEPS:
	    break




























