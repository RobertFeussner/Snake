# This implementation enables you to segment single images and save them only using a terminal
# Uses the Pytorch Deeplab V2 implemetation model
# CUDA has to be enabled

import argparse
import random
import matplotlib.pyplot as plt
import collections
import torchvision
from torch.utils import data

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Get the arguments to know what to do 

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("picture", type=str, help="Name of the picture in the same directory without jpg ending.")
parser.add_argument("--AimDir", type=str, default='',help="Path to the directory to save the result.")
parser.add_argument("--postProcessing", action="store_true",help="Whether to use Postprocessing.")
parser.add_argument("--PathToPretrainedModel", type=str, help="Path to the pretrained Model. Default is /root/20000StepsDefaultParametersBatch6/VOC12_scenes_20000.pth")
args = parser.parse_args()

PATH = args.picture + '.jpg'

# Approximately the mean
mean=(128, 128, 128)

# get the pictures and save their size (needed later on for changing the tensor to an image)
image = cv2.imread(PATH, cv2.IMREAD_COLOR)
size = image.shape
np.array(size)
image = np.asarray(image, np.float32)
# Subtract the mean (for normalization and numerical stability)
image -= mean

# Prepare it for the model
img_h, img_w, _ = image.shape
inter_size = (img_h, img_w)
image = image.transpose((2, 0, 1))
image = torch.from_numpy(image)

# Change the single picture to a batch (otherwise a Pytorch Net cannot work with it)
image = image.unsqueeze(0)

# Prepare for the use of the GPU 
gpu0 = 0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu0)
cudnn.enabled = True

# Initilize model
model = Res_Deeplab(num_classes=21)

# We use the path to the state dictionary on our own server as a default, can be changed with an argument
pathToTrainedModel = '/root/20000StepsDefaultParametersBatch6/VOC12_scenes_20000.pth'
if args.PathToPretrainedModel:
    pathToTrainedModel = PathToPretrainedModel
saved_state_dict = torch.load(pathToTrainedModel)
model.load_state_dict(saved_state_dict)

#prepare for evaluation and use of cuda (GPU)
model.eval()
model.cuda()

image = Variable(image).cuda() #gets and saves a gpu output

interp = nn.Upsample(size=inter_size, mode='bilinear', align_corners=True)

# puts the tensor through the model and returns the prediction (the segmentation)
pred = interp(model(image))

# Converts the prediciton to a image and saves it under the atop defined name (given with the arguments)
def saveImage(pred):
    output = pred.cpu().data[0].numpy()

    output = output[:,:size[0],:size[1]]
    output = output.transpose(1,2,0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

    fig, ax = plt.subplots()

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Save the Image 
    ax.set_title('Prediction')
    ax.imshow(output, cmap=cmap, norm=norm)
    ax.axis('off')
    s = '' + args.AimDir + args.picture + '.png'
    fig.savefig(s)

# actually save our image via the function above
if args.postProcessing:
    print('Sorry, not implemented yet!')
else:
    saveImage(pred)


























