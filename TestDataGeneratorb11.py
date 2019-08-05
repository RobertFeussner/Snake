import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data

sys.path.append('Pytorch-Deeplab') # needed for the next 2 lines

from deeplab.model import Res_Deeplab
from deeplab.datasets import VOCDataSet
from collections import OrderedDict
import os

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/root/VOCdevkit/VOC2012/'
DATA_LIST_PATH = '/root/Snake/Pytorch-Deeplab/dataset/list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = '/root/20000StepsDefaultParametersBatch6/VOC12_scenes_20000.pth'

SAVE_TO = '/root/VOC12_After_Deeplab_Test'


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


def main():
    """Create the model and start the evaluation process."""

    gpu0 = 0

    model = Res_Deeplab(num_classes=NUM_CLASSES)
    
    saved_state_dict = torch.load(RESTORE_FROM)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(VOCDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(321, 321), mean=IMG_MEAN, scale=False, mirror=False), 
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(321, 321), mode='bilinear', align_corners=True) #changed to model 321,321
    data_list = []

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        torch.save(batch, SAVE_TO + '/batch' + str(index) + '.pth') # Save the batch
        image, label, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda(gpu0))

        output = interp(output)
        torch.save(batch, SAVE_TO + '/prediction' + str(index) + '.pth') #Save b11 prediction

        output = output.cpu().data[0].numpy()

        output = output[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        data_list.append([gt.flatten(), output.flatten()])

    get_iou(data_list, NUM_CLASSES)


if __name__ == '__main__':
    main()
