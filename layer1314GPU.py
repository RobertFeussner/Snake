import torch
import torch.nn as nn
import os
import collections
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
Module = nn.Module
from itertools import repeat

unfold = F.unfold
SIZE = 321
DOWNSAMPLE_SIZE = 50
PATHb12 = "/root/VOC12_After_b12/TrainBatch3TensorsGPU/predictions"
PATHb11 = "/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU"
BATCHES = 1 #should be changed to 3525 when I train

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
        x = F.linear(x, torch.tensor([[1.0]]))
        x = - self.min_pool(-x)
        return x


net = Net()
net.to("cuda:0")

#labels --- from Robert's part ---> ground truth
for i in range(BATCHES):
    predictions = torch.load(PATHb12 + str(i)+ '.pth') # b12 3 predictions
    predictions = predictions.float()
    #predictions = torch.nn.functional.interpolate(predictions, size=(SIZE,SIZE), mode="bilinear") #upsample them back
    predictions = predictions.cuda()
    print("predictions are")
    print(predictions)

    #labels = ground truth
    labels = torch.load(PATHb11 + str(i) + '.pth')  # b11 labels
    labels = images.float()
    #labels = torch.nn.functional.interpolate(images, size=(DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), mode="bilinear")
    labels = labels.cuda()

    print("labels are")
    print(labels)


    for j in range(3):
        prediction = predictions[j]
        #print(prediction)



