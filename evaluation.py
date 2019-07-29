import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
from torch.utils import data
import scipy.misc
import sys
import os
import os.path as osp

num_classes = 21
BATCHES = 3
SIZE = 321

#predicted = our result
#target = ground_truth

'''
class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass

class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=num_classes**2)
        assert bincount_2d.size == num_classes**2
        conf = bincount_2d.reshape((num_classes, num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confusion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

def evaluate(predicted, target):
    normalized = False
    ignore_index = None
    if ignore_index is None:
        ignore_index = None
    elif isinstance(ignore_index, int):
        ignore_index = (ignore_index,)
    else:
        ignore_index = tuple(ignore_index)

    conf_metric = ConfusionMatrix(normalized)

    assert predicted.size(0) == target.size(0), \
        'number of targets and predicted outputs do not match'
    assert predicted.dim() == 3 or predicted.dim() == 4, \
        "predictions must be of dimension (N, H, W) or (N, K, H, W)"
    assert target.dim() == 3 or target.dim() == 4, \
        "targets must be of dimension (N, H, W) or (N, K, H, W)"

    # If the tensor is in categorical format convert it to integer format
    if predicted.dim() == 4:
        _, predicted = predicted.max(1)
    if target.dim() == 4:
        _, target = target.max(1)

    conf_metric.add(predicted.view(-1), target.view(-1))

    conf_matrix = conf_metric.value()
    if signore_index is not None:
        for index in ignore_index:
            conf_matrix[:, ignore_index] = 0
            conf_matrix[ignore_index, :] = 0
    true_positive = np.diag(conf_matrix)
    false_positive = np.sum(conf_matrix, 0) - true_positive
    false_negative = np.sum(conf_matrix, 1) - true_positive

    # Just in case we get a division by 0, ignore/hide the error
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / (true_positive + false_positive + false_negative)

    return iou, np.nanmean(iou)

'''
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
cudnn.enabled = True
cudnn.benchmark = True

all_predictions = []
all_targets = []


for i in range(BATCHES):
    #load results from b12
    predictions = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions' + str(i)+ '.pth')
    predictions = predictions.float()

    #load labels = ground truth
    targets = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels' + str(i) + '.pth')
    targets = targets.float()

    images = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/images' + str(i) + '.pth')
    images = images.float()


    for j in range(3):
        image = images[j]
        prediction = predictions[j]

        size = image.shape
        size = np.array(size)
        #output = model(Variable(image, volatile=True).cuda(gpu0))
        #output = output.cpu().data[0].numpy()

        output = prediction[:, :size[0], :size[1]]
        gt = np.asarray(targets[j].numpy()[:size[0], :size[1]], dtype=np.int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
        all_predictions.append([gt.flatten(), output.flatten()])

        print("predictions")
        print(output)

        target = targets[j].unsqueeze(0)
        all_targets.append(target)
        print("ground truth")
        print(target)


for i in range(3 * BATCHES):
    print("evaluation")
    #print(evaluate(all_predictions[i],all_targets[i]))
    print("\n")




