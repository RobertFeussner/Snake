import torch
import numpy as np

from eval_segm import *

pathL = 'C:\\Users\\Robert Feussner\\Desktop\\ForTesting\\labels0.pth'
pathP = 'C:\\Users\\Robert Feussner\\Desktop\\ForTesting\\predictions0.pth'

pred = torch.load(pathP, map_location='cpu')
label = torch.load(pathL, map_location='cpu')
pred = pred[0]
pred = torch.argmax(pred, dim=0).numpy()
label = label[0].detach().numpy()
acc = pixel_accuracy(pred, label)
print(acc)

# out = torch.argmax(out, dim=0).numpy()