import torch
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Visualize a Test Image and save it as an image")
parser.add_argument("index", type=int, help="Which of the 1449 Test images to look at")
args = parser.parse_args()

LOAD_FROM = '/root/VOC12_After_Deeplab_Test/batch'

def saveImage(gt, pred1, pred2, img):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(2, 2)
    ax1, ax2 = axes[0]
    ax3 , ax4 = axes[1]

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

    ax1.set_title('Image')
    ax1.imshow(img)
    ax1.axis('off')

    ax2.set_title('Ground Truth')
    ax2.imshow(gt, cmap=cmap, norm=norm)
    ax2.axis('off')

    ax3.set_title('Prediction Deeplab V2')
    ax3.imshow(pred1, cmap=cmap, norm=norm)
    ax3.axis('off')

    ax4.set_title('With Postprocessing')
    ax4.imshow(pred2, cmap=cmap, norm=norm)
    ax4.axis('off')

    fig.savefig('TestImage' + str(args.index))

def changeOutput(output):
    output = output.data[0].numpy()
    output = output[:,:size[0],:size[1]]
      
    output = output.transpose(1,2,0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    return output

batch = torch.load(LOAD_FROM + str(args.index) + '.pth', map_location='cpu')
image, label, size, name = batch
size = size[0].numpy()
gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
pred1 = torch.load('/root/VOC12_After_Deeplab_Test/prediction' + str(args.index) + '.pth', map_location='cpu')
pred1 = changeOutput(pred1)
pred2 = torch.load('/root/VOC12_After_Deeplab_Test/prediction' + str(args.index) + '.pth', map_location='cpu') #change to new one!!!
pred2 = changeOutput(pred2)

img = plt.imread('/root/VOCdevkit/VOC2012/JPEGImages/' + str(batch[3][0]) + '.jpg')

saveImage(gt, pred1, pred2, img)



























