import torch
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Load a prediction tensor and save it as an image")
parser.add_argument("tensor", type=str, help="Path of the prediction tensor batch")
parser.add_argument("index", type=int,help="Which of the three pictures per batch to look at")
parser.add_argument("saveAs", type=str ,help="Name for saving")
args = parser.parse_args()

# Define the helper function
def decode_segmap(image, nc=21):
   
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
 
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb

def saveImage(prediction):
    prediction = torch.argmax(prediction, dim=0).numpy()
    img = decode_segmap(prediction)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    fig.savefig(args.saveAs)
	
predictionBatch = torch.load(args.tensor, map_location='cpu')

prediction = predictionBatch[args.index]
saveImage(prediction)