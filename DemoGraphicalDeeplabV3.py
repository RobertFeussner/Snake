from tkinter import *
from PIL import Image
import os #for removing the temporary files

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models

img = None
imgForSaving = None

# IMPORTANT! AUTOMATICALLY DOWNLOADS THE STATE DICTIONARY OF THE PRETRAINED DEEPLAB V3
# (Can take some time and space)

# Load the on the coco dataset pretrained Deeplab V3 Model from the official torchvision model website
# Define 21 classes to be consistent with the Pascal Voc 2012 dataset
deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21)

# First off, resize the image so its size is consistent with the rest of the pictures
# Then convert it to a tensor and then normalize it (values computed over big datasets)
trans = torchvision.transforms.Compose([torchvision.transforms.Resize(510), torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

# used for evaluation only, no need for training here
deeplab.eval()

if torch.cuda.is_available():
	deeplab.cuda()

root = Tk()

# Define the helper function
# Turns the a predicted tensor into the 21 
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



# Load an image according to the given path in entryLoad and show it in its grid
def load():
	loadingPath = entryLoad.get()
	global img
	img = Image.open(loadingPath)
	#saves it as a png, loads and destroys it (tkinter only shows png) 
	img.save('temp7543892758.png')
	imgtemp = PhotoImage(file='temp7543892758.png')
	os.remove('temp7543892758.png')
	loadedImageLabel.configure(image=imgtemp)
	loadedImageLabel.image = imgtemp
	
def segment():
	global img
	global imgForSaving
	# transform the image for passing through the net
	tensorImage = trans(img)
	# Unsqueeze the single picture, so that it is a batch of one (Pythorch only works with batches of one)
	tensorImage = tensorImage.unsqueeze(0)
	if torch.cuda.is_available():
		tensorImage = Variable(tensorImage).cuda()
	prediction = deeplab(tensorImage) # Takes a lot of computation and memory!
	if torch.cuda.is_available():
		prediction = prediction.cpu()
	
	# prediction = torch.load('out.pth') # For mockup, fast and dirty, loads an old result
	
	# prediction is actually a dictionary with its result saved in 'out'
	pred = prediction['out']
	# Undo the unsqueezing
	pred = pred.squeeze()
	# Important, from the first dimension, where each of the 21 indices refers to one class, 
	# we take the index with the highest probability and save it in a numpy array
	pred = torch.argmax(pred, dim=0).numpy()
	
	# Get the picture with the help function and show it
	predImage = decode_segmap(pred)
	fig, ax = plt.subplots()
	ax.imshow(predImage)
	ax.axis('off')
	fig.savefig('temp7543892758.png')
	path = entrySave.get()
	if(path != ''):
		fig.savefig(path)
	predImage = PhotoImage(file='temp7543892758.png')
	os.remove('temp7543892758.png')
	segmentedImageLabel.configure(image=predImage)
	segmentedImageLabel.image=predImage

# Just the boring UI stuff... Still, had to be done

fileLoad=Label(root, text='Enter path to image to load here:')
fileSave=Label(root, text='Enter path for segmentation to save here (Saves if not empty):')

entryLoad = Entry(root, width=60)
entrySave = Entry(root, width=60)

buttonLoad = Button(root, text='Load', command=load, width=6)
buttonSegment = Button(root, text='Segment', command=segment, height=6, width=14)

loadedImageLabel = Label(root)
segmentedImageLabel = Label(root)
legendImage = PhotoImage(file='legend.png')
legend = Label(root, image=legendImage)

fileLoad.grid(row=0,column=0, columnspan=2, sticky=W)
entryLoad.grid(row=1,column=0, sticky=W)
buttonLoad.grid(row=1,column=1,sticky=W)

fileSave.grid(row=2,column=0, columnspan=2, sticky=W)
entrySave.grid(row=3,column=0, sticky=W)

buttonSegment.grid(row=0,column=2, rowspan=4, sticky=W)

loadedImageLabel.grid(row=4,column=0, columnspan=2, sticky=W)
segmentedImageLabel.grid(row=4,column=2,sticky=W)

legend.grid(row=5,column=0, columnspan=3, sticky=W)

mainloop()















