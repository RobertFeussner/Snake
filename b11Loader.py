# Generic code for loading the results of the Deeplab V2 model for being included in the code for postprocessing

import torch

BATCHES = 5 #probably tops
PATH = '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU'

# iterates over all the batches (as many as one wants)
for i in range(BATCHES):
	labels = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels' + str(i)+ '.pth')	# three ground truth labels
	images = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/images' + str(i)+ '.pth')	# the 3 original images
	predictions = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions' + str(i)+ '.pth') # b11 3 predictions
	# each batch has three compontens, you can access them individually like in the following loop
	for j in range(3):
		image = images[j] # iterate every picture directly
		print(image)