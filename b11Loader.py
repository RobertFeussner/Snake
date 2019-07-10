import torch

BATCHES = 5 #probably tops
PATH = '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU'

for i in range(BATCHES):
	labels = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/labels' + str(i)+ '.pth')	# three ground truth labels
	images = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/images' + str(i)+ '.pth')	# the 3 original images
	predictions = torch.load('/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions' + str(i)+ '.pth') # b11 3 predictions
	for j in range(3):
		image = images[j] # iterate every picture directly
		print(image)