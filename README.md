# Vision Sub-system of the Snake Robot
## Semantic Segmentation Using Fully Convolutional Network and Markov Random Field

This project is a reimplememtation of the paper "[Deep Learning Markov Random Field for Semantic Segmentation](http://personal.ie.cuhk.edu.hk/~pluo/pdf/DPN_pami.pdf)"(Liu et al) for providing a Vision System for the Snake Robot Project of the Technical university of Munich. We also used the paper "[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)"(Long et al) in order to get a deeper grasp of Fully Convolutional Networks. For training and evaluation we used the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset and its [augmented supplement](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).

## Single Programs Explained

In this section we explain our code and the code we used from other sources and give some examples of how to use it.

### Pytorch Deeplab

This folder contains a Github [Deeplab V2](https://github.com/speedinghzl/Pytorch-Deeplab) implementation based on the ResNet101. As it is written in Python 2, whereas we used Python 3, which is not backwards compatible, we had to change it a little. Also, we have done some adaptations to fit our project better.
We used this to train our Deeplab V2 model and do our evaluations on it. We got a final accuracy of around 73% on the test set, similiar to the result they have in their project.
Please make sure, that you need to download all the files  mentioned on their project site, as they are required for it to work and cannot be downloaded by git directly (files too big).
#### Plot Accuracy
For plotting the accuracay that was saved in log files using nohup, additionally the script __plotLossAndAccuracy.py__ was written. Using matplotlib, it visualizes the decrease of the loss as well as a slight increase of the accuracy during training.

### DataGeneratorb11

Once you have trained your model and saved its state dictonary via the Pytorch Deeplab implementation, you can use the _DataGeneratorb11.py_ in order to run all the 10575 training images through the model and save its predictions, the groundtruth and the images in batches of a size of your choice. They can be used for postprocessing. It is important that you change __pathToTrainedModel__ inside the script. Also make sure that all the given paths exist in that way, if necessary change them. It should be mentioned that folder are not created automatically, they have to be created manually. Then, you can simply run it with the default parameters with the following command:

```
python DataGeneratorb11
```

If you want to change other parameters, e.g. need smaller batches, because the GPU does not have enough space, you can change it directly in the file itself (it is commented).

### b11Loader

This script itself (_b11Loader.py_) has no function on its own, but explains how to load the data produced by DataGeneratorb11 for postprocessing purposes.

### TestDataGeneratorb11

Similiar to the DataGeneratorb11, this script loads the test images, passes them through the final model and prepares the data for postprocessing. Simultaneously it calculates the Mean IoU over the test set and offers thus an evalution possibility. Be sure to adapt all the paths in the script before running it via

```
python TestDataGeneratorb11.py
```

### Layer 12 (b12)

There are two separate files layer12GPU.py (which is used to process the training data) and layer12GPUTest.py (which is used to process the test data). The main difference is only that we used different paths and batch sizes for our training and test data. If you want to run your own data through this layer, make sure all the paths are adapted within the file. The file automatically takes the input images and predictions, calculates the its own custom filters and uses them in the convolution operation. There are also some hyperparameters that can be adapted at the top of the file. The file can just be run via

```
python layer12GPU.py
```
or
```
python layer12GPUTest.py
```
### Layers 13-14 (b13-14)

This file contains the implementation of layer13-14, which constains the architecture of the layer, the training, validation and the evaluation of layers. The main_phase variable, which can be 'eval' or 'not_eval', sets the run to use the training/validation or the evaluation of the model on the test data. The hyperparameters and the data loaded are the ones which got the best performance. The file can be run via

```
python layer1314GPU.py
```

### layer15
This script takes data from layer14 and computes the corresponding output for layer 15 and mIoU. To adapt which data to take in or where to save the output data, simply adapt the file paths within the script. For the training and testing data sets these parameters are already provided (at the time of submission, training is commented out and testing is not). Further explanations can be found in the detailed commments within the script. Finally, to run the script navigate to the Snake directory and run the following command
```
python layer15.py
```

### Demo

The script _Demo.py_ offers a simple terminal based way to segment single images without postprocessing. By passing it the required arguments, you can simply segment images that are new to the model. It uses the Pytorch Deeplab V2 model. An example use case would be a Snake robot locating and classifing objects. If the you have the same path to your state dictonary as the default and an image, for example __sheep.jpg__ in the same folder, you can simply run the following which saves the segmented image under __sheep.png__.

```
python Demo.py sheep
```

Otherwise, you can directly define its path and the folder where to save it interactively in the terminal.

```
python Demo.py sheep --AimDir='Myfolder/' --PathToPretrainedModel='/myroot/my_state_dictonary.pth'
```
This saves the image in __Myfolder/sheep.png__

### LoadAndVisualizeTensor

This script takes a prediction tensor, for example from the training set without postprocessing, and visualizes it using an extern function. E.g. with a prediction batch saved in __/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions0.pth__ where you want the first prediction of the batch (index 0) saved as __segmentation.png__ you can use the following command.

```
python LoadAndVisualizeTensor.py '/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions0.pth' 0 'segmentation'
```

### LoadAndVisualizeTensorTest

This script allows you to visualize any of the 1449 Test images that are saved to the disk. Once you successfully ran _TestDataGenerator.py_, did the postprocessing steps, created a folder __TestImages__ and updated your paths inside the script, you can simply run in example:

```
python LoadAndVisualizeTensorTest.py 12
```

This will save the 13. test image with its ground truth and its predictions both normal and postprocessed under __TestImages/TestImage13.png__

### TestImages 

A folder containing some of the images visualized by _LoadAndVisualizeTensorTest.py_.

### Logs

A folder containing all the logs used to print the loss/accuracies of different test runs of the Deeplab V2 Pytorch model. Also a screenshot of our final evaluation without postprocessing.

## Shortcut - DemoGraphicalV3

For a simple use of segmentation we offer _DemoGraphical.py_. It is a simple to use program written by _Feussner_ to provide segmentation for single images. To list a few of the advantages:

* It uses the modern and offical Pytorch implementation of the Deeplab V3
* It is pretrained on the Coco Dataset (higher amount of images, include the Pascal Voc 2012 classes)
* At its first run it automatically downloads its pretrained models state dictionary
* It can be run on its own (and the legend.png) without the rest of the repository
* It can be run on both CPU and GPU
* It has a self explanatory graphical interface ideally for novices
* It uses tkinter for the graphical usage which is preinstalled with most python versions

The first run can take some time as the model (around 180MB) has to be downloaded. After that, you can simply run it, enter your Image name in the load Entry, click the load button and then the segment button. After a short time (depending on your computational power) the segmented picture will appear. If you entered a name in the Save path, it will additionally be automatically saved when segmenting. Loading and segmenting new images also takes less time once running, as the model and its weights stay in the random access memory and do not have to be loaded for every image separately.

## Prerequisites

For everything to run properly you have to install a few prerequisites, of which you might already have some due to prior projects. This list might not be complete and we advise the users.

* Python 3.6
* Pip or Conda
* [Pascal VOC 2012 devkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
* [AugmentationPascalVoc supplement](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
* [A pretrained Deeplab V2 model](https://drive.google.com/file/d/0BxhUwxvLPO7TVFJQU1dwbXhHdEk/view)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for GPU (needs NVIDIA graphic card)
* [Pytorch](https://pytorch.org/)
* Python libraries, e.g.
    * Matplotlib
    * Numpy
    * openCV
    * Scipy
    * etc.

The paths of the single files also have to be correct or changed in above mentioned scripts at the concerning Paths. Further, we highly recommend a UNIX based enviroment.


## Versioning

We use [Github](https://github.com/) for versioning our project.

## Authors

* __Feussner, Robert__ - _Providing/Adapting the Deeplab models, creating the Demo scripts, Generating the b11 data, Visualizing the results_
* __Haller von Hallerstein, Patrick__ - _Creating b12, coding the custom filter initialization for b12, generating the data of b12_
* __Mustea, Iulia-Otilia__ - _Creating b13-b14, generating data of b14, hyper-parameter tuning for b13-b14, evaluating results for b14 and b15_
* __Peisker, Tim__ - _Creating b15, computation module of b15, testing of b15_



## Acknowledgments

This code was partly adapted from the [Deeplab V2](https://github.com/speedinghzl/Pytorch-Deeplab) project from Zilong Huang.
