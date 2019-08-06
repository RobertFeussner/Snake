# Vision Sub-system of the Snake Robot
## Semantic Segmentation Using Fully Convolutional Network and Markov Random Field

This project is a reimplememtation of the paper "[Deep Learning Markov Random Field for Semantic Segmentation]"(http://personal.ie.cuhk.edu.hk/~pluo/pdf/DPN_pami.pdf)(Liu et al) for providing a Vision System for the Snake Robot Project of the Technical university of Munich. We also used the paper "[Fully Convolutional Networks for Semantic Segmentation]"(https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)(Long et al) in order to get a deeper grasp of Fully Convulutional Networks.

## Single Programs Explained

In this section we explain our code and the code we used from other sources and give some examples of how to use it.

### Pytorch Deeplab

This folder contains a Github [Deeplab V2](https://github.com/speedinghzl/Pytorch-Deeplab) implementation based on the ResNet101. As it is written in Python 2, whereas we used Python 3, which is not backwards compatible, we had to change it a little. Also, we have done some adaptations to fit our project better.
We used this to train our Deeplab V2 model and do our evaluations on it. We got a final accuracy of around 73% on the test set, similiar to the result they have in their project.
Please make sure, that you need to download all the files  mentioned on their project site, as they are required for it to work and cannot be downloaded by git directly (files too big).

## DataGeneratorb11

Once you have trained your model and saved its state dictonary via the Pytorch Deeplab implementation, you can use the _DataGeneratorb11.py_ in order to run all the 10575 training images through the model and save its predictions, the groundtruth and the images in batches of a size of your choice. They can be used for postprocessing. It is important that you change __pathToTrainedModel__ inside the script. Also make sure that all the given paths exist in that way, if necessary change them. It should be mentioned that folder are not created automatically, they have to be created manually. Then, you can simply run it with the default parameters with the following command:

```
python DataGeneratorb11
```

If you want to change other parameters, e.g. need smaller batches, because the GPU does not have enough space, you can change it directly in the file itself (it is commented).

## b11Loader

This script itself (_b11Loader.py_) has no function on its own, but explains how to load the data produced by 








### TestImages 

A folder containing some of the images visualized by _LoadAndVisualizeTensorTest.py_.

### Logs

A folder containing all the logs used to print the loss/accuracies of different test runs of the Deeplab V2 Pytorch model. Also a screenshot of our final evalution without postprocessing.





These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
