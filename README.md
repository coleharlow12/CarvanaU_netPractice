# PytorchCarvanaU-net
The purpose of this project is to gain additional experience using U-nets for image segmentation problems. The structure of this U-net is copied from the winner of a Kaggle competition using U-nets to do image segmentation for a carvana competition. The initial file structure is based on the video "PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby" by Aladdin Persson. 

The video and dataset can be found through the links below at the time of writing
[U-net Tutorial](https://www.youtube.com/watch?v=IHq1t7NxS8k&t=826s&ab_channel=AladdinPersson)
[Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)

The aim of the carvana competition was to remove the background from car images so that the car could be displayed in front of any background. To do this there is both the raw .jpg images of the car with the background and mask.gif files showing what areas are car and what areas are background. Example images are shown below

![Raw jpg](ImgForREADME/0cdf5b5d0ce1_01.jpg)
![Mask](ImgForREADME/0cdf5b5d0ce1_01_mask.gif)

# Usage
In order to use the following script it is necessary to first download the kaggle dataset. For this script since it is just for practice I only downloaded train.zip and train_masks.zip and place their contents into /data/train_images and /data/train_masks. I then run the script createVal.py to create a validation set which doesn't reference the test data. Within the script createVal.py the paramater "NumVal" can be adjusted to change the number of files that are moved from the training dataset to the validation dataset. 

The script is currently configured to detect and utilize gpus. In any case after all dependencies have been downloaded the file train.py can be excuted to train the algorithm. I have also included files which are useful if interested in running this library on paperspace gradient or a similar cloud computing 

# Paperspace Instructions
To use this library on paperspace gradient these are the steps that I followed. First create a paperspace gradient Kernel Session. Then 

# File Structure
- Train.py: This is the file that should be run in order to train the algorithm. At the start of this file there are a number of parameters that can be set such as the learning rate, batch size, number of epochs, number of workers etc. There is also an option to resize the images which I do since I have limited computational resources even when using paperspace. Within train.py there are two functions train_fn which performs one epoch of training and main which performs the entire training algorithm

- dataset.py: This file creates the class CarvanaDataset which has the functions __len__ and __getitem__ which allow this dataset type to be used with the Torch Dataloader to load sets of images in at random for each batch. If specified a transform can also be applied to the data as it is loaded. 

- utils.py: This file is used to create a number of functions. Examples include functions to save and load checkpoints, functions to create datasets and their associated dataloaders, functions to check the accuracy and dice score of the predictions and finally a function to save the predicted masks as images for later viewing.

- model.py: This function contains the structure of the UNET itself. The structure is based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" with some slight modificatons. The primary modification is the replacement of how padding is done here versus the paper. The file first defines a class DoubleConv whiich handles the two convolutions at each step of the UNET. The file then defines the class UNET. UNET has two functions __init__ which is where all the layers of the network are defined, and forward which defines the output of the current network given input x. A function test is also defined to ensure the model is initialized correctly
