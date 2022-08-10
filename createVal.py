# The purpose of this script is to split the training dataset provided 
# Kaggle into a training and validation set. It does this by randomly
# selecting files and moving those files into a new folder

import os
import random
import numpy as np
import shutil

cwd = os.getcwd()
imDir = os.path.join(cwd,'data','train_images')
newImDir = os.path.join(cwd,'data','val_images')
maskDir = os.path.join(cwd,'data','train_masks')
newMaskDir = os.path.join(cwd,'data','val_masks')

files = os.listdir(imDir)

#Ensures all listed files are jpg
for file in files:
	if not file.endswith(".jpg"):
		files.remove(file)

#Checks the total number of files
NumFiles = len(files)
print("There are :",NumFiles,"files")

NumVal = 100 #The number of files to use in the validation set

fileInd = np.arange(0,NumFiles)
randInd = np.random.choice(fileInd,size=NumVal,replace=False)

listVal = []

# Creates
for randNum in randInd:
	imPath = os.path.join(imDir,files[randNum])
	newImPath = os.path.join(newImDir,files[randNum])

	maskPath = os.path.join(maskDir,os.path.splitext(files[randNum])[0]+'_mask.gif')
	newMaskPath = os.path.join(newMaskDir,os.path.splitext(files[randNum])[0]+'_mask.gif')

	shutil.move(imPath,newImPath)
	shutil.move(maskPath,newMaskPath)