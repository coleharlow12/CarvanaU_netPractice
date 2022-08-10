import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# These datasets are defined as a "Map-Style" dataset and are comptabile with the pyTorch DataLoader
class CarvanaDataset(Dataset):
	def __init__(self,image_dir,mask_dir,transform=None):
		self.image_dir = image_dir		#image directory (independent vars)
		self.mask_dir = mask_dir		#mask directory (dependent vars)
		self.transform = transform		#Transform to apply to data
		posIms = os.listdir(image_dir)	#Lists all the files in the directory
		valIms = []						#Creates list to store name of all valid images
		for im in posIms:				#Loops through all the files
			if im.endswith(".jpg"):		#If the file has the .jpg extension then it is an image
				valIms.append(im)		# Add image to the list
		self.images = valIms 			#Saves list of available images

	# Returns the number of total images
	def __len__(self):
		return len(self.images)

	# Gets an image from the dataset and returns it to the loader
	def __getitem__(self,index):
		img_path = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg","_mask.gif"))
		image = np.array(Image.open(img_path).convert("RGB"))
		mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
		# 0.0, 255.0 for black and white respectively
	
		# Two binary values 
		mask[mask == 255.0] = 1.0

		# Applies a transform if one is specified
		if self.transform is not None:
			augmentations = self.transform(image=image, mask=mask)
			image = augmentations["image"]
			mask = augmentations["mask"]

		return image,mask	