import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	check_accuracy,
	save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 240 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

# Does one epoch of training
def train_fn(loader, model, optimizer, loss_fn, scaler):
	loop = tqdm(loader) #Creates progress bar

	for batch_idx, (data, targets) in enumerate(loop):
		data = data.to(device=DEVICE)
		targets = targets.float().unsqueeze(1).to(device=DEVICE)

		# Forward
		with torch.cuda.amp.autocast():
			predictions = model(data)
			loss = loss_fn(predictions, targets)

		# Backward
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		# Update tqdm loop
		loop.set_postfix(loss=loss.item())


# Train the model
def main():
	train_transform = A.Compose(
		[
			A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
			A.Rotate(limit=35, p=1.0),
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.1),
			A.Normalize(
				mean=[0.0, 0.0, 0.0],
				std=[1.0, 1.0, 1.0],
				max_pixel_value=255.0
			),
			ToTensorV2(),
		],

	)

	val_transforms = A.Compose(
		[
			A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
			A.Normalize(
				mean=[0.0, 0.0, 0.0],
				std =[1.0, 1.0, 1.0],
				max_pixel_value = 255.0,
			),
			ToTensorV2(),
		],
	)

	model = UNET(in_channels=3, out_channels=1).to(DEVICE)
	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_loader, val_loader = get_loaders(
		TRAIN_IMG_DIR,
		TRAIN_MASK_DIR,
		VAL_IMG_DIR,
		VAL_MASK_DIR,
		BATCH_SIZE,
		train_transform,
		val_transforms
	)

	#Scalers are used to handle the problem of underflowing gradients
	#Underflowing gradients occur when gradients are too small to take into account due to
	#Computer storage limitations. This is different than vanishing gradient problem
	scaler = torch.cuda.amp.GradScaler()

	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss_fn, scaler)

		# Save model
		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		save_checkpoint(checkpoint)

		# Check accuracy
		check_accuracy(val_loader, model, device=DEVICE)

		# Print some examples to a folder
		save_predictions_as_imgs(
			val_loader, model, folder="saved_images/",  device=DEVICE
		)

if __name__ == "__main__":
	main()