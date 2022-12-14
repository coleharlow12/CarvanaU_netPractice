import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

# Used to save the model weights to a backup file
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
	print("=> Saving Checkpoint")
	torch.save(state, filename)

# Used to load the learned weights of a model
def load_checkpoint(checkpoint, model):
	print("=> Loading Checkpoint ")
	model.load_state_dict(checkpoint["state_dict"])

# Creates loaders to get training and testing data
def get_loaders(
	train_dir,
	train_maskdir,
	val_dir,
	val_maskdir,
	batch_size,
	train_transform,
	val_transform,
	num_workers=4,
	pin_memory=True,
):

	#Create Carvana dataset
	train_ds = CarvanaDataset(
		image_dir = train_dir,
		mask_dir = train_maskdir,
		transform = train_transform,
	)

	#Specifying the training dataset
	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=True,
	)

	val_ds = CarvanaDataset(
		image_dir=val_dir,
		mask_dir=val_maskdir,
		transform=val_transform,
	)

	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False,
	)

	return train_loader, val_loader
	
# Checks the accuracy at each iteration
def check_accuracy(loader, model, device="cuda"):
	num_correct = 0
	num_pixels = 0
	dice_score = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device).unsqueeze(1) #label doesn't have a channel
			preds = torch.sigmoid(model(x))
			preds = (preds > 0.5).float() # for binary this will be update for multi-class
			num_correct += (preds==y).sum()
			num_pixels += torch.numel(preds)
			dice_score += (2 * (preds * y).sum()) / (
				(preds + y).sum() + 1e-8
				)
            
	print(
		f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
	)
	print(f"Dice Score: {dice_score/len(loader)}")
	model.train()

# Saves the output of the U-NET as a image
def save_predictions_as_imgs(
	loader, model, folder="saved_images/", device="cuda"
):
	model.eval()
	for idx, (x,y) in enumerate(loader):
		x = x.to(device=device)
		with torch.no_grad():
			preds = torch.sigmoid(model(x))
			preds = (preds > 0.5).float()
		torchvision.utils.save_image(
			preds, f"{folder}/pred_{idx}.png"
		)
		torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

	model.train()

