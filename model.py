import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# Used to do horizontal convolution steps in the U-net.
# Its called DoubleConv because those horizontal steps all perform two convolutions
class DoubleConv(nn.Module): 
	# Runs when class is intilized
	def __init__(self,in_channels,out_channels):
		# This class inherits from the nn.Module class which has its own intialization.
		# The super keyword allows for both the nn.Module init and our custom init to run
		# If I didn't include the super it would just run my init
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels,out_channels, kernel_size = 3,
					  stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),

			nn.Conv2d(out_channels,out_channels, kernel_size = 3,
					  stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self,x):
		return self.conv(x)


class UNET(nn.Module): 
	# This defines all the modules we will use but doesn't necessarily link them in the correct order
	def __init__(
		self, in_channels=3,out_channels=1,features=[64,128,256,512],
	):
		super(UNET, self).__init__()
		self.ups = nn.ModuleList()		#For the upsteps, acts like a normal list but is properly registered
		self.downs = nn.ModuleList()	#For the down steps
		self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

		# Down sampling part of UNET
		for feature in features:
			self.downs.append(DoubleConv(in_channels,feature)) #Add layer to the module list
			in_channels = feature


		# Up part of UNET
		for feature in reversed(features):
			self.ups.append(
				nn.ConvTranspose2d(
					feature*2,feature, kernel_size=2, stride = 2,
					)
				)
			self.ups.append(DoubleConv(feature*2,feature))

		#Implements bottom most part of the U-net
		self.bottleneck = DoubleConv(features[-1],features[-1]*2)
		#Final convolution
		self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


	def forward(self, x):
	 	skip_connections = []

	 	#Downward steps
	 	for down in self.downs:
	 		x = down(x)
	 		skip_connections.append(x)
	 		x = self.pool(x)


 		x = self.bottleneck(x)
 		skip_connections = skip_connections[::-1] #Reverses the list skip_connections

 		# We do up step and then double convolution hence the two for two steps (ex range(0,6,2) = 0 2 4)
 		for idx in range(0, len(self.ups), 2):
 			x = self.ups[idx](x) # We are doing the 2d transpose convolution
 			sing_connect = skip_connections[idx//2] #The two is because of the 2 steps. We don't have a skip convolution at every step only after up 

 			# its possible the maxpooling step causes the skip_connections to have different size than x which would throw an error
 			# The code found here prevents that from happening by resizing the image
	 		if x.shape != sing_connect.shape:
	 			x = TF.resize(x, size=sing_connect.shape[2:]) #Resizes to the correct Height and Width

 			concat_skip = torch.cat((sing_connect,x),dim=1) #Concatenate along the channel dimensions
 			x = self.ups[idx+1](concat_skip)

 		# Performs the final convolution step
 		return self.final_conv(x)


def test():
	x = torch.randn((3,1,161,161))
	model = UNET(in_channels=1,out_channels=1)
	preds = model(x)
	print(preds.shape)
	print(x.shape)
	assert preds.shape == x.shape

if __name__ == "__main__":
	test() 