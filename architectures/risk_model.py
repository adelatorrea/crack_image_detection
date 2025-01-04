import torch.nn as nn

class RiskDetectionModel(nn.Module):
	def __init__(self, IMG_SIZE):
		super(RiskDetectionModel, self).__init__()
		self.IMG_SIZE = IMG_SIZE
		self.model = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(256 * (self.IMG_SIZE // 8) * (self.IMG_SIZE // 8), 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.model(x)

