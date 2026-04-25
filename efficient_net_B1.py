import torchvision
import torch
from torch.optim.lr_scheduler import ExponentialLR


def load_model(device):
  # load the efficient b1 net
  weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
  model_efficient_b1_0 = torchvision.models.efficientnet_b1(weights=weights).to(device)

  # freeze the base layers
  for param in model_efficient_b1_0.features.parameters():
    param.requires_grad = False 

  # update the classifier head of the model
  model_efficient_b1_0.classifier = nn.Sequential(
      nn.Dropout(0.2, inplace=True),
      nn.Linear(1280, out_features=13, bias = True)
  ).to(device)

 # Optimizer
def get_optimizer():
  optimizer = torch.optim.Adam(model_efficient_b1_0.parameters, lr=0.001, weight_decay=0.0005)
  return optimizer

def get_loss_fn():
  # Loss function
  loss_fn = nn.CrossEntropyLoss()
  return loss_fn

def get_scheduler(optimizer: torch.optim.Optimizer, gamma: float)
  # scheduler
  scheduler = ExponentialLR(optimizer, gamma=gamma)
  return scheduler
  
  
