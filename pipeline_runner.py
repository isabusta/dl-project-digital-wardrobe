from data_processing import get_dataloaders
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

class PipelineRunner:

    train_transform_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Standards
                             std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Image Transformer for classification prediction
    classification_prediction_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Standards
                             std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pass

    def run(self):
        # 1. Data
        train_loader, val_loader, class_names = get_dataloaders("data/raw/")
        # 2. Model
        # 3. Training
        # 4. Evaluation
        pass

