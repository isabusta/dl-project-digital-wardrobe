"""
Load images, preprocessing, prepare for training
"""

# Imports


# Define configurations for images
CONFIG = {

}


class ClothingDataset(Dataset):
    """
    Loads data from directory 
    """

    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.transform = self._get_transforms(mode)
        self.samples = [] # [(path, label_idx),....]
        self._scan_directory()

    def _scan_directory(self):
        # Get all subcategories 
        # Store all images in self.samples
        pass

    def _get_transforms(self, mode):
        # Resize, normalization 
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Open image 
        # return image, label
        pass


def get_dataloaders(data_dir, test_size = 0.2, random_seed = 42):
    """
    Create data loader for train, validation 
    Returns: train_loader, val_loader, class_names
    train_dataset = ClothingDataset(data_dir, mode='train')
    val_dataset = ClothingDataset(data_dir, mode='val')
    
    """
    pass