"""
Load images, preprocessing, prepare for training
We use the DeepFashion2 dataset which has COCO-format annotations.
Dataset structure:
- Training images: train/image
- Training annotations: train/annos
- Validation images: validation/image
- Validation annotations: validation/annos

Each image has a unique six-digit name (e.g. 000001.jpg) with a corresponding
annotation file (e.g. 000001.json). We convert these annotations to COCO format
using deepfashion2_to_coco.py before loading them here.

Dataset: https://github.com/switchablenorms/DeepFashion2

"""

# Imports
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torchvision.transforms as T

# Define configurations for images
# automatically detect if running on Colab or local Mac
if os.path.exists('/content/'):
    # running on Google Colab
    CONFIG = {
        "val_images": "/content/validation/validation/image/",
        "val_annos": "/content/drive/MyDrive/Deepfashion2/deepfashion2_val.json",
        "train_images": "/content/train/train/image/",
        "train_annos": "/content/drive/MyDrive/Deepfashion2/deepfashion2_train.json"
    }
else:
    # running locally, update paths to match your setup
    CONFIG = {
        "val_images": "/Users/isabellebustamante/deepfashion2/validation/image/",
        "val_annos": "/Users/isabellebustamante/deepfashion2/json_for_validation/deepfashion2_val.json",
        "train_images": "",
        "train_annos": ""
    }

class ClothingDataset(Dataset):
    """
    Loads data from directory
    Each sample returns one image and its annotations
    (bounding boxes, category labels, pixel masks).
    """

    def __init__(self, root_dir, annos_json, mode='train'):
        self.root_dir = root_dir
        self.coco = COCO(annos_json) #Loads the entire COCO annotation JSON file into memory and builds the index. This replaces the need for manually scanning the directory for image files and annotation files.
        self.transform = self._get_transforms(mode)
        self.samples = list(self.coco.imgs.keys())# gets the list of all image ids from the COCO index
        # self._scan_directory() not needed, COCO() already indexes all images

    #def _scan_directory(self):
        # Get all subcategories 
        # Store all images in self.samples


    def _get_transforms(self, mode):
        # Resize, normalization
        # ToTensor converts the PIL image to a PyTorch tensor
        return T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Open image
        # get the image id
        img_id = self.samples[idx]

        # get the image filename from the COCO index
        img_info = self.coco.imgs[img_id]

        #build  the path and open the image, force RGB format
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # load annotations for this image, one image can have multiple garments, each is a different annotation
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # extract bounding boxes, category labels, pixel masks from each annotation
        boxes, labels, masks = [], [], []
        for ann in anns:
            #COCO stores box as [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])# convert to [x1, y1, x2, y2]
            labels.append(ann['category_id']) # category_id is an integer from 1-13 representing the garment type
            masks.append(self.coco.annToMask(ann))  # annToMask converts the polygon segmentation into a binary pixel mask, 1 = garment pixel, 0 = background pixel

        # return image, label
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.tensor(np.array(masks), dtype=torch.uint8),
            'image_id': torch.tensor([img_id]),
        }

        return self.transform(image), target


def get_dataloaders(data_dir=None, test_size = 0.2, random_seed = 42):
    """
    Create data loader for train, validation 
    Returns: train_loader, val_loader, class_names
    train_dataset = ClothingDataset(data_dir, mode='train')
    val_dataset = ClothingDataset(data_dir, mode='val')
    
    """
    # create the validation dataset using ClothingDataset class
    val_dataset = ClothingDataset(
        CONFIG['val_images'],  # folder with images
        CONFIG['val_annos'],  # COCO annotation file
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return val_loader

if __name__ == '__main__':
    val_dataset = ClothingDataset(
        CONFIG['val_images'],
        CONFIG['val_annos'],
        mode='val'
    )
    print(f"Validation images: {len(val_dataset)}")
    image, target = val_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")