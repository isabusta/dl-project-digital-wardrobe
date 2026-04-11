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
import torchvision.transforms as T
import json
# import numpy as np  # will be needed again for masks in Mask R-CNN extension

# Define configurations for images
# automatically detect if running on Colab
if os.path.exists('/content/drive/MyDrive/'):
    # running on Google Colab
    CONFIG = {
        "val_images": "/content/validation/validation/image/",
        "val_annos": "/content/drive/MyDrive/Deepfashion2/deepfashion2_val.json",
        "train_images": "/content/train/train/image/",
        "train_annos": "/content/drive/MyDrive/Deepfashion2/deepfashion2_train.json"
    }
else:
    # running locally — update these paths to match your machine
    CONFIG = {
        "val_images": "/Users/isabellebustamante/deepfashion2/validation/image/",
        "val_annos": "/Users/isabellebustamante/deepfashion2/json_for_validation/deepfashion2_val.json",
        "train_images": "",
        "train_annos": ""
    }


class ClothingDataset(Dataset):
    """
    Loads data from directory.
    Each sample returns one image and its category label.
    """

    def __init__(self, root_dir, annos_json, mode='train'):
        self.root_dir = root_dir
        self.coco = COCO(annos_json)  # loads the entire COCO annotation JSON file into memory and builds the index
        self.transform = self._get_transforms(mode)
        self.samples = list(self.coco.imgs.keys())  # list of all image IDs
        # self._scan_directory() COCO() already indexes all images

    #def _scan_directory(self):
        # Get all subcategories
        # Store all images in self.samples

    def _get_transforms(self, mode):
        # Resize, normalization
        return T.Compose([T.ToTensor()])

    def __len__(self):
        # returns total number of images in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Open image
        img_id = self.samples[idx]

        # get the image filename from the COCO index
        img_info = self.coco.imgs[img_id]

        # build the path and open the image, force RGB format
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # return image, label
        # taking the first garment's category as the image label
        # boxes, labels, masks = [], [], []
        # for ann in anns:
        #     x, y, w, h = ann['bbox']
        #     boxes.append([x, y, x+w, y+h])
        #     labels.append(ann['category_id'])
        #     masks.append(self.coco.annToMask(ann))
        label = anns[0]['category_id'] if anns else 0

        return self.transform(image), label


def get_dataloaders(data_dir=None, test_size=0.2, random_seed=42):
    val_dataset = ClothingDataset(
        CONFIG['val_images'],  # folder with images
        CONFIG['val_annos'],   # COCO annotation file
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # add train_loader here when training on Colab
    return val_loader




"""
create dataset via: 
train_set = FashionDataSet(/content/train)
"""
class ClothingDatasetResize(Dataset):
  def __init__(self, root, transform = None):
    self.root = root
    self.transform = transform

    self.img_dir = os.path.join(root, "image")
    self.ann_dir = os.path.join(root, "annos")
    self.images = sorted(os.listdir(self.img_dir))

  def __len__(self):
    return len(self.images)

  #returns one sample of Data
  def __getitem__(self, idx):
      img_name = self.images[idx]
      img_path = os.path.join(self.img_dir, img_name)

      ann_path = os.path.join(
          self.ann_dir,
          img_name.replace(".jpg", ".json")
      )

      # 1. open image and save original size
      image = Image.open(img_path).convert("RGB")
      orig_w, orig_h = image.size
      new_w, new_h = 2224, 224
      ratio_w = new_w / orig_w
      ratio_h = new_h / orig_h

      # ---------- ANNOTATION ----------
      with open(ann_path) as f:
          ann = json.load(f)

      boxes = []
      labels = []

        # DeepFashion2 JSON Struktur
      for key, item in ann.items():

          if key.startswith("item"):
            b = item['bounding_box']
            boxes.append([
                b[0] * ratio_w,
                b[1] * ratio_h,
                b[2] * ratio_w,
                b[3] * ratio_h
            ])
            labels.append(item["category_id"])
      target = {}
      target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
      target["labels"] = torch.tensor(labels, dtype=torch.long)

      if self.transform:
          image = self.transform(image)

      return image, target


if __name__ == '__main__':
    val_dataset = ClothingDataset(
        CONFIG['val_images'],
        CONFIG['val_annos'],
        mode='val'
    )
    print(f"Validation images: {len(val_dataset)}")
    image, label = val_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")



