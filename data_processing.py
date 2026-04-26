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
from os import listdir


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


# Prepare Data for classification trainig
# First we need to crop the training pictures according to the boxes
# in the groundtruth images
# for example start: 1000, sample_size: 5000

def crop_images_by_groundtruth(annos_dir, img_dir, img_output_dir, annos_output_dir, start: int, sample_size):
  if not os.path.exists(output_dir):
        os.makedirs(img_output_dir)

  if not os.path.exists(annos_output_dir):
    os.makedirs(annos_output_dir)

  annos_list = os.listdir(annos_dir)
  annos_list.sort()
  annos_list = annos_list[start : start + sample_size]

  # get the path/directory
  for anno_name in annos_list:
    # check if the image ends with png
    if (anno_name.endswith(".json")):

      ann_path = os.path.join(
          annos_dir,
          anno_name
      )

      img_name = anno_name.replace(".json", "")
      img_path = os.path.join(
          img_dir,
          img_name + ".jpg"
      )

      if not os.path.exists(img_path):
        print(f"Image missing: {img_path}")
        continue

      #open image and save original size
      image = Image.open(img_path).convert("RGB")

      # ---------- ANNOTATION ----------
      with open(ann_path) as f:
          ann = json.load(f)

      for i, (key, item) in enumerate(ann.items()):

          if key.startswith("item"):
            box = item['bounding_box']
            # mask the picture accoridng to the box (mask with white pixel or black pixels?)
            """
            bounding_box: [x1,y1,x2,y2]，where x1 and y_1 represent the upper left point
            coordinate of bounding box, x_2 and y_2 represent the lower right point coordinate
            of bounding box. (width=x2-x1;height=y2-y1)
            """
            region = image.crop((box[0], box[1], box[2], box[3]))

            final_image = region

            # save (category)
            category = item.get('category_name', 'unknown').replace(" ", "_")
            final_image.save(os.path.join(img_output_dir, f"{img_name}_{category}_{i}.jpg"))

             # Write a json file in annos_output_dir with item
            anno_output_path = os.path.join(
                annos_output_dir,
                f"{img_name}_{category}_{i}.json"
            )

            with open(anno_output_path, "w") as out_f:
                json.dump({f"item{i}": item}, out_f, indent=4)

  print(f"Done. {len(annos_list)} Files edited.")


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



