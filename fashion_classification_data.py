from torch.utils.data import Dataset
import json
from PIL import Image

# Datset for the classification task

class ClassificationTrainDataset(Dataset):
  def __init__(self, root, transform = None):
    self.root = root
    self.transform = transform

    self.img_dir = os.path.join(root, "image")
    self.ann_dir = os.path.join(root, "annos")
    self.images = sorted(os.listdir(self.img_dir))

  def __len__(self):
    return len(self.images)

  # returns one sample of Data
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

      new_w, new_h = 224, 224

      ratio_w = new_w / orig_w
      ratio_h = new_h / orig_h

      # ---------- ANNOTATION ----------
      with open(ann_path) as f:
          ann = json.load(f)

      labels = []
      for key, item in ann.items():

          if key.startswith("item"):

            labels.append(item["category_id"])


      if len(labels) == 0:
        raise ValueError(f"No labels found in {ann_path}")
      label = torch.tensor(labels[0] - 1, dtype=torch.long)

      if self.transform:
          image = self.transform(image)

      return image, label