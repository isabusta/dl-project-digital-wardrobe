import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image


TYPE_ORDER = ["texture", "sleeve", "length", "neckline", "fabric", "fit"]
NUM_CLASSES_PER_TYPE = {"texture": 7, "sleeve": 3, "length": 3, "neckline": 4, "fabric": 6, "fit": 3}


class AttributeDataset(Dataset):
    def __init__(self, crops_root: str | Path, split: str, transform=None):
        assert split in {"train", "val", "test"}, f"Unknown split: {split!r}"
        self.crops_root = Path(crops_root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, dict[str, torch.Tensor]]]:
        csv_path = self.crops_root / "labels.csv"
        samples = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != self.split:
                    continue
                img_path = self.crops_root / self.split / row["filename"]
                labels = {
                    t: torch.tensor(int(row[t]), dtype=torch.long)
                    for t in TYPE_ORDER
                }
                samples.append((img_path, labels))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, labels


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataloaders(
    crops_root: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    train_tf = build_train_transform(image_size)
    eval_tf  = build_eval_transform(image_size)

    datasets = {
        "train": AttributeDataset(crops_root, "train", transform=train_tf),
        "val":   AttributeDataset(crops_root, "val",   transform=eval_tf),
        "test":  AttributeDataset(crops_root, "test",  transform=eval_tf),
    }

    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory),
        "val":   DataLoader(datasets["val"],   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
        "test":  DataLoader(datasets["test"],  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
    }




if __name__ == "__main__":
    CROPS_ROOT = "/Users/isabellebustamante/DeepFashion_attribute_recognition/crops"

    for split in ("train", "val", "test"):
        ds = AttributeDataset(CROPS_ROOT, split)
        print(f"{split:5s}: {len(ds)} samples")

    print("\n--- DataLoader batch check ---")
    loaders = build_dataloaders(CROPS_ROOT, batch_size=8, num_workers=0)
    for name, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f"{name:5s}: images {tuple(imgs.shape)}  "
              f"labels {{ {', '.join(f'{k}: {v.tolist()}' for k, v in labels.items())} }}")
