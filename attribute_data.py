"""
Data loading for DeepFashion fine-grained attribute recognition.

"""

from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image


# Attribute metadata
# Attribute type -> list of (attribute_name, global_index_in_26dim_vector)
ATTRIBUTES_BY_TYPE = {
    "texture": [("floral", 0), ("graphic", 1), ("striped", 2), ("embroidered", 3),
                ("pleated", 4), ("solid", 5), ("lattice", 6)],
    "sleeve": [("long_sleeve", 7), ("short_sleeve", 8), ("sleeveless", 9)],
    "length": [("maxi_length", 10), ("mini_length", 11), ("no_dress", 12)],
    "neckline": [("crew_neckline", 13), ("v_neckline", 14), ("square_neckline", 15),
                 ("no_neckline", 16)],
    "fabric": [("denim", 17), ("chiffon", 18), ("cotton", 19), ("leather", 20),
               ("faux", 21), ("knit", 22)],
    "fit": [("tight", 23), ("loose", 24), ("conventional", 25)],
}

TYPE_ORDER = ["texture", "sleeve", "length", "neckline", "fabric", "fit"]
NUM_CLASSES_PER_TYPE = {t: len(ATTRIBUTES_BY_TYPE[t]) for t in TYPE_ORDER}



# Helpers
def attr_path_to_partition_path(attr_path: str) -> str:
    """Convert attribute-file path format to partition-file path format.

    img_highres_subset/Striped_Denim_Shorts-img_00000037.jpg
        ->  img/Striped_Denim_Shorts/img_00000037.jpg
    """
    assert attr_path.startswith("img_highres_subset/"), f"Unexpected prefix: {attr_path}"
    stripped = attr_path[len("img_highres_subset/"):]
    last_hyphen = stripped.rfind('-')
    assert last_hyphen != -1, f"No hyphen found in: {attr_path}"
    return "img/" + stripped[:last_hyphen] + '/' + stripped[last_hyphen + 1:]


def labels_26dim_to_class_indices(labels_26: list[int]) -> dict[str, int]:
    """Convert a 26-dim binary vector into 6 class indices (one per type).

    Example:
        [0,0,1,0,0,0,0, 0,0,1, 0,0,1, 0,0,0,1, 0,0,1,0,0,0, 0,0,1]
        -> {"texture": 2, "sleeve": 2, "length": 2,
            "neckline": 3, "fabric": 2, "fit": 2}
    """
    result = {}
    for type_name in TYPE_ORDER:
        # Pull out the sub-vector for this type using the global indices
        global_indices = [idx for (_, idx) in ATTRIBUTES_BY_TYPE[type_name]]
        sub_vector = [labels_26[i] for i in global_indices]
        # Within-type class index = position of the single positive in the sub-vector
        assert sum(sub_vector) == 1, (
            f"Expected exactly one positive in {type_name} group, got {sub_vector}"
        )
        result[type_name] = sub_vector.index(1)
    return result

# Dataset class
class AttributeDataset(Dataset):
    """DeepFashion fine-grained attribute dataset.

    Args:
        data_root: Path to the folder containing the `img/` directory
                   and the annotation .txt files.
        split: One of "train", "val", "test".
        transform: Optional torchvision transform applied to the PIL image
                   before it's returned. Should produce a tensor.

    Returns from __getitem__:
        (image_tensor, labels_dict)
        image_tensor: shape [3, H, W] (after transform)
        labels_dict:  {"texture": int, "sleeve": int, "length": int,
                       "neckline": int, "fabric": int, "fit": int}
    """

    def __init__(self, data_root: str | Path, split: str, transform=None):
        assert split in {"train", "val", "test"}, f"Unknown split: {split}"
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # Load and join the two annotation files ONCE at init.
        # Store only what we need per sample: (relative_path, class_indices_dict).
        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[str, dict[str, int]]]:
        # Parse partition file into a path -> split lookup
        part_file = self.data_root / "list_eval_partition.txt"
        with open(part_file) as f:
            plines = f.readlines()
        # plines[0] = count, plines[1] = header, plines[2:] = data
        part_lookup = {}
        for line in plines[2:]:
            parts = line.strip().split()
            part_lookup[parts[0]] = parts[1]

        # Parse attribute file, joining on split as we go
        attr_file = self.data_root / "fine_list_attr_img.txt"
        with open(attr_file) as f:
            lines = f.readlines()

        samples = []
        for line in lines[2:]:
            parts = line.strip().split()
            attr_path = parts[0]
            labels_26 = [int(x) for x in parts[1:]]

            normalized_path = attr_path_to_partition_path(attr_path)
            sample_split = part_lookup.get(normalized_path)

            if sample_split != self.split:
                continue

            class_indices = labels_26dim_to_class_indices(labels_26)
            samples.append((normalized_path, class_indices))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, class_indices = self.samples[idx]

        # Load the image
        full_path = self.data_root / rel_path
        image = Image.open(full_path).convert("RGB")

        # Apply transforms (resize, normalize, to tensor, etc.)
        if self.transform is not None:
            image = self.transform(image)

        # Convert the dict of ints into a dict of tensors
        # (CrossEntropyLoss expects a LongTensor target, not a Python int)
        labels = {k: torch.tensor(v, dtype=torch.long) for k, v in class_indices.items()}

        return image, labels


# Transform pipelines

from torchvision import transforms

# ImageNet normalization stats — required when using any ImageNet-pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int = 224):
    """Training transform: resize + light augmentation + ImageNet normalization.

    Augmentations chosen conservatively for clothing attribute recognition:
    - HorizontalFlip: safe (garments are left-right symmetric for these attributes)
    - ColorJitter: improves robustness to lighting
    - No rotation/vertical flip/random crop: could change the ground-truth label
      (e.g., cropping off sleeves would make a "long_sleeve" label wrong)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224):
    """Evaluation transform: deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# DataLoader factory


from torch.utils.data import DataLoader


def build_dataloaders(
        data_root: str | Path,
        batch_size: int = 32,
        image_size: int = 224,
        num_workers: int = 0,
        pin_memory: bool = False,
):
    """Construct train/val/test DataLoaders.

    Args:
        data_root: Path to the DeepFashion attribute dataset folder.
        batch_size: Samples per batch. 32 is a reasonable default;
                    increase if you have GPU memory headroom.
        image_size: Side length of the (square) resized images.
        num_workers: Parallel data-loading processes. Use 0 on macOS,
                     2-4 on Linux/Colab.
        pin_memory: Set True when training on GPU for faster transfers.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    train_tf = build_train_transform(image_size)
    eval_tf = build_eval_transform(image_size)

    datasets = {
        "train": AttributeDataset(data_root, split="train", transform=train_tf),
        "val": AttributeDataset(data_root, split="val", transform=eval_tf),
        "test": AttributeDataset(data_root, split="test", transform=eval_tf),
    }

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory),
    }

    return loaders


if __name__ == "__main__":
    DATA_ROOT = "/Users/isabellebustamante/DeepFashion_attribute_recognition"

    # Dataset-level sanity check (same as before)
    for split in ["train", "val", "test"]:
        ds = AttributeDataset(DATA_ROOT, split=split, transform=None)
        print(f"{split}: {len(ds)} samples")

    # DataLoader-level sanity check
    print("\n--- DataLoader sanity check ---")
    loaders = build_dataloaders(DATA_ROOT, batch_size=32, num_workers=0)

    for split_name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"\n{split_name}:")
        print(f"  image batch shape: {images.shape}")
        print(f"  image dtype:       {images.dtype}")
        print(f"  image range:       [{images.min():.3f}, {images.max():.3f}]")
        print(f"  labels keys:       {list(labels.keys())}")
        print(f"  labels['texture'] shape: {labels['texture'].shape}")
        print(f"  labels['texture'] dtype: {labels['texture'].dtype}")
        print(f"  labels['texture'] first 5: {labels['texture'][:5].tolist()}")