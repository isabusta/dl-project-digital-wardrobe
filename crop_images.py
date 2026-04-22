"""
Step 1 — Crop images to ground truth bounding boxes.

Reads:
  list_eval_partition.txt     → split label per image
  attributes/fine_list_attr_img.txt → image paths + 26-dim attribute vector
  bbox/list_bbox.txt          → bounding box per image

Writes:
  crops/train/  *.jpg
  crops/val/    *.jpg
  crops/test/   *.jpg
  crops/labels.csv  (filename, split, texture, sleeve, length, neckline, fabric, fit)
"""

import csv
from pathlib import Path
from PIL import Image

DATA_ROOT  = Path("/Users/isabellebustamante/DeepFashion_attribute_recognition")
CROPS_ROOT = DATA_ROOT / "crops"

# Attribute groups in order — mirrors attribute_data.py
ATTRIBUTES_BY_TYPE = {
    "texture":  [0, 1, 2, 3, 4, 5, 6],
    "sleeve":   [7, 8, 9],
    "length":   [10, 11, 12],
    "neckline": [13, 14, 15, 16],
    "fabric":   [17, 18, 19, 20, 21, 22],
    "fit":      [23, 24, 25],
}
TYPE_ORDER = ["texture", "sleeve", "length", "neckline", "fabric", "fit"]


# ── helpers ───────────────────────────────────────────────────────────────────

def attr_path_to_standard(attr_path: str) -> str:
    """img_highres_subset/Item-img_N.jpg  →  img/Item/img_N.jpg"""
    fname = attr_path.split("/", 1)[-1]
    idx = fname.rfind("-img_")
    if idx == -1:
        return attr_path
    return f"img/{fname[:idx]}/img_{fname[idx + 5:]}"


def std_path_to_crop_filename(std_path: str) -> str:
    """img/Striped_Denim_Shorts/img_00000037.jpg  →  Striped_Denim_Shorts_img_00000037.jpg"""
    parts = std_path.split("/")   # ['img', 'Item_Name', 'img_NNNNN.jpg']
    return f"{parts[1]}_{parts[2]}"


def labels_to_class_indices(labels_26: list[int]) -> dict[str, int]:
    """Convert 26-dim binary vector to one class index per attribute group."""
    result = {}
    for type_name in TYPE_ORDER:
        indices = ATTRIBUTES_BY_TYPE[type_name]
        sub = [labels_26[i] for i in indices]
        result[type_name] = sub.index(1)
    return result


# ── load annotation files ─────────────────────────────────────────────────────

def load_partition(path: Path) -> dict[str, str]:
    partition = {}
    for raw in path.read_text().splitlines()[2:]:
        parts = raw.split()
        if len(parts) == 2:
            partition[parts[0]] = parts[1]
    return partition


def load_bbox_index(path: Path) -> dict[str, tuple[int, int, int, int]]:
    index = {}
    for raw in path.read_text().splitlines()[2:]:
        parts = raw.split()
        if len(parts) == 5:
            index[parts[0]] = (int(parts[1]), int(parts[2]),
                               int(parts[3]), int(parts[4]))
    return index


def load_attr_records(path: Path) -> list[tuple[str, list[int]]]:
    records = []
    for raw in path.read_text().splitlines()[2:]:
        parts = raw.split()
        if parts:
            records.append((parts[0], [int(x) for x in parts[1:]]))
    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading annotation files…")
    partition  = load_partition(DATA_ROOT / "list_eval_partition.txt")
    bbox_index = load_bbox_index(DATA_ROOT / "bbox" / "list_bbox.txt")
    attr_records = load_attr_records(
        DATA_ROOT / "attributes" / "fine_list_attr_img.txt"
    )
    print(f"  {len(partition)} partition entries")
    print(f"  {len(bbox_index)} bbox entries")
    print(f"  {len(attr_records)} attribute records\n")

    # Create output directories
    for split in ("train", "val", "test"):
        (CROPS_ROOT / split).mkdir(parents=True, exist_ok=True)

    # Open CSV for writing
    csv_path = CROPS_ROOT / "labels.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "split"] + TYPE_ORDER)

    counts   = {"train": 0, "val": 0, "test": 0}
    skipped  = 0

    print("Cropping images…")
    for i, (attr_path, labels_26) in enumerate(attr_records):
        std_path = attr_path_to_standard(attr_path)

        # Look up split and bbox
        split = partition.get(std_path)
        bbox  = bbox_index.get(std_path)

        if split is None or bbox is None:
            skipped += 1
            continue

        # Open and crop
        img_file = DATA_ROOT / std_path
        if not img_file.exists():
            skipped += 1
            continue

        image = Image.open(img_file).convert("RGB")
        x1, y1, x2, y2 = bbox

        # Guard against degenerate boxes
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        crop = image.crop((x1, y1, x2, y2))

        # Save crop
        crop_filename = std_path_to_crop_filename(std_path)
        crop.save(CROPS_ROOT / split / crop_filename, quality=95)

        # Write label row
        class_indices = labels_to_class_indices(labels_26)
        writer.writerow(
            [crop_filename, split] + [class_indices[t] for t in TYPE_ORDER]
        )

        counts[split] += 1

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:>6} / {len(attr_records)} processed…")

    csv_file.close()

    print(f"\nDone.")
    print(f"  train : {counts['train']} crops")
    print(f"  val   : {counts['val']} crops")
    print(f"  test  : {counts['test']} crops")
    print(f"  skipped: {skipped}")
    print(f"  labels saved to: {csv_path}")


if __name__ == "__main__":
    main()
