import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from attribute_data import TYPE_ORDER, NUM_CLASSES_PER_TYPE, build_eval_transform

# Optional: map class indices to human-readable names.
# Update these once you know your label ordering from list_attr_cloth.txt.
LABEL_NAMES = {
    "texture":  ["floral", "graphic", "striped", "embroidered", "pleated", "solid", "lattice"],
    "sleeve":   ["long", "short", "sleeveless"],
    "length":   ["maxi", "mini", "no_dress"],  # Changed: maxi first, then mini, then no_dress
    "neckline": ["crew", "v-neck", "square", "no_neckline"],  # Changed: crew first, not round
    "fabric":   ["denim", "chiffon", "cotton", "leather", "faux", "knit"],  # Changed: denim first, faux not fur
    "fit":      ["tight", "loose", "conventional"],  # Changed: tight first, conventional not fitted
}


def predict_image(model: torch.nn.Module, image_path: str, device: str) -> dict:
    """
    Run a single image through the attribute model.

    Returns a dict keyed by attribute type:
        {
          "texture": {"class": 5, "label": "solid", "confidence": 0.82, "probs": [...]},
          ...
        }
    """
    model.eval()
    transform = build_eval_transform()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    with torch.inference_mode():
        logits = model(tensor)   # dict of {type: [1, num_classes]}

    predictions = {}
    for attr_type in TYPE_ORDER:
        probs = torch.softmax(logits[attr_type], dim=1)[0]   # [num_classes]
        confidence, class_idx = probs.max(dim=0)
        class_idx = class_idx.item()
        names = LABEL_NAMES.get(attr_type, [])
        label = names[class_idx] if class_idx < len(names) else f"class_{class_idx}"
        predictions[attr_type] = {
            "class": class_idx,
            "label": label,
            "confidence": confidence.item(),
            "probs": probs.tolist(),
        }
    return predictions


def print_predictions(predictions: dict) -> None:
    """Print predictions with confidence bars to the terminal."""
    print(f"\n{'Attribute':<12}  {'Prediction':<14}  {'Conf':>6}  Bar")
    print("-" * 52)
    for attr_type in TYPE_ORDER:
        p = predictions[attr_type]
        bar_len = int(p["confidence"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{attr_type:<12}  {p['label']:<14}  {p['confidence']*100:5.1f}%  {bar}")


def show_predictions(image_path: str, predictions: dict) -> None:
    """Display the image with predicted attribute labels and confidence scores."""
    img = Image.open(image_path).convert("RGB")

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 6),
                                          gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: image with text overlay
    ax_img.imshow(img)
    ax_img.axis("off")
    label_lines = "\n".join(
        f"{t:<10} {predictions[t]['label']}  ({predictions[t]['confidence']*100:.0f}%)"
        for t in TYPE_ORDER
    )
    ax_img.set_title(label_lines, fontsize=9, family="monospace", loc="left", pad=8)

    # Right: horizontal confidence bars per attribute
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    y_positions = range(len(TYPE_ORDER))

    for i, attr_type in enumerate(TYPE_ORDER):
        p = predictions[attr_type]
        ax_bar.barh(i, p[""], color=colors[i], alpha=0.8)
        ax_bar.text(p["confidence"] + 0.01, i,
                    f"{p['label']}  {p['confidence']*100:.0f}%",
                    va="center", fontsize=9)

    ax_bar.set_yticks(list(y_positions))
    ax_bar.set_yticklabels(TYPE_ORDER)
    ax_bar.set_xlim(0, 1.35)
    ax_bar.set_xlabel("Softmax probs")
    ax_bar.set_title("Attribute Predictions")
    ax_bar.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.show()


def show_batch_predictions(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device: str,
                           n: int = 8) -> None:
    """Show predictions vs ground truth for the first n images in a dataloader."""
    model.eval()
    images, labels = next(iter(dataloader))
    images = images[:n]

    with torch.inference_mode():
        logits = model(images.to(device))

    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 8))
    axes = axes.flatten()

    inv_mean = [-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    inv_std  = [1 / s for s in [0.229, 0.224, 0.225]]

    import torchvision.transforms.functional as F

    for idx in range(n):
        ax = axes[idx]
        img_t = images[idx].cpu().clone()
        img_t = F.normalize(img_t, mean=inv_mean, std=inv_std).clamp(0, 1)
        ax.imshow(img_t.permute(1, 2, 0).numpy())
        ax.axis("off")

        lines = []
        for attr_type in TYPE_ORDER:
            probs = torch.softmax(logits[attr_type][idx], dim=0)
            pred_idx = probs.argmax().item()
            conf = probs[pred_idx].item()
            true_idx = labels[attr_type][idx].item()
            correct = "✓" if pred_idx == true_idx else "✗"
            names = LABEL_NAMES.get(attr_type, [])
            pred_label = names[pred_idx] if pred_idx < len(names) else str(pred_idx)
            lines.append(f"{correct} {attr_type[:3]}: {pred_label} {conf*100:.0f}%")

        ax.set_title("\n".join(lines), fontsize=6.5, family="monospace")

    plt.suptitle("batches", fontsize=10)
    plt.tight_layout()
    plt.show()
