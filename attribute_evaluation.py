import os
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
PILImage = Image
from sklearn.metrics import classification_report

from attribute_data import TYPE_ORDER, LABEL_NAMES, build_dataloaders
from attribute_model_efficientnetB0 import create_attribute_efficientnet_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.exists('/content/drive/MyDrive/'):
    CONFIG = {
        "crops_root":      "/content/crops",
        "checkpoint_path": "/content/drive/MyDrive/Deepfashion2/checkpoints/exp_weighted_loss_30ep_best.pth",
        "output_path":     "/content/drive/MyDrive/Deepfashion2/checkpoints/attribute_eval_results.json",
    }
else:
    CONFIG = {
        "crops_root":      "data/crops",
        "checkpoint_path": "attribute_efficientnet_best.pth",
        "output_path":     "attribute_eval_results.json",
    }


def load_model(checkpoint_path: str = None) -> torch.nn.Module:
    path  = checkpoint_path or CONFIG["checkpoint_path"]
    model = create_attribute_efficientnet_model(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']} | val acc: {ckpt['val_acc']:.4f}")
    return model


def evaluate(model: torch.nn.Module, split: str = "val",
             crops_root: str = None, save_path: str = None) -> dict:
    """
    Run evaluation on a given split (val or test).
    Prints per-class F1, precision, recall for each attribute head.
    Saves summary JSON if save_path is provided.
    """
    assert split in {"val", "test"}, "split must be 'val' or 'test'"

    loaders = build_dataloaders(
        crops_root=crops_root or CONFIG["crops_root"],
        batch_size=32,
        num_workers=2,
    )

    all_preds  = {t: [] for t in TYPE_ORDER}
    all_labels = {t: [] for t in TYPE_ORDER}

    model.eval()
    with torch.inference_mode():
         for images, y in loaders[split]:
            images  = images.to(device)
            outputs = model(images)
            for t in TYPE_ORDER:
                all_preds[t].extend(outputs[t].argmax(dim=1).cpu().tolist())
                all_labels[t].extend(y[t].tolist())

    results = {}
    for t in TYPE_ORDER:
        print(f"\n{'='*40}")
        print(f"  {t.upper()}")
        print(f"{'='*40}")
        report = classification_report(
            all_labels[t],
            all_preds[t],
            target_names=LABEL_NAMES[t],
            zero_division=0,
            output_dict=True,
        )
        print(classification_report(
            all_labels[t],
            all_preds[t],
            target_names=LABEL_NAMES[t],
            zero_division=0,
        ))
        results[t] = {
            "accuracy":    round(report["accuracy"], 4),
            "macro_f1":    round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "per_class":   {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"], 4),
                    "f1":        round(report[cls]["f1-score"], 4),
                    "support":   report[cls]["support"],
                }
                for cls in LABEL_NAMES[t]
            },
        }

    overall_acc = sum(r["accuracy"] for r in results.values()) / len(TYPE_ORDER)
    print(f"\n{'='*40}")
    print(f"  OVERALL mean accuracy: {overall_acc:.4f}")
    print(f"{'='*40}")

    summary = {"split": split, "overall_mean_accuracy": round(overall_acc, 4), "heads": results}

    out = save_path or CONFIG["output_path"]
    with open(out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nResults saved to: {out}")

    return summary


def predict_image(model: torch.nn.Module, image_path: str) -> dict:
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img    = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
    return {t: LABEL_NAMES[t][logits[t].argmax(dim=1).item()] for t in TYPE_ORDER}


def show_predictions(image_path: str, predictions: dict):
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(8, 4))
    ax_img.imshow(Image.open(image_path).convert("RGB"))
    ax_img.axis("off")
    ax_text.axis("off")
    text = "\n\n".join(f"{t.upper()}\n{predictions[t]}" for t in TYPE_ORDER)
    ax_text.text(0.1, 0.95, text, va="top", fontsize=12, family="monospace",
                 transform=ax_text.transAxes)
    plt.tight_layout()
    plt.show()


def show_attribute_showcase(model: torch.nn.Module, image_paths: list[str], device: str = None):
    """Show multiple crops in one figure, each with its attribute predictions."""
    from torchvision import transforms
    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    n   = len(image_paths)
    fig = plt.figure(figsize=(10, n * 2.8))
    gs  = plt.GridSpec(n, 2, width_ratios=[1, 2], hspace=0.55, wspace=0.15)

    model.eval()
    with torch.inference_mode():
        for i, path in enumerate(image_paths):
            img    = PILImage.open(path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(_device)
            logits = model(tensor)
            preds  = {t: LABEL_NAMES[t][logits[t].argmax(dim=1).item()] for t in TYPE_ORDER}

            ax_img = fig.add_subplot(gs[i, 0])
            ax_img.imshow(img)
            ax_img.axis("off")
            label = os.path.basename(path).split("_img_")[0].replace("_", " ")
            ax_img.set_title(label, fontsize=8, fontweight='bold')

            ax_txt = fig.add_subplot(gs[i, 1])
            ax_txt.axis("off")
            lines = [f"{t:<10} {preds[t]}" for t in TYPE_ORDER]
            ax_txt.text(0.02, 0.95, "\n".join(lines), va='top', fontsize=9,
                        family='monospace', transform=ax_txt.transAxes, linespacing=1.85)

    plt.suptitle("Attribute Predictions — Diverse Showcase", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = load_model()
    evaluate(model, split="test")
