import torch
import matplotlib.pyplot as plt
from PIL import Image

from attribute_data import TYPE_ORDER


"""
Takes a cropped clothing image.
Returns the predicted class index for each attribute type
(texture, sleeve, length, neckline, fabric, fit).
"""
def predict(cropped_image, model, transformer, device):
    # 1. Transform image
    image_tensor = transformer(cropped_image).unsqueeze(0).to(device)

    # 2. Set model to eval mode and run prediction
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor)

    # 3. Return predicted class index per attribute
    return {t: logits[t].argmax(dim=1).item() for t in TYPE_ORDER}


"""
Same as predict() but returns the full softmax probability distribution
for each attribute type, so you can inspect scores for all classes.
"""
def predict_probs(cropped_image, model, transformer, device):
    # 1. Transform image
    image_tensor = transformer(cropped_image).unsqueeze(0).to(device)

    # 2. Set model to eval mode and run prediction
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor)

    # 3. Return softmax probabilities per attribute
    return {t: torch.softmax(logits[t], dim=1)[0].tolist() for t in TYPE_ORDER}


"""
Displays the image on the left and the predicted attribute classes on the right.
predictions: dict returned by predict()
"""
def show_predictions(image_path, predictions):
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(8, 4))

    # 1. Show image
    ax_img.imshow(Image.open(image_path).convert("RGB"))
    ax_img.axis("off")

    # 2. Show predicted attributes as text
    ax_text.axis("off")
    text = "\n\n".join(f"{t.upper()}\nclass {predictions[t]}" for t in TYPE_ORDER)
    ax_text.text(0.1, 0.95, text, va="top", fontsize=12, family="monospace",
                 transform=ax_text.transAxes)

    plt.tight_layout()
    plt.show()
