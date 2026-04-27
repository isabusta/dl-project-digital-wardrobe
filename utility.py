import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

"""
train_image, target = train_dataset[20]
plot_image(train_image, target)
"""
def plot_image(tensor_image, target):

  # sample_img is Tensor [3, 224, 224]
  img_np = tensor_image.permute(1, 2, 0).cpu().numpy()
  h, w = img_np.shape[:2] # h=224, w=224

  fig, ax = plt.subplots(figsize=(5, 5))

  # from normalization to pixel
  for box in target['boxes']:
    xmin, ymin, xmax, ymax = box.tolist()

    x = xmin * w
    y= ymin * h
    width = (xmax - xmin) * w
    height = (ymax - ymin) * h

    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


  ax.imshow(img_np)
  plt.show()

def plot_image_1(tensor_image, target):
    img_np = tensor_image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))

    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box.tolist()
   
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    ax.imshow(img_np)
    plt.show()

def collate_fn(batch):
  # batch = list of tuples: (image, boxes, labels)
  images = torch.stack([b[0] for b in batch])
  targets = [b[1] for b in batch]
  return images, targets

def mask_boxes(image, boxes):
    """
    Masks an image by setting all pixels outside the predicted bounding boxes to white.
    Returns:
        The modified image tensor with pixels outside boxes set to white.
    """
    # Create a boolean mask with same spatial dimensions as the image
    mask = torch.zeros(image.shape[1:], dtype=torch.bool)

    # Mark all pixels inside each bounding box as True
    for box in boxes:
        x1, y1, x2, y2 = box.int()
        mask[y1:y2, x1:x2] = True

    # Set all pixels outside the boxes to white
    image[:, ~mask] = 1.0

    return image

def save_model(model, model_name: str, path: str):
  # create Model directory
  MODEL_PATH = Path(path)
  MODEL_PATH.mkdir(parents=True, exist_ok=True)
  MODEL_NAME = model_name
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
  torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def load_model(model: torch.nn.Module, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load parameters
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    # Shift to device
    model.to(device)

    return model
