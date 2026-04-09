"""

image_path = "/content/train/train/image/000021.jpg"
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

def collate_fn(batch):
  # batch = list of tuples: (image, boxes, labels)
  images = torch.stack([b[0] for b in batch])
  targets = [b[1] for b in batch]
  return images, targets


def save_model(model, model_name: str, path: str):
  # create Model directory
  MODEL_PATH = Path(path)
  MODEL_PATH.mkdir(parents=True, exist_ok=True)
  MODEL_NAME = model_name
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
  torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)