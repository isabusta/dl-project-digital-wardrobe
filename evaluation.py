from PIL import Image
import torchvision

"""
This Method takes a model and an image path as input
Then the model is used
to make a object detection.
img_path: path to the image
returns: box x1, y1, x2, y2
transformer: transfomer to transform the data suitable for the model
"""
def make_box_prediction(model, img_path: str, transformer, device):
  # open image and save original size
  image = torchvision.io.read_image(img_path).type(torch.float32)

  # transform the image
  image = transformer.transform(image).unsqueeze(0)

  # put the image on the device
  image.to(device)

  with torch.inference_mode():
    # set model in evaluation mode
    model.eval()

    # compute the prediction
    output = model(image)

  return output

def find_best_clothing_box(predictions):
    """
    We are looking for the Box with the highest Score,
    which may contain fashion items. The COCO-Modell does not have fashion item categories.
    We therefore take the person Box oder the Box with the highest confidence.
    """
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    if len(boxes) == 0:
      return None

    # COCO Label for 'person' is 1. Often fashion is on persons.
    # Wir suchen zuerst nach Personen.
    person_indices = (labels == 1).nonzero(as_tuple=True)[0]

    if len(person_indices) > 0:
      # Take the person-Box with the highest Score
      best_person_idx = person_indices[scores[person_indices].argmax()]
      return boxes[best_person_idx].cpu().numpy()

    # if no person was found, just take the beste Box
    best_box_idx = scores.argmax()
    return boxes[best_box_idx].cpu().numpy()


