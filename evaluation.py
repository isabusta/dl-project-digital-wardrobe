from PIL import Image
import torch
import torchvision
from torchvision.ops import nms
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_best_clothing_box_1(predictions):
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



# Pipeline for detecting, cropping and classifying

categories = {0: "short sleeve top", 1: "long sleeve top", 2: "short sleeve outwear",
              3: "long sleeve outwear", 4: "vest", 5: "sling",
              6: "shorts", 7: "trousers", 8: "skirt",
              9:"sleeve dress", 10: "long sleeve dress", 11: "vest dress",
              12: "sling dress"}

def detect_crop_(image_path, detector, detector_transforms):
    # 1. load image
    orig_img = Image.open(image_path).convert("RGB")

    # 2. Transorm the image and shift to device
    img_tensor = detector_transforms(orig_img).to(device)

    # 3. Faster R-CNN Prediction
    print("start object detection...")

    # set model in evaluation mode
    detector.eval()

    with torch.inference_mode():
        predictions = detector([img_tensor])

    # 4. Find box with highest score (x1, y1, x2, y2)
    box = find_best_clothing_box(predictions)

    if box is None:
        print("No box found")
        return orig_img, None, "No Detection", 0.0

    x1, y1, x2, y2 = box
    print(f"Bounding Box found: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    # 5. Crop image accoring to found box (PIL uses x1, y1, x2, y2)
    cropped_img = orig_img.crop((x1, y1, x2, y2))

    # return the croped image
    return cropped_img

def predict(cropped_image, classifier, classifier_transforms):
  # first create tensor of cropped image
  cropped_image_tensor = classifier_transforms(cropped_image).unsqueeze(0).to(device)

  # Make classification prediction
  # 1. set model in evaluation model
  classifier.eval()

  with torch.inference_mode():
    # 2. compute y logits
    predictions = classifier(cropped_image_tensor)

  probs = torch.softmax(predictions, dim=1)
  prediction = torch.argmax(probs, dim=1).item()
  prediction_label = categories[prediction]

  return prediction, prediction_label
