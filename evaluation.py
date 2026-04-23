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
