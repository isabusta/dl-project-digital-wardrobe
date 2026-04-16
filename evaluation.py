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
