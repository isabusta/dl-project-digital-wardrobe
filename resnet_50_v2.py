import torchvision
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def create_resnet_50_v2_model(device):

    # Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

    # Initialize the model
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    num_classes = 14  # 13 classes + background class

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device=device)
    return model
