import os
import torch
import torchvision
from torch import nn

from resnet_50_v2 import create_resnet_50_v2_model
from attribute_model_efficientnetB0 import create_attribute_efficientnet_model
from pipeline import Pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.exists('/content/drive/MyDrive/'):
    CKPT_DIR = '/content/drive/MyDrive/Deepfashion2/checkpoints'
else:
    CKPT_DIR = 'checkpoints'

DETECTOR_PATH   = f'{CKPT_DIR}/model_3_resnet50_v2adamW.pth'
CLASSIFIER_PATH = f'{CKPT_DIR}/efficient_net_B1_fine_tuned.pth'
ATTR_PATH       = f'{CKPT_DIR}/exp_weighted_loss_30ep_best.pth'


def load_pipeline(debug=False, eval_mode=False) -> Pipeline:
    # detection model
    detector = create_resnet_50_v2_model(device)
    detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))

    # classifier
    classifier = torchvision.models.efficientnet_b1(weights=None)
    classifier.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(1280, 13)
    )
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))

    # attribute model
    attr_model = create_attribute_efficientnet_model(device)
    attr_ckpt  = torch.load(ATTR_PATH, map_location=device)
    attr_model.load_state_dict(attr_ckpt['model_state_dict'])

    pipeline = Pipeline(
        obj_detector=detector,
        classifier=classifier,
        attr_model=attr_model,
        debug=debug,
        eval_mode=eval_mode,
    )

    print(f'Pipeline ready. (device: {device})')
    return pipeline


if __name__ == '__main__':
    pipeline = load_pipeline()
