# Digital Wardrobe Project

## Getting the data

Download the data from the shared Google Drive folder: https://drive.google.com/drive/folders/1C3nL9-XXMb_OBw8sebamyO9c704BE8_E. It contains:
- `train.zip` training images (191,961 images)
- `validation.zip` validation images (32,153 images)
- `deepfashion2_train.json` training annotations loaded by `data_processing.py` during model training
- `deepfashion2_val.json` validation annotations loaded by `data_processing.py` during model evaluation

### Setup Locally
1. Download `train.zip` and `validation.zip` from shared Drive https://drive.google.com/drive/folders/1C3nL9-XXMb_OBw8sebamyO9c704BE8_E
2. Unzip with password 2019Deepfashion2**
3. Update `CONFIG` in `data_processing.py` to your local paths:
```python
CONFIG = {
    "train_images": "/path/to/train/image/",
    "train_annos": "/path/to/deepfashion2_train.json",
    "val_images": "/path/to/validation/image/",
    "val_annos": "/path/to/deepfashion2_val.json"
}
```
4. Run `data_processing.py` to confirm everything loads correctly

### Setup in Google Colab for training
1. Mount Drive and clone repo:
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/isabelle-bustamante/DL_Project_Digital_Wardrobe.git
```
2. Install packages:
```python
!pip install pycocotools torch torchvision Pillow numpy
```
3. Unzip images to local storage:
```python
import zipfile

with zipfile.ZipFile('/content/drive/MyDrive/Deepfashion2/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/train/', pwd=b'2019Deepfashion2**')

with zipfile.ZipFile('/content/drive/MyDrive/Deepfashion2/validation.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/validation/', pwd=b'2019Deepfashion2**')
```
4. Update `CONFIG` in `data_processing.py`:
```python
CONFIG = {
    "train_images": "/content/train/train/image/",
    "train_annos": "/content/drive/MyDrive/Deepfashion2/deepfashion2_train.json",
    "val_images": "/content/validation/validation/image/",
    "val_annos": "/content/drive/MyDrive/Deepfashion2/deepfashion2_val.json"
}
```
5. Test the pipeline:
```python
!python DL_Project_Digital_Wardrobe/data_processing.py
```