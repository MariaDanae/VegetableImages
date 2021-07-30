# Implementation of Augmentor and DenseNet on Vegetable Images

## Overview
This is an Image Classification solution using Colab notebooks for classifying vegetable images. These images were taken from Kaggle `kritikseth/fruit-and-vegetable-image-recognition`, augmented using Augmentor `https://github.com/mdbloice/Augmentor` and modeled using DenseNet121. The accuracy of the validation set was ~89% and ~91% for the test set.

## Getting Started
To run this solution, simply follow these steps:
1) Copy the VegetableImages.ipynb file and open as a Colab notebook in your Google Drive.
2) Create a Kaggle account to obtain your own username and password which are saved in a file called `kaggle.json` in your Google Drive under folder path `/content/gdrive/My Drive/Kaggle`
3) Before running the VegetableImages.ipyn file, please change the directory locations according to your own folder structure in Google Drive.

## How the application works
### Connect to Google Drive
We will need to connect to Google Drive to be able to access the `kaggle.json` saved in the `Getting Started` section and to save files done in later steps.
```python
from google.colab import drive
drive.mount('/content/gdrive')
%cd /content/gdrive/My Drive/Kaggle
```
### Create Image training, validation and test datasets from Kaggle
After looking through some kaggle datasets using `!kaggle datasets list -s vegetable`, the kaggle dataset choosen was `fruit-and-vegetable-image-recognition`. The reason being is that it was relatively small compared to some other datasets and everything is being done in GoogleDrive; meaning space is limited.

```python
!kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition
!unzip \*.zip  && rm *.zip

DIR = '/content/gdrive/My Drive/Kaggle'
train_folder_names = [entry for entry in os.listdir(f'{DIR}/train')]
test_folder_names = [entry for entry in os.listdir(f'{DIR}/test')]
```
Small analysis of Kaggle images:
- 36 classes
- Approximately 100 images per class
- Manual cleaning required: Removed a few outliers (Apple earpods, Ginger women, Paprika anime, fruit smoothies)
- - Images include an assortment of cartoon images, sliced or cooked vegetables, vegetable alone or grouped, fruit on tree, etc.

Remove corrupted images (ie. images that are badle encoded do not contain “JFIF” in header) (referrence: https://keras.io/examples/vision/image_classification_from_scratch/)
```python
count = 0
nb_removed = 0
for folder_name in train_folder_names:
    folder_path = os.path.join(DIR, 'train', folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            count += 1
            print(count,":\t",fpath, ':\t', is_jfif)
        finally:
            fobj.close()

        if not is_jfif:
            nb_removed += 1
            os.remove(fpath)

print(f"Total {count} images")
print(f"Removed {nb_removed} images")
```

### Alter images using Augmentor
- Different types of image augmentation exist:
    - Tensorflow.keras.preprocessing.image's ImageDataGenerator
    - GitHub: 
      - Augmentor: focus on geometric
      - ImgAug: run on multiple CPU cores
      - Albumentations: attempts to cover all types of augmentation
      - AutoAugment and DeepAugment: searches best augmentation policies

### Create datasets based on these augmented images
### Use DenseNet model
### Results: Confusion Matrix and accuracy ~91%

## Main References
https://medium.com/analytics-vidhya/how-to-fetch-kaggle-datasets-into-google-colab-ea682569851a
https://neptune.ai/blog/google-colab-dealing-with-files
https://keras.io/examples/vision/image_classification_from_scratch/
https://neptune.ai/blog/data-augmentation-in-python
https://github.com/mdbloice/Augmentor
https://augmentor.readthedocs.io/en/master/code.html
https://keras.io/api/preprocessing/image/
https://www.kaggle.com/mauricioasperti/cats-vs-dogs-image-classification/notebook
https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow
https://keras.io/api/applications/densenet/
https://keras.io/api/layers/core_layers/dense/
