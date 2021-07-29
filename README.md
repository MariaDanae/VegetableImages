# Implementation of Augmentor and DenseNet on Vegetable Images

## Overview
This is an Image Classification solution using Colab notebooks for classifying vegetable images found on Kaggle using Augmentor and DenseNe.

## How to use

1) Copy the VegetableImages.ipynb file and open as a Colab notebook in your Google Drive.
2) Create a Kaggle account to obtain your own username and password which are saved in a file called "kaggle.json" and should be saved in your Google Drive along with the VegetableImages.ipynb downloaded in the previous step
3) Open the VegetableImages.ipyn file and change the directory locations according to your own folder structure.

## Running the solution
The following steps were taken to achieve this solution:
1) Obtain kaggle zipped file of vegetable images and unzip them
- Selected Kaggle repo was “kritikseth/fruit-and-vegetable-image-recognition” due to its small size compared to a few others (unzipped in Colab with limited space)
- Analysis of Kaggle images:
    - Remove corrupted images (ie. that do not contain “JFIF” in header)
    - 36 classes
    - 100 images per class
    - Manual cleaning required: Removed a few outliers (Apple earpods, Ginger women, Paprika anime, fruit smoothies)
    - Images include an assortment of cartoon images, sliced or cooked vegetables, vegetable alone or grouped, fruit on tree, etc.
2) Alter images using Augmentor
- Different types of image augmentation exist:
    - Tensorflow.keras.preprocessing.image's ImageDataGenerator
    - GitHub: 
      - Augmentor: focus on geometric
      - ImgAug: run on multiple CPU cores
      - Albumentations: attempts to cover all types of augmentation
      - AutoAugment and DeepAugment: searches best augmentation policies

3) Create datasets based on these augmented images
4) Use DenseNet model
5) Results: Confusion Matrix and accuracy ~91%

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
