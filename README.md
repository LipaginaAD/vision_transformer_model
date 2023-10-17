# vision_transformer_model

CV model to classify animals with method from paper  "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"

## Data
dataset from Kaggle - https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset

Animal Species Classification has 15 classes of animals with 2000 images per class in training set and 100-200 images per class in testing set

## Usage

```
!python vit_model/train.py
```
You can also change the following settings:

--img_size - image size,  default=(224, 224) 

--epochs - number of epochs, default=5

--batch_size - number of samples per batch, default=32

--patch_size - saize of patches, default=16

--augmentation - Using or not augmentation, default=False

--d_model - latent vector size, default=768

--num_heads - number of heads in multi-head attention, default=8

--num_blocks - number of transformer encoder blocks, default=8

--learning_rate - optimizer's learning rate, default=0.001
