"""
Create Pytorch DataLoader's for image classification
"""
import zipfile
from pathlib import Path
import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import einops
from einops import rearrange

# try:
#   import einops
# except:
#   print("Couldn't find einops... installing it")
#   !pip install einops
#   import einops
# from einops import rearrange

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
PATCH_SIZE = 28

def unzip_train_test_data(zip_file_dir:str,
                          data_path:str,
                          train_or_test:str):
  """
  Args:
    zip_file_dir: str - Directory of zip file (in format: 'your_dir/filename.zip')
    data_path: str - Directory to unzip files
    train_or_test: str - Name of dataset: train or test

  Return:
    A directory of unzipped data

  Usage example:
    directory = unzip_train_test_data(zip_file_dir='your_directory/zip_file_name.zip',
                                      data_path='/content/data',
                                      train_or_test='train')
  """
  data_path = Path(data_path)
  data_dir = data_path / train_or_test

  # Create directory
  if data_dir.is_dir():
    print(f'{data_dir} directory already exist')
  else:
    print(f'Create {data_dir} directory')
    data_dir.mkdir(parents=True, exist_ok=True)

  with zipfile.ZipFile(zip_file_dir, 'r') as zip_ref:
    print("Unzipping data...")
    zip_ref.extractall(data_dir)
  os.remove(zip_file_dir)

  return str(data_dir)

def create_patches(img:torch.Tensor,
                   patch_size:int=PATCH_SIZE):
  """
  Create patches from image for vision transformer and flatten them

  Args:
    img - image tensor
    patch_size -  size of patches

  Returns:
    torch.Tensor with shape (1, number_of pathes, patch_size * patch_size * color_channel)
     with flatten pathes

  Usage example:
    flatten_patches = create_patches(img=img,
                                     patch_size=28)
  """
  #img = img.unsqueeze(0) # for batch
  flatten_pathes = rearrange(img, 'c (h h1) (w w1)  -> (h w) (h1 w1 c)', h1=patch_size, w1=patch_size)

  return flatten_pathes


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       img_size,
                       transform: bool=False,
                       batch_size: int=BATCH_SIZE,
                       patch_size: int=PATCH_SIZE,):

  """
  Creates a training and testing DataLoaders

  Takes in a training and testing directories and turns them into PyTorch Datasets and then into a PyTorch DataLoaders

  Args:
    train_dir: str - directory of training data
    test_dir: str - directory of testing data
    transform: bool - True means using data augmentation (RandomPerspective(),
                                                          RandomRotation(),
                                                          RandomHorizontalFlip(),
                                                          RandomVerticalFlip(),
                                                          Resize(224),
                                                          ToTensor()),
                      False only Resize(224) and converting images into tensors with ToTensor()
    batch_size: int=BATCH_SIZE - Number of samples per batch
    patch_size: int=PATCH_SIZE - Number of pathes


  Returns:
    A tuple of (train_ds, test_ds, class_names)
    Where class_name a list of the target classes

  Usage example:
    train_ds, test_ds, class_names = create_dataloaders(train_dir='path/to/train_dir',
                                                        test_dir='path/to/test_dir',
                                                        img_size=(224, 224),
                                                        transform=True,
                                                        batch_size=32,
                                                        patch_size=28)

  """

  # Create transforms
  augmentation_transform = transforms.Compose([
      transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
      transforms.RandomRotation(degrees=(0, 180)),
      transforms.RandomHorizontalFlip(p=0.2),
      transforms.RandomVerticalFlip(p=0.2),
      transforms.Resize(size=img_size),
      transforms.ToTensor()#,
     #transforms.Lambda(lambda x: create_patches(x, patch_size=patch_size))
  ])
  simple_transform = transforms.Compose([
      transforms.Resize(size=img_size),
      transforms.ToTensor()#,
      #transforms.Lambda(lambda x: create_patches(x, patch_size=patch_size))
  ])

  if transform:
    train_transform = augmentation_transform
  else:
    train_transform = simple_transform

  # Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(root=train_dir,
                                    transform = train_transform)
  test_data = datasets.ImageFolder(root=test_dir,
                                  transform = simple_transform)

  class_names = train_data.classes

  # Create a datasets with DataLoader
  train_ds = DataLoader(train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)
  test_ds = DataLoader(test_data,
                       batch_size=batch_size,
                       shuffle=False,
                       drop_last=True)

  return train_ds, test_ds, class_names
