import torch
from torch import nn
import os
import preprocess_data, utils, create_model, engine
import argparse

IMG_SIZE = (224, 224)
PATCH_SIZE = 16
BATCH_SIZE = 32
MAX_LEN = int((IMG_SIZE[0] / PATCH_SIZE) * (IMG_SIZE[1] / PATCH_SIZE) + 1) # Number of pathces plus one for cls token
D = 768 # latent vector size
AUGMENTATION = False
NUM_HEADS = 12
NUM_BLOCKS = 12
EPOCHS = 5 # 30
LR = 0.001


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_size', default=IMG_SIZE, help="size of images, default=(224, 224)")
parser.add_argument('-p', '--patch_size', type=int, default=PATCH_SIZE, help="size of pathes, default=28")
parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help="number of images per batch, default=32")
parser.add_argument('-a', '--augmentation', type=bool, default=AUGMENTATION, help="Using or not augmentation, default=False")
parser.add_argument('-d', '--d_model', type=int, default=D, help="latent vector size, default=512")
parser.add_argument('-nh', '--num_heads', type=int, default=NUM_HEADS, help="number of heads in multi-head attention, default=8")
parser.add_argument('-bl', '--num_blocks', type=int, default=NUM_BLOCKS, help="number of transformer encoder blocks, default=8")
parser.add_argument('-e', '--epochs', type=int, default=EPOCHS, help="number of epochs, default=30")
parser.add_argument('-lr', '--learning_rate', type=float, default=LR, help="optimizer's learning rate, default=0.001")
#parser.add_argument('-s', '--save_dir', type=str,  help="directory to save model")
#parser.add_argument('-m', '--model_name', type=str,  help="model's name")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available else 'cpu'


torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_dir = preprocess_data.unzip_train_test_data(zip_file_dir='/content/data/train.zip',
                                                  data_path='/content/data',
                                                  train_or_test='train')
test_dir = preprocess_data.unzip_train_test_data(zip_file_dir='/content/data/test.zip',
                                                  data_path='/content/data',
                                                  train_or_test='test')

train_ds, test_ds, class_names = preprocess_data.create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    img_size=args.img_size,
                                                                    transform=args.augmentation,
                                                                    batch_size=args.batch_size,
                                                                    patch_size=args.patch_size)

model = create_model.ViT(batch_size=args.batch_size,
                          patch_size=args.patch_size,
                          max_len=MAX_LEN,
                          d_model=args.d_model,
                          num_heads=args.num_heads,
                          num_classes=len(class_names),
                          num_blocks=args.num_blocks).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

results = engine.train(model=model,
                      train_data=train_ds,
                      test_data=test_ds,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      device=device)


# utils.save_model(model=model,
#                   save_dir=args.save_dir,
#                   model_name=args.model_name)

