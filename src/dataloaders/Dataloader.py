from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset,random_split
from torchvision import transforms


IMG_SIZE = 224
BATCH_SIZE = 16
SPLIT_RATIO = 0.8

## Convert data from  labels to integers
classes = ['CE','LAA']
class_to_int = {classes[i] : i for i in range(len(classes))}


class CustomDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = class_to_int[self.dataframe.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label


image_shape = (IMG_SIZE, IMG_SIZE)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(image_shape),
    transforms.ToTensor(),
])
# Replace DataFrame with your data frame which contains names of images in data files and their corresponding labels
train_dataset = CustomDataset(df_train_splitted, '/kaggle/input/strip-ai-new-data/output/train/tiles2', transform=transform)

train_size = int(SPLIT_RATIO * len(df_train_splitted))
val_size = len(df_train_splitted) - train_size
# Split the data
train_dataset, validation_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoader instances
batch_size = BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)