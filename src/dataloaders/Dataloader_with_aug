from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import pytorch_lightning as pl
from typing import Tuple


class BloodClotDataset(Dataset):
    """
    BloodClotDataset: Custom dataset class for the Blood Clot dataset.

    Attributes:
        - dataframe (pd.DataFrame): Pandas DataFrame containing image file names and labels.
        - image_dir (str): Directory path where images are stored.
        - transform (callable, optional): Optional transform to be applied on a sample.

    Methods:
        - __len__(self) -> int: Return the number of samples in the dataset.
        - __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: Get a sample from the dataset.

    Example:
        dataset = BloodClotDataset(dataframe, image_dir, transform=transforms.Compose([...]))
    """

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image tensor and label tensor.
        """
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        label = torch.tensor(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


class BloodDataModule(pl.LightningDataModule):
    """
    BloodDataModule: LightningDataModule for the Blood Clot dataset.

    Attributes:
        - train_dataframe (pd.DataFrame): Pandas DataFrame containing training data.
        - image_dir (str): Directory path where images are stored.
        - batch_size (int): Batch size for dataloaders.
        - seed (int): Random seed for train-test split.
        - label_mapping (dict): Mapping of class labels to indices.
        - train_dataset (BloodClotDataset): Training dataset.
        - val_dataset (BloodClotDataset): Validation dataset.

    Methods:
        - setup(self, stage=None): Setup the data module and perform train-test split.
        - train_dataloader(self) -> DataLoader: Return the training dataloader.
        - val_dataloader(self) -> DataLoader: Return the validation dataloader.
        - transform(self, split: str) -> transforms.Compose: Return the appropriate data transform.

    Example:
        data_module = BloodDataModule(train_dataframe, image_dir, batch_size=16, seed=19)
    """

    def __init__(self, train_dataframe, image_dir, batch_size=16, seed=19):
        super(BloodDataModule, self).__init__()
        self.train_dataframe = train_dataframe
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.seed = seed
        self.label_mapping = {}
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """
        Setup the data module and perform train-test split.
        """
        train_data, val_data = train_test_split(self.train_dataframe, test_size=0.2, random_state=self.seed)

        self.label_mapping = {label: idx for idx, label in enumerate(train_data['label'].unique())}

        train_data['label'] = train_data['label'].map(self.label_mapping)
        val_data['label'] = val_data['label'].map(self.label_mapping)

        self.train_dataset = BloodClotDataset(train_data, self.image_dir, transform=self.transform("train"))
        self.val_dataset = BloodClotDataset(val_data, self.image_dir, transform=self.transform("val"))

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def transform(self, split: str) -> transforms.Compose:
        """
        Return the appropriate data transform.

        Args:
            split (str): Split type ('train' or 'val').

        Returns:
            transforms.Compose: Data transform.
        """
        if split == "train":
            # Data augmentation transforms for training
            return transforms.Compose([
                transforms.RandomAffine(0, scale=(0.8, 1.2), translate=(0.1, 0.1)),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
            ])
        elif split == "val":
            return transforms.Compose([
                transforms.ToTensor(),
            ])
