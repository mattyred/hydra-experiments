import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from torchvision import transforms

class CIFAR10CDataset(Dataset):
    def __init__(self, data_path: Path, corruption_type: str, severity: int, transform=None):
        """
        severity: 1 to 5
        corruption_type: e.g., 'gaussian_noise', 'fog', etc.
        """
        images = np.load(data_path / f"{corruption_type}.npy")
        labels = np.load(data_path / "labels.npy")
        
        # Each corruption file has 50,000 images. 
        # Indices 0-10000: Severity 1, 10000-20000: Severity 2, etc.
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.data = images[start_idx:end_idx]
        self.targets = labels[start_idx:end_idx]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, target.astype(np.int64)

class CIFAR10CDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        corruption_type: str, 
        severity: int, 
        batch_size: int = 64, 
        num_workers: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = Path(data_dir)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def setup(self, stage=None):
        self.test_dataset = CIFAR10CDataset(
            self.data_path, 
            self.hparams.corruption_type, 
            self.hparams.severity, 
            transform=self.transforms
        )

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)