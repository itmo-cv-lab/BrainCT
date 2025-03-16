"""Enhancement Dataset Module"""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EnhancementDataset(Dataset):
    """Class for Enhancement dataset"""

    def __init__(self, dataset_path, mode="train", transform=None):
        """Dataset Initialization"""

        assert mode == "train" or mode == "test"

        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.image_path = []

        source_path = os.path.join(self.dataset_path, self.mode, "source")
        target_path = os.path.join(self.dataset_path, self.mode, "target")

        names = sorted(os.listdir(source_path))
        self.paths = [
            (os.path.join(source_path, name), os.path.join(target_path, name))
            for name in names
        ]

    def __len__(self):
        """Length of the dataset"""

        return len(self.paths)

    def __getitem__(self, idx):
        """Get i-th portion of data"""

        source_path, target_path = self.paths[idx]

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.transform:
            source = self.transform(source)
            target = self.transform(target)

        return source, target


def get_transforms_enhancement(image_size=512):
    """Get image transformations"""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
