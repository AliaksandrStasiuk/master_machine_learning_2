from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class DogCatDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 1]
        image_file = self.data.iloc[idx, 0]

        # print(image_file)

        image  = Image.open(image_file)

        if self.transforms:
            image = self.transforms(np.uint8(image))
        else:
            image = image.astype(np.float32)

        return image, label