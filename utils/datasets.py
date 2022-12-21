import os

import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image


class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        super().__init__()
        print('PATH=', path)
        path = os.path.expanduser(path)
        self.transforms = transforms

        npz_file = np.load(path)
        self.data = npz_file['data']
        self.targets = npz_file['targets']

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if len (x.shape)>3:
            x = x.reshape(3,96,96)
            x = x.astype(np.uint8)
        x = to_pil_image(x)
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__
        return s
