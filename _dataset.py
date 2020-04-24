import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Adverdataset(Dataset):

    def __init__(self, root, label, transform):
        self.root = root
        if label is None:
            self.label = None
        else:
            self.label = torch.from_numpy(label).long()
        self.transform = transform
        self.images = []
        for fname in sorted(os.listdir(root)):
            self.images.append(self.transform(Image.open(os.path.join(self.root, fname))))

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.label is None:
            return img
        else:       
            label = self.label[idx]
            return img, label

    def __len__(self):
        return len(self.images)