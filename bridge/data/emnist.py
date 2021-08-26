import os, shutil
import urllib
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import save_image


class EMNIST(Dataset):
    def __init__(self, root="./dataset", load=True, source_root=None, imageSize=28,
                 train=True, num_channels=3, device='cpu'):  # load=True means loading the dataset from existed files.
        super(EMNIST, self).__init__()
        self.data = torch.load(os.path.join(root, "data.pt"))
        self.targets = torch.load(os.path.join(root, "targets.pt"))
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        return img, targets

    def __len__(self):
        return len(self.targets)
