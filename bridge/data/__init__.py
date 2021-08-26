import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, LSUN
from .cacheloader import CacheLoader
from .celeba import CelebA
#from datasets.ffhq import FFHQ
#from datasets.stackedmnist import Stacked_MNIST


def get_dataloader(d_config, data_init, data_folder, bs, training=False):
    """
    Args:
        d_config: data configuration
        data_init: whether to initialize sampling from images
        data_folder: data folder
        bs: dataset batch size
        training: whether the dataloader are used for training. If False, sampling is assumed
    Returns:
        training dataloader (and test dataloader if training)
    """
    kwargs = {'batch_size': bs, 'shuffle': True, 'num_workers': d_config.num_workers, 'drop_last': True}
    if training:
        dataset, testset = get_dataset(d_config, data_folder)
        dataloader = DataLoader(dataset, **kwargs)
        testloader = DataLoader(testset, **kwargs)
        return dataloader, testloader

    if data_init:
        dataset, _ = get_dataset(d_config, data_folder)
        return DataLoader(dataset, **kwargs)
    else:
        return None


def get_dataset(d_config, data_folder):
    cmp = lambda x: transforms.Compose([*x])

    if d_config.dataset == 'CIFAR10':

        train_transform = [transforms.Resize(d_config.image_size), transforms.ToTensor()]
        test_transform = [transforms.Resize(d_config.image_size), transforms.ToTensor()]
        if d_config.random_flip:
            train_transform.insert(1, transforms.RandomHorizontalFlip())

        path = os.path.join(data_folder, 'CIFAR10')
        dataset = CIFAR10(path, train=True, download=True, transform=cmp(train_transform))
        test_dataset = CIFAR10(path, train=False, download=True, transform=cmp(test_transform))

    elif d_config.dataset == 'CELEBA':

        train_transform = [transforms.CenterCrop(140), transforms.Resize(d_config.image_size), transforms.ToTensor()]
        test_transform = [transforms.CenterCrop(140), transforms.Resize(d_config.image_size), transforms.ToTensor()]
        if d_config.random_flip:
            train_transform.insert(2, transforms.RandomHorizontalFlip())

        path = os.path.join(data_folder, 'celeba')
        dataset = CelebA(path, split='train', transform=cmp(train_transform), download=True)
        test_dataset = CelebA(path, split='test', transform=cmp(test_transform), download=True)

    # elif d_config.dataset == 'Stacked_MNIST':

    #     dataset = Stacked_MNIST(root=os.path.join(data_folder, 'stackedmnist_train'), load=False,
    #                             source_root=data_folder, train=True)
    #     test_dataset = Stacked_MNIST(root=os.path.join(data_folder, 'stackedmnist_test'), load=False,
    #                                  source_root=data_folder, train=False)

    elif d_config.dataset == 'LSUN':

        ims = d_config.image_size
        train_transform = [transforms.Resize(ims), transforms.CenterCrop(ims), transforms.ToTensor()]
        test_transform = [transforms.Resize(ims), transforms.CenterCrop(ims), transforms.ToTensor()]
        if d_config.random_flip:
            train_transform.insert(2, transforms.RandomHorizontalFlip())

        path = data_folder
        dataset = LSUN(path, classes=[d_config.category + "_train"], transform=cmp(train_transform))
        test_dataset = LSUN(path, classes=[d_config.category + "_val"], transform=cmp(test_transform))

    # elif d_config.dataset == "FFHQ":

    #     train_transform = [transforms.ToTensor()]
    #     test_transform = [transforms.ToTensor()]
    #     if d_config.random_flip:
    #         train_transform.insert(0, transforms.RandomHorizontalFlip())

    #     path = os.path.join(data_folder, 'FFHQ')
    #     dataset = FFHQ(path, transform=train_transform, resolution=d_config.image_size)
    #     test_dataset = FFHQ(path, transform=test_transform, resolution=d_config.image_size)

    #     num_items = len(dataset)
    #     indices = list(range(num_items))
    #     random_state = np.random.get_state()
    #     np.random.seed(2019)
    #     np.random.shuffle(indices)
    #     np.random.set_state(random_state)
    #     train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
    #     dataset = Subset(dataset, train_indices)
    #     test_dataset = Subset(test_dataset, test_indices)

    else:
        raise ValueError("Dataset [" + d_config.dataset + "] not configured.")

    return dataset, test_dataset


def logit_transform(image: torch.Tensor, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(d_config, X):
    if d_config.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    elif d_config.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if d_config.rescaled:
        X = 2 * X - 1.
    elif d_config.logit_transform:
        X = logit_transform(X)

    return X


def inverse_data_transform(d_config, X):

    if d_config.logit_transform:
        X = torch.sigmoid(X)
    elif d_config.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)