from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .datasets import NPZDataset
import os
from .STL10_transforms import *
mean = {
    'mnist': (0.1307,),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4865, 0.4409),
    'stl10': (0.4467, 0.4398, 0.4066),
}

std = {
    'mnist': (0.3081,),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'stl10': (0.2603, 0.2566, 0.2713),
}


#################################################
##################### STL-10 ####################
#################################################
def stl10_train_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def stl10_test_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, split='test', transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader



#################################################
############ STL-10 Pert #####################
#################################################
def stl10_pert_test_loader(batch_size, root, download=True):
    transform = transforms.Compose([
    ## Uncomment the perturbation that you want to test upon
    # MotionBlur(size=2),
    # ClippedZoom(zoom_factor=2),
    # Brightness(c=1),
    # ShotNoise(n=100),
    # ZoomBlur(scale=2),
    # SnowNoise(radius=10),
    # Translate(trans=50),
    # Rotate (angle=20),
    # Tilt(angle=20),
    # Scale (scale=10),
    # Speckle (scale=1),
    # GaussianBlur(sigma=1),
    # Spatter(sigma=1),
    # Shear (shear=1),
    # Occlusion(),
    # Wave_transform(),
    # transforms.ColorJitter(brightness=0,saturation=2.8,contrast=0,hue=0), #selects uniformly b/w (0,saturation)
    # gaussian_blur(),
    # add_snow(),
    # elastic_transform_class(),
    rotate_scale(),
    transforms.ToTensor(),
    # AddGaussianNoise(std=1),
    transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, split='test', transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader

def stl10_pert_train_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        ## Uncomment the perturbation that you want to train (in the form of data augmentation) upon
        # transforms.AugMix(),
        # elastic_transform_class(),
        # Occlusion(),
         # Wave_transform(),
        # transforms.ColorJitter(brightness=0,saturation=2.8,contrast=0,hue=0), #selects uniformly b/w (0,saturation)
        # gaussian_blur(),
        transforms.ToTensor(),
        # AddGaussianNoise(std=1),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader
