import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
from torchvision import utils, transforms, datasets

class ConvertPIL(object):
    def __init__(self):
        self.pil_image = transforms.ToPILImage()

    def __call__(self, sample):
        image = sample['image']
        image = self.pil_image(image)
        return {'image': image, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        # h, w = image.shape[:2]
        h, w = image.size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        resize = transforms.Resize((new_h, new_w))
        img = resize(image)
        return {'image': img, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}
    
class RandHorizFlip(object):
    def __init__(self, p):
        self.horiz_flip = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, sample):
        image = sample['image']
        image = self.horiz_flip(image)
        return {'image': image, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}

class RandVertFlip(object):
    def __init__(self, p):
        self.vert_flip = transforms.RandomVerticalFlip(p=p)

    def __call__(self, sample):
        image = sample['image']
        image = self.vert_flip(image)
        return {'image': image, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}

class RandCrop(object):
     def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
        self.crop = transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
        
     def __call__(self, sample):
        image = sample['image']
        image = self.crop(image)
        return {'image': image, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        # print(image.shape)
        return {'image': image, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}