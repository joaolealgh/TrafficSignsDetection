import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import utils
class TrafficSignsDataset(Dataset):
    """
    Custom Dataset for the Traffic Sign Dataset - GTSRB
    """
    def __init__(self, annotations_file, root_dir, transform=None):
        """
        Arguments:
            annotations_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_file = pd.read_csv(annotations_file)
        self.img_labels = self.annotations_file['ClassId']
        self.img_path = self.annotations_file['Path']
        self.img_width = self.annotations_file['Width']
        self.img_height = self.annotations_file['Height']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_path.iloc[idx])
        image = io.imread(img_path)
        label = self.img_labels.iloc[idx]
        img_width = self.img_width.iloc[idx]
        img_height = self.img_height.iloc[idx]

        sample = {'image': image, 'label': label, 'img_width': img_width, 'img_height': img_height}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_classes(self):
        return self.img_labels.unique()


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


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
    
class CustomNormalize(object):
    """Normalize dataset."""
    def __init__(self, mean, std):
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image = sample['image']
        image = self.norm(image)
        return {'image': image, 'label': sample['label'], 'img_width': sample['img_width'], 'img_height': sample['img_height']}


def get_mean_std_custom_dataset(loader):
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)

	for _, sample in enumerate(loader):
		images = sample['image']
		b, c, h, w = images.shape
		nb_pixels = b * h * w
		sum_ = torch.sum(images, dim=[0, 2, 3])
		sum_of_square = torch.sum(images ** 2,
								dim=[0, 2, 3])
		fst_moment = (cnt * fst_moment + sum_) / (
					cnt + nb_pixels)
		snd_moment = (cnt * snd_moment + sum_of_square) / (
							cnt + nb_pixels)
		cnt += nb_pixels

	mean, std = fst_moment, torch.sqrt(
		snd_moment - fst_moment ** 2)    
        
	return mean,std


def calculate_mean_std_custom_dataset():
    data_transform = transforms.Compose([
            ConvertPIL(),
            Rescale((32, 32)),
            ToTensor()
    ])

    dataset = TrafficSignsDataset(annotations_file='../../dataset/GTSRB/Train.csv', 
                            root_dir='../../dataset/GTSRB',
                            transform=data_transform)

    dataset_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=64,
                                            shuffle=False,
                                            num_workers=4)

    mean, std = get_mean_std_custom_dataset(dataset_loader)
    print(mean, std)
    # TODO: Export mean and std to be able to skip this step everytime this is run
    return mean, std


def show_traffic_signs(sample_batched):
    """Show traffic signs for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


def plot_gtsrb_dataset_images(loader):
    for i_batch, sample_batched in enumerate(loader):
        # print(i_batch, sample_batched['image'].size())
        # observe 1st batch and stop.
        if i_batch == 1:
            plt.figure()
            show_traffic_signs(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
