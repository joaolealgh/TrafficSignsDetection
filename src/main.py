import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

import argparse
from dotenv import load_dotenv

from train import train_all_epochs, test
from dataset_functions import *
from model import CustomCNN

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

torch.manual_seed(42)

# Load from .env file
PLOT_DATASET = bool(os.environ.get("PLOT_DATASET", False))
CALCULATE_MEAN_STD = bool(os.environ.get("CALCULATE_MEAN_STD", False))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", 32))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", 32))
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
LR = float(os.environ.get("LR", 0.001))
DATASET_DIR = str(os.environ.get("DATASET_DIR", '../dataset/GTSRB'))
EPOCHS = int(os.environ.get("EPOCHS", 10))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
SEED = int(os.environ.get("SEED", 42))

def train_model(device, plot_dataset, dataset_path):
    if CALCULATE_MEAN_STD:
        mean, std = calculate_mean_std_custom_dataset(dataset_path)
    else:
        mean = torch.tensor([0.3403, 0.3121, 0.3214])
        std = torch.tensor([0.2724, 0.2608, 0.2669])

    training_data_transform = transforms.Compose([
        ConvertPIL(),
        Rescale(IMAGE_SIZE),
        RandCrop(32),
        RandHorizFlip(0.5),
        RandVertFlip(0.5),
        # RandomRotation(),
        ToTensor(),
        CustomNormalize(mean, std)
    ])

    validation_data_transform = transforms.Compose([
        ConvertPIL(),
        Rescale(IMAGE_SIZE),
        ToTensor(),
        CustomNormalize(mean, std)
    ])

    
    train_dataset = TrafficSignsDataset(annotations_file=os.path.join(dataset_path, 'Train.csv'), 
                                        root_dir=DATASET_DIR,
                                        transform=training_data_transform)

    valid_dataset = TrafficSignsDataset(annotations_file=os.path.join(dataset_path, 'Train.csv'), 
                                        root_dir=DATASET_DIR,
                                        transform=validation_data_transform)

    total_count = len(train_dataset)
    train_count = int(0.8 * total_count)
    print(total_count)
    print(train_count)
    indices = np.arange(0, total_count, 1)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_count], indices[train_count:]

    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, val_idx)

    train_dataset_loader = DataLoader(train_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

    validation_dataset_loader = DataLoader(valid_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)
    
    if plot_dataset:
        plot_gtsrb_dataset_images(train_dataset_loader)

    
    num_classes = len(train_dataset.dataset.get_classes())
    print(num_classes)

    net = CustomCNN(num_classes=num_classes)

    net.to(device)

    train_all_epochs(device, net, train_dataset_loader, EPOCHS, validation_dataset_loader, lr=LR)


def test_model(model_path, device, plot_dataset, dataset_path):
    if CALCULATE_MEAN_STD:
        mean, std = calculate_mean_std_custom_dataset(dataset_path)
    else:
        mean = torch.tensor([0.3403, 0.3121, 0.3214])
        std = torch.tensor([0.2724, 0.2608, 0.2669])


    test_data_transform = transforms.Compose([
                            ConvertPIL(),
                            Rescale(IMAGE_SIZE),
                            ToTensor(),
                            CustomNormalize(mean, std)
                        ])
    
    test_traffic_dataset = TrafficSignsDataset(annotations_file=os.path.join(dataset_path, 'Test.csv'), 
                                            root_dir=dataset_path, 
                                            transform=test_data_transform)

    test_dataset_loader = DataLoader(test_traffic_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

    if plot_dataset:
        plot_gtsrb_dataset_images(test_dataset_loader)

    num_classes = len(test_traffic_dataset.get_classes())
    net = CustomCNN(num_classes=num_classes)
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()
 
    test(net, device, test_dataset_loader)


def main(action, dataset_path, plot_dataset, device, model_path):
    if device == 'CUDA' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if action == 'TRAIN':
        train_model(device, plot_dataset, dataset_path)
    
    else:
        test_model(model_path, device, plot_dataset, dataset_path)


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action',
                        '-a',
                        choices=['TRAIN', 'INFERENCE'], 
                        required=True,
                        help='TRAIN or INFERENCE')
    
    parser.add_argument('--dataset-path',
                        default=DATASET_DIR,
                        help='Path to download the dataset to')
    
    parser.add_argument('--display-dataset', 
                        default=PLOT_DATASET,
                        type=bool,
                        help='Display a subset of images from dataset')

    parser.add_argument('--device',
                        choices=['CUDA', 'CPU'],
                        default='CPU',
                        type=str,
                        help='Device CUDA or CPU')
    
    parser.add_argument('--model-path',
                        help='Location to save the model if action is TRAIN or location to load the model from if action is INFERENCE')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = run()

    action = args.action
    dataset_path = args.dataset_path
    plot_dataset = args.display_dataset
    device = args.device
    model_path = args.model_path

    main(action, dataset_path, plot_dataset, device, model_path)
