import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from src import config

# --- Create Custom PyTorch Dataset ---
class PneumoniaDataset(Dataset):  # class dataset defined so that Pytorch can interact.
    def __init__(self, images, labels, transform=None): # method stores images, lables and transform.
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self): # len method that simply tells the total number of images in the dataset.
        return len(self.images)

    def __getitem__(self, idx): # method that defines how to get a single image and its corresponding label. 
        image = self.images[idx]
        # Ensure label is a float for BCEWithLogitsLoss
        label = self.labels[idx].astype(np.float32) 
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(label)

def get_dataloaders(data_path, batch_size):  # function get_dataloaders loads the pneumoniamnist.npz file
    
    # Load the dataset
    try:
        data = np.load(data_path) ## load dataset file name
    except FileNotFoundError: ## If computer could not find the file name.
        print(f"Error: '{data_path}' not found.")
    
    # Extract the splitting from dataset
    train_images, train_labels = data['train_images'], data['train_labels']
    print(f"Training data shape: {train_images.shape}")
    
    val_images, val_labels = data['val_images'], data['val_labels']
    print(f"Validation data shape: {val_images.shape}")
    
    test_images, test_labels = data['test_images'], data['test_labels']
    print(f"Test data shape: {test_images.shape}")

    # --- Define Transformations ---
    # Inception-V3 requires 299x299 input. 
    # as expected by ImageNet-pre-trained models.
    # Data augmentation is applied only to the training set.

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3), # InceptionV3 needs 3 channels
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # --- Create Datasets and DataLoaders ---
    train_dataset = PneumoniaDataset(train_images, train_labels, transform=data_transforms['train'])
    val_dataset = PneumoniaDataset(val_images, val_labels, transform=data_transforms['val'])
    test_dataset = PneumoniaDataset(test_images, test_labels, transform=data_transforms['val']) # No augmentation for test

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    return dataloaders, train_labels  # A dictionary containing the three ready-to-use dataloaders. 
#The train_labels array, which is needed in the main script to calculate the class weights for handling the imbalanced dataset.
