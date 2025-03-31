import numpy as np
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch

def tensor2np(img):
    return img.detach().cpu().numpy().transpose((1, 2, 0))
def tensor2PIL(img):
    return ToPILImage(img.detach().cpu())
def np2PIL(img):
    if img.ndim > 2: img=img.squeeze(axis=2)
    return Image.fromarray((img.clip(0,1)*255).astype(np.uint8))


def save_training_history(train_losses, val_errors, save_dir='./training_history'):
    """
    Save training and validation metrics history as both plot and CSV
    
    Args:
        train_losses: Dictionary of epoch to training loss
        val_errors: Dictionary of epoch to validation MSE
        save_dir: Directory to save the plots and data
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(train_losses.keys()), list(train_losses.values()), label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    val_epochs = list(val_errors.keys())
    # Convert CUDA tensors to CPU numpy arrays
    val_mses = [val.cpu().item() if isinstance(val, torch.Tensor) else val for val in val_errors.values()]
    plt.plot(val_epochs, val_mses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE History')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_{timestamp}.png'))
    plt.close()

def save_augmented_samples(dataset, num_samples=5, save_dir='./augmented_samples'):
    """
    Save samples of augmented images alongside their original versions
    
    Args:
        dataset: BasicDataset instance with transformations
        num_samples: Number of samples to save
        save_dir: Directory to save the samples
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(min(num_samples, len(dataset))):
        original = dataset.get_raw_image(i)

        augmented_data = dataset[i]
        augmented_img = augmented_data['image']

        augmented_img = np.array(augmented_img)
        augmented_img = augmented_img.transpose(1, 2, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(augmented_img)
        plt.title('Augmented')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f'augmentation_sample_{i}_{timestamp}.png'))
        plt.close()
