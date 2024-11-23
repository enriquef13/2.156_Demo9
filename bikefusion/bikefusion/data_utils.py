import torchvision.transforms.functional as F
import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, random_split
from .diffusion import InpaintingDenoisingDiffusion
import os

def to_mask_map(maks, image_size=(80,128)):
    bs = maks.shape[0]
    mask_map = np.zeros((bs, 1, image_size[0], image_size[1]))
    
    for i in range(bs):
        left, bottom, right, top = maks[i]
        mask_map[i, 0, top:bottom, left:right] = 1
        
    return mask_map

def pad_to_square(image, target_size=(128, 128)):
    # Calculate padding for each dimension
    h, w = image.shape[1], image.shape[2]
    delta_w = target_size[1] - w
    delta_h = target_size[0] - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # Pad with white (255) for each side
    padded_image = F.pad(image, padding, fill=255)
    return padded_image

def preprocess(images):
    # Convert arrays to tensors
    images = torch.tensor(images).float()

    # Apply padding to each image in the dataset
    images = torch.stack([pad_to_square(img) for img in images])
    
    # Normalize images to [-1, 1]
    images = images / 255.0 * 2 - 1
    
    return images

def un_pad(image, target_size=(80, 128)):
    # Calculate padding for each dimension
    h, w = image.shape[1], image.shape[2]
    delta_w = w - target_size[1]
    delta_h = h - target_size[0]
    
    # Unpad the image
    un_padded_image = image[:, delta_h // 2:h - delta_h // 2, delta_w // 2:w - delta_w // 2]
    
    return un_padded_image

def postprocess(images):
    # Convert tensors to arrays
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif isinstance(images, np.ndarray):
        pass
    else:
        raise ValueError("images must be either a torch.Tensor or a np.ndarray")

    # Unpad each image in the dataset
    images = np.stack([un_pad(img) for img in images])
    
    # Rescale images to [0, 255]
    images = (images + 1) / 2 * 255
    
    return images.astype(np.uint8)

def load_data(split="train", path="data/"):
    """
    A function to load the data.

    Parameters:
    - split: str, "train" or "test"

    Returns:
    - masked_images: np.ndarray, shape (n_images, channels, height, width), the images with random masks applied
    - masks: np.ndarray, shape (n_images, 4), the boundaries of the masks (left, bottom, right, top)
    - parametric: np.ndarray, shape (n_images, 3), the parametric representation of the images
    - description: list, the description of the images
    - images: np.ndarray, shape (n_images, channels, height, width), the original images
    """
    masked_images = []
    if split == "train":
        for i in range(5):
            masked_im_slice = np.load(f"{path}masked_train_{i}.npy")
            masked_images.append(masked_im_slice)
        masked_images = np.concatenate(masked_images, axis=0)

        images = []
        for i in range(5):
            im_slice = np.load(f"{path}images_train_{i}.npy")
            images.append(im_slice)
        images = np.concatenate(images, axis=0)
    else:
        masked_images = np.load(f"{path}masked_test.npy")
        images = None

    description = pickle.load(open(f"{path}desc_{split}.pkl", "rb"))

    parametric = np.load(f"{path}param_{split}.npy")

    masks = np.load(f"{path}mask_{split}.npy")


    return masked_images, masks, parametric, description, images

def load_bikefusion_and_data(current_dir):
    partial_images, masks, parametric, description, targets  = load_data(path=os.path.join(current_dir, 'data/'))
    
    training_images = preprocess(targets)
    
    dataset = TensorDataset(training_images)

    # split to training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    Diffuser = InpaintingDenoisingDiffusion(train_dataset, val_dataset, image_size=128)
    Diffuser.load_checkpoint(os.path.join(current_dir, 'chekpoint/bikefusion.pt'))
    
    return Diffuser