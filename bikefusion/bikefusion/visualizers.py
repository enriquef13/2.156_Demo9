import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_imagesets(*args,titles=None):
    """
    A function to visualize image sets in rows. 
    Each row corresponds to a different image set with images stacked corresponding to oneanother.
    
    Parameters:
    - args: np.ndarray,torch.tensor the image sets to visualize (must be in B x C x H x W format)
    """
    
    if titles is not None:
        assert len(titles) == len(args), "Number of titles must match number of image sets"
    elif titles is None:
        pass
    elif not isinstance(titles[0], str) and len(titles[0]) > 1:
        for i in range(len(args)):
            assert len(titles[i]) == len(args[i]), "Number of titles must match number of images in each set or be 1"
    
    n_sets = len(args)
    n_images = len(args[0])
    fig, axs = plt.subplots(n_sets, n_images, figsize=(n_images * 1.5, n_sets * 1.5))
    for i in range(n_sets):
        for j in range(n_images):
            if isinstance(args[i][j], torch.Tensor):
                image = args[i][j].detach().cpu().numpy()
            elif isinstance(args[i][j], np.ndarray):
                image = args[i][j]
            else:
                raise ValueError("Image must be a numpy array or torch tensor")
            if image.shape[0] == 1:
                if n_sets == 1:
                    axs[j].imshow(image[0], cmap="gray")
                else:
                    axs[i, j].imshow(image[0], cmap="gray")
            else:
                if n_sets == 1:
                    axs[j].imshow(np.transpose(image, (1, 2, 0)))
                else:
                    axs[i, j].imshow(np.transpose(image, (1, 2, 0)))
            if n_sets == 1:
                axs[j].axis("off")
            else:
                axs[i, j].axis("off")
            if titles is not None and not isinstance(titles[i], str):
                if len(titles[i]) == 1:
                    axs[j].set_title(titles[i][0])
                else:
                    axs[i, j].set_title(titles[i][j])
            elif titles is not None:
                if n_sets == 1:
                    axs[j].set_title(titles[i])
                else:
                    axs[i, j].set_title(titles[i])
                
                
    plt.tight_layout()
    plt.show()
    
def visualize_image_evolution(images, titles=None):
    """
    A function to visualize the evolution of images over time.
    
    Parameters:
    - images: np.ndarray, shape (n_images, n_timesteps, channels, height, width), the images to visualize
    - titles: list, the titles for each iteration
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
        
    if titles is not None:
        assert len(titles) == images.shape[1], "Number of titles must match number of timesteps"
    n_images = images.shape[0]
    n_timesteps = images.shape[1]
    fig, axs = plt.subplots(n_images, n_timesteps, figsize=(n_timesteps * 1.5, n_images * 1.5))
    for i in range(n_images):
        for j in range(n_timesteps):
            if images.shape[2] == 1:
                axs[i, j].imshow(images[i, j, 0], cmap="gray")
            else:
                axs[i, j].imshow(np.transpose(images[i, j], (1, 2, 0)))
            axs[i, j].axis("off")
            if titles is not None:
                axs[i, j].set_title(titles[j])
    plt.tight_layout()
    plt.show()