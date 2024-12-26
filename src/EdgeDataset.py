import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class Edgedataset(Dataset):
    def __init__(self, ballpatchdataset):
        self.ballpatchdataset = ballpatchdataset

    def __len__(self):
        return len(self.ballpatchdataset)

    def __getitem__(self, idx):
        patch = self.ballpatchdataset[idx]['patch']
        gray_patch = patch.mean(dim=0, keepdim=True)
        # Convert to numpy array for OpenCV processing (ensure it has the correct dtype)
        gray_patch_np = (gray_patch.squeeze().numpy() * 255).astype(np.uint8)  # Shape: [H, W]

        # Apply Sobel filter to get the edge detection
        sobel_x = cv2.Sobel(gray_patch_np, cv2.CV_64F, 1, 0, ksize=1)  # Gradient along X axis
        sobel_y = cv2.Sobel(gray_patch_np, cv2.CV_64F, 0, 1, ksize=1)  # Gradient along Y axis

        # Calculate the magnitude of the gradient (combining both X and Y gradients)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

        # Normalize the magnitude for better visualization and convert to uint8
        sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))  # Shape: [H, W]

        # Convert the Sobel edge-detected result back to a PyTorch tensor
        edge_patch = torch.from_numpy(sobel_magnitude).unsqueeze(0).float() / 255.0  # Shape: [1, H, W]

        return {'gray': gray_patch, 'edge': edge_patch}

    def show(self, idx):
        sample = self[idx]
        gray = sample['gray']
        edge = sample['edge']
        gray_np = gray.squeeze().numpy()  # Shape: [H, W]
        edge_np = edge.squeeze().numpy()  # Shape: [H, W]

        # Create a figure to display both grayscale and edge images
        fig, axs = plt.subplots(1, 2, figsize=(5, 3))
        axs[0].imshow(gray_np, cmap='gray')
        axs[0].set_title('Grayscale Patch')
        axs[0].axis('off')

        axs[1].imshow(edge_np, cmap='gray')
        axs[1].set_title('Edge-detected Patch (Sobel)')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()