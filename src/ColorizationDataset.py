import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class ColorizationDataset(Dataset):
    def __init__(self, ballpatchdataset):
        self.ballpatchdataset = ballpatchdataset

    def __len__(self):
        return len(self.ballpatchdataset)

    def __getitem__(self, idx):
        original_patch = self.ballpatchdataset[idx]['patch']
        gray_patch = original_patch.mean(dim=0, keepdim=True)
        return {'gray': gray_patch, 'original': original_patch}

    def show(self, idx):
        sample = self[idx]
        gray = sample['gray']
        original = sample['original']
        gray_np = gray.squeeze().numpy()  # Shape: [H, W]
        original_np = original.permute(1, 2, 0).numpy()
        # Create a figure to display both grayscale and edge images
        fig, axs = plt.subplots(1, 2, figsize=(5, 3))
        axs[0].imshow(gray_np, cmap='gray')
        axs[0].set_title('Grayscale Patch')
        axs[0].axis('off')

        axs[1].imshow(original_np, cmap='gray')
        axs[1].set_title('Original patch')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()
