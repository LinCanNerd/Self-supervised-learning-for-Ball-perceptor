import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class ColorizationDataset(Dataset):
    def __init__(self, ballpatchdataset):
        """
        Initialize the ColorizationDataset.

        Args:
            ballpatchdataset: The original dataset containing image patches.
        """
        self.ballpatchdataset = ballpatchdataset

    def __len__(self):
        """
        Return the length of the dataset (number of samples).
        """
        return len(self.ballpatchdataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing:
            - 'gray': Grayscale version of the image patch (1 channel).
            - 'original': The original color image patch (C x H x W).
        """
        # Retrieve the original image patch at the specified index
        original_patch = self.ballpatchdataset[idx]['patch']

        # Compute the grayscale version by taking the mean across color channels
        gray_patch = original_patch.mean(dim=0, keepdim=True)

        return {'gray': gray_patch, 'original': original_patch}

    def show(self, idx):
        """
        Display the grayscale and original image patches for a given index.

        Args:
            idx: Index of the sample to visualize.
        """
        # Retrieve the grayscale and original patches
        sample = self[idx]
        gray = sample['gray']
        original = sample['original']

        # Convert the grayscale patch to a NumPy array for visualization
        gray_np = gray.squeeze().numpy()  # Shape: [H, W]

        # Convert the original patch from [C, H, W] to [H, W, C] for visualization
        original_np = original.permute(1, 2, 0).numpy()

        # Create a figure to display both grayscale and original images side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(5, 3))

        # Display the grayscale patch
        axs[0].imshow(gray_np, cmap='gray')
        axs[0].set_title('Grayscale Patch')
        axs[0].axis('off')  # Remove axis for a cleaner look

        # Display the original patch
        axs[1].imshow(original_np, cmap='gray')
        axs[1].set_title('Original Patch')
        axs[1].axis('off')  # Remove axis for a cleaner look

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
