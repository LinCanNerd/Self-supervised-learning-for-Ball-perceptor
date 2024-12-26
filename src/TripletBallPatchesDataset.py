from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import torch
import tqdm
class TripletBallPatchesDataset(Dataset):
    def __init__(self, original_dataset,model,k=5, batch_size = 256):
        self.original_dataset = original_dataset
        self.model = model
        self.positive_patch = []
        self.negative_patch = []
        self.batch_size = batch_size

        for i in range(len(self.original_dataset)):
            sample = self.original_dataset[i]
            if sample['contains_ball'] == True:
                self.positive_patch.append(self.rgb_to_grayscale(sample['patch']))
            elif sample['contains_ball'] == False:
                self.negative_patch.append(self.rgb_to_grayscale(sample['patch']))
        
        # Precompute all possible (anchor, positive) combinations
        self.triplets = []
        for anchor in tqdm.tqdm(self.positive_patch):
            subtriplets = []
            while len(subtriplets) < k:
                positive = random.choice(self.positive_patch)
                if not torch.eq(anchor, positive).all():  # Avoid self-pairing
                    # Select a random batch of negatives
                    random_negatives = random.sample(self.negative_patch, min(self.batch_size, len(self.negative_patch)))

                    # Compute embeddings for the negatives in the batch
                    with torch.no_grad():
                        anchor_embedding = self.model(anchor.unsqueeze(0))
                        positive_embedding = self.model(positive.unsqueeze(0))
                        negative_embeddings = torch.stack([self.model(neg.unsqueeze(0)) for neg in random_negatives])

                    # Compute distances
                    anchor_negative_distances = torch.cdist(anchor_embedding, negative_embeddings).squeeze(0)
                    positive_negative_distances = torch.cdist(positive_embedding, negative_embeddings).squeeze(0)

                    # Find the negative with the smallest sum of distances
                    total_distances = anchor_negative_distances + positive_negative_distances
                    best_negative_idx = torch.argmin(total_distances).item()
                    best_negative = random_negatives[best_negative_idx]

                    # Add the triplet
                    subtriplets.append((anchor, positive, best_negative))
            self.triplets+=subtriplets


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        
        anchor, positive, negative = self.triplets[idx]

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }

    def rgb_to_grayscale(self,rgb_tensor):
        # rgb_tensor is expected to be of shape (C, H, W), where C = 3 for RGB
        grayscale_tensor = rgb_tensor.mean(dim=0, keepdim=True)  # Average across the 3 channels
        return grayscale_tensor
    
    def show(self, idx):
        triplet = self[idx]
        # Extract patches
        anchor_patch = triplet['anchor'].squeeze(0).numpy()  # Remove the channel dimension
        positive_patch = triplet['positive'].squeeze(0).numpy()  # Remove the channel dimension
        negative_patch = triplet['negative'].squeeze(0).numpy()  # Remove the channel dimension
        
        # Display the triplet
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(anchor_patch, cmap='gray')  # Use grayscale colormap
        plt.title('Anchor')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(positive_patch, cmap='gray')  # Use grayscale colormap
        plt.title('Positive')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(negative_patch, cmap='gray')  # Use grayscale colormap
        plt.title('Negative')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()