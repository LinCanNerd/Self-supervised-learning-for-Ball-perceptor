import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import vgg16
from torchvision.ops import MultiScaleRoIAlign
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import models



class AppleDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, image_file.replace('.png', '.csv'))

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert the numpy ndarray to a PIL image
        image = Image.fromarray(image)

        # Apply the transformations (if any)
        if self.transform:
            image = self.transform(image)

        apples_annotations = []

        # Read the annotations if they exist
        if os.path.exists(annotation_path):
            annotations = pd.read_csv(annotation_path)
            for _, row in annotations.iterrows():
                if row['label'] == 1:  # Assuming 1 indicates apple
                    x, y, r = row['c-x'], row['c-y'], row['radius']
                    # Convert circle annotation to rectangle
                    x_min = x - r
                    y_min = y - r
                    x_max = x + r
                    y_max = y + r
                    apples_annotations.append([x_min, y_min, x_max, y_max])

        # If no apples, create an empty tensor for boxes
        if len(apples_annotations) > 0:
            apples_annotations_tensor = torch.tensor(apples_annotations, dtype=torch.float32)
        else:
            apples_annotations_tensor = torch.empty((0, 4))  # No apples found

        # Ensure the label tensor matches the number of annotations
        if len(apples_annotations) > 0:
            apples_labels = [1 for _ in range(len(apples_annotations))]
        else:
            apples_labels = [0]

        labels_tensor = torch.tensor(apples_labels, dtype=torch.int64)

        target = {
            "boxes": apples_annotations_tensor,
            "labels": labels_tensor,
        }

        return image, target



    
class AppleDetector(FasterRCNN):
    def __init__(self, num_classes=2, backbone=None):
        """
        Initialize the Faster R-CNN model with a VGG16 backbone.
        Parameters:
        - num_classes (int): Number of classes (1 apple class + background).
        - backbone (torch.nn.Module): The backbone used for feature extraction.
        """
        # If no backbone is passed, use a default VGG16 backbone
        if backbone is None:
            # Load VGG16 as the backbone with pretrained weights
            vgg = vgg16(weights=None)
            # Use features from the last conv layer
            backbone = vgg.features[:30]  # Up to the last conv layer
            backbone.out_channels = 512


            
        # Define anchor generator for RPN
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),  # Different sized anchors for apple detection
            aspect_ratios=((0.5, 1.0, 2.0),)  # Different aspect ratios for anchors
        )
        
        # Define ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],  # Feature map to use
            output_size=7,        # Output size of ROI align
            sampling_ratio=2      # Sampling ratio
        )
        
        # Initialize the Faster R-CNN model with the custom backbone
        super(AppleDetector, self).__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
        
    def forward(self, images, targets=None):
        """
        Forward pass through the Faster R-CNN model.
        Parameters:
        - images (Tensor or List[Tensor]): Input image(s) for detection.
        - targets (List[Dict], optional): Ground-truth targets for training (defaults to None).
        Returns:
        - result (List[Dict]): Prediction or loss dictionary during training.
        """
        return super(AppleDetector, self).forward(images, targets)
