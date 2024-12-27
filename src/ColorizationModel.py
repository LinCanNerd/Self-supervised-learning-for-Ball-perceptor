import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_residual=False):
        """
        Initialize the DepthwiseSeparableConv block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the depthwise convolution.
            use_residual: If True, adds a residual connection.
        """
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: applies separate convolution for each channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        # Pointwise convolution: combines the depthwise output to produce out_channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # Batch normalization for stability
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function
        self.relu = nn.ReLU()
        # Flag to enable/disable residual connection
        self.use_residual = use_residual

    def forward(self, x):
        """
        Forward pass of the DepthwiseSeparableConv block.
        """
        # Apply depthwise convolution
        z = self.depthwise(x)
        # Apply pointwise convolution
        z = self.pointwise(z)
        # Apply batch normalization
        z = self.bn(z)
        # Add residual connection if enabled
        z = self.relu(z) + x if self.use_residual else self.relu(z)
        return z


class ColorizationModel(nn.Module):
    def __init__(self, backbone_size=9):
        """
        Initialize the ColorizationModel.

        Args:
            backbone_size: Number of repeated backbone layers for feature extraction.
        """
        super(ColorizationModel, self).__init__()

        # Initial convolutional layers
        self.conv1 = DepthwiseSeparableConv(1, 8, stride=1)
        self.conv2 = DepthwiseSeparableConv(8, 16, stride=1)
        self.conv3 = DepthwiseSeparableConv(16, 32, stride=1)

        # Backbone: Stack of DepthwiseSeparableConv layers with residual connections
        backbone_layers = []
        for _ in range(backbone_size):
            backbone_layers.append(DepthwiseSeparableConv(32, 32, stride=1, use_residual=True))
        self.backbone = nn.Sequential(*backbone_layers)

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder: Upsampling and reconstructing the color image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)  # Final layer to produce 3-channel RGB output
        )

    def forward(self, x):
        """
        Forward pass of the ColorizationModel.

        Args:
            x: Input grayscale image with shape [N, 1, H, W].

        Returns:
            Output colorized image with shape [N, 3, H, W].
        """
        # Apply initial convolutions with pooling
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))

        # Pass through the backbone feature extractor
        x = self.backbone(x)

        # Reshape feature map if needed (placeholder operation for clarity)
        N, C, H, W = x.shape
        feature_map = x.view(N, -1)  # Flatten features for potential downstream tasks
        x = feature_map.view(N, C, H, W)  # Reshape back to original dimensions

        # Decode the features into a colorized image
        output = self.decoder(x)

        return output
