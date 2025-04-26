import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """
    PatchGAN-inspired discriminator for super-resolution.
    The discriminator processes high-resolution images and outputs a probability
    map indicating the realness of each patch in the image.
    """
    
    def __init__(self, in_channels=3, num_filters=64):
        """
        Initializes the PatchDiscriminator.

        Parameters:
        - in_channels (int): Number of input channels (e.g., 3 for RGB images).
        - num_filters (int): Number of filters for the first convolutional layer, 
                             which doubles after each layer.
        """
        super(PatchDiscriminator, self).__init__()
        
        # Define convolutional layers with increasing number of filters
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1)
        
        # Final layer to produce a single-channel probability map
        self.output_conv = nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        
        # Activation layers
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Debug: Log the discriminator configuration
        print(f"Initialized PatchDiscriminator with base filter count {num_filters}.")
        
    def forward(self, x):
        """
        Forward pass of the discriminator.

        Parameters:
        - x (Tensor): Input high-resolution image with shape (batch_size, 3, height, width).
        
        Returns:
        - out (Tensor): Patch-based probability map.
        """
        # Pass through each convolutional layer with LeakyReLU activations
        out = self.leaky_relu(self.conv1(x))  # Shape: (batch, num_filters, H/2, W/2)
        out = self.leaky_relu(self.conv2(out))  # Shape: (batch, num_filters*2, H/4, W/4)
        out = self.leaky_relu(self.conv3(out))  # Shape: (batch, num_filters*4, H/8, W/8)
        out = self.leaky_relu(self.conv4(out))  # Shape: (batch, num_filters*8, H/8 - 1, W/8 - 1)
        
        # Final convolution to get probability map
        out = self.output_conv(out)  # Shape: (batch, 1, H/8 - 2, W/8 - 2)
        
        # Debug: Print final output shape
        print(f"Discriminator output shape: {out.shape}")
        
        return out
