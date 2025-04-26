import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block used in the generator. Each block contains two convolutional
    layers with ReLU activation in between, followed by a skip connection.
    """
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer in the residual block
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)  # Set inplace=False to prevent in-place modification
        # Second convolutional layer in the residual block
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass for the residual block.

        Parameters:
        - x (Tensor): Input tensor with shape (batch, channels, height, width)

        Returns:
        - Tensor: Output tensor after applying residual block operations
        """
        residual = x  # Save the input as residual for skip connection
        out = self.conv1(x)  # Apply first convolution
        out = self.relu(out)  # Apply ReLU activation
        out = self.conv2(out)  # Apply second convolution
        out += residual  # Add skip connection
        return out

class CustomGenerator(nn.Module):
    """
    Custom Generator model for super-resolution. Takes an input low-resolution image
    and generates a high-resolution image by progressively refining details.
    """
    def __init__(self, in_channels=3, upscale_factor=2):
        """
        Initializes the generator model with a series of residual blocks and upsampling layers.
        
        Parameters:
        - in_channels (int): Number of input channels, set to 3 for RGB images.
        - upscale_factor (int): Upscaling factor for the output resolution.
        """
        super(CustomGenerator, self).__init__()
        
        # Store the upscale factor as a class attribute to make it accessible for testing
        self.upscale_factor = upscale_factor

        # Initial convolution layer to process the input feature map with 3 channels (RGB)
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks to capture image features
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=64) for _ in range(8)]  # Stack of 8 residual blocks
        )
        
        # Convolution layer after residual blocks to process features
        self.conv_after_residual = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Upsampling layers to increase the spatial resolution
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),  # Prepare for upsampling
            nn.PixelShuffle(2),  # First upscaling (factor of 2)
            nn.ReLU(inplace=False),  # Set inplace=False to avoid in-place modification
        )
        
        # Final convolution layer to generate the 3-channel output image
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Ensure output has 3 channels
    
    def forward(self, x):
        """
        Forward pass of the generator model.

        Parameters:
        - x (Tensor): Input tensor representing a low-resolution image of shape (batch, in_channels, height, width)

        Returns:
        - Tensor: Output tensor representing a high-resolution image of shape (batch, in_channels, upscale_factor * height, upscale_factor * width)
        """
        # Initial convolution to process the input image
        out = self.initial_conv(x)
        
        # Save the result of the initial conv layer to add as a skip connection later
        initial_conv_out = out
        
        # Pass through residual blocks
        out = self.residual_blocks(out)
        
        # Pass through additional convolution layer after residual blocks
        out = self.conv_after_residual(out)
        
        # Add skip connection from initial_conv_out
        out += initial_conv_out
        
        # Apply upsampling layers to increase spatial resolution
        out = self.upsampling(out)
        
        # Final convolution to produce the high-resolution output image
        out = self.output_conv(out)
        
        return out
