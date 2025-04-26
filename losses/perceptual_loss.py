import torch
import torch.nn as nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    """
    Perceptual loss for measuring similarity between generated and target images.
    Utilizes features extracted from a pre-trained VGG19 model to compute perceptual loss.
    """
    def __init__(self, loss_type="l1"):
        """
        Initialize PerceptualLoss by selecting specific layers from VGG19 for feature extraction.

        Parameters:
        - loss_type (str): Type of loss function to use ('l1' for L1 Loss, 'l2' for MSE Loss).
        """
        super(PerceptualLoss, self).__init__()
        
        # Load a pre-trained VGG19 model from torchvision and access its feature layers
        vgg = vgg19(pretrained=True).features
        # Specify the layers from VGG19 to use for perceptual feature extraction
        # These layers correspond to specific points in the VGG network
        self.layers = [3, 8, 17, 26]
        # Create a ModuleList to store only the required VGG layers
        self.features = nn.ModuleList([vgg[i] for i in self.layers])
        
        # Set the loss function based on the loss_type parameter
        # Use L1 loss for "l1" and MSE loss for any other value
        self.loss = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
        
        # Ensure that all ReLU layers do not modify tensors in-place
        # This prevents in-place operations that interfere with gradient computation
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False  # Disable in-place operations to preserve gradient integrity

    def forward(self, generated, target):
        """
        Forward pass to compute perceptual loss by comparing features of generated and target images.

        Parameters:
        - generated (Tensor): Generated high-resolution images.
        - target (Tensor): Ground truth high-resolution images.
        
        Returns:
        - Tensor: Calculated perceptual loss value.
        """
        # Initialize feature tensors for generated and target images
        generated_features = generated
        target_features = target
        perceptual_loss = 0.0  # Accumulator for the perceptual loss
        
        # Pass the generated and target images through the selected VGG layers
        for layer in self.features:
            # Extract features from the current layer for both images
            generated_features = layer(generated_features)
            target_features = layer(target_features)
            
            # Compute the loss between generated and target features at the current layer
            perceptual_loss += self.loss(generated_features, target_features)
        
        # Return the accumulated perceptual loss across all selected layers
        return perceptual_loss
