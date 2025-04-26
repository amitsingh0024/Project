import torch
import torch.nn as nn

class PixelLoss(nn.Module):
    """
    Pixel-wise loss for super-resolution, using L1 or L2 distance.
    """
    
    def __init__(self, loss_type="l1"):
        """
        Initializes the PixelLoss.

        Parameters:
        - loss_type (str): Type of pixel loss to use ("l1" for L1 loss, "l2" for L2 loss).
        """
        super(PixelLoss, self).__init__()
        
        if loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid loss_type. Choose 'l1' or 'l2'.")

        # Debug: Output selected loss type
        print(f"Initialized PixelLoss with '{loss_type}' loss.")
        
    def forward(self, generated, target):
        """
        Computes the pixel-wise loss between generated and target images.

        Parameters:
        - generated (Tensor): Generated high-resolution image.
        - target (Tensor): Ground truth high-resolution image.
        
        Returns:
        - Tensor: Calculated pixel-wise loss.
        """
        # Calculate and return the pixel-wise loss
        return self.loss(generated, target)
