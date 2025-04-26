import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN-based training. This class provides separate loss functions
    for the generator and discriminator.
    """
    
    def __init__(self, loss_type="bce"):
        """
        Initializes the AdversarialLoss.

        Parameters:
        - loss_type (str): Type of adversarial loss to use ("bce" for Binary Cross-Entropy).
        """
        super(AdversarialLoss, self).__init__()
        
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Invalid loss_type. Currently, only 'bce' is supported.")
        
        # Debug: Output selected loss type
        print(f"Initialized AdversarialLoss with '{loss_type}' loss.")
    
    def generator_loss(self, fake_preds):
        """
        Computes the generator's adversarial loss.

        Parameters:
        - fake_preds (Tensor): Discriminator predictions for generated (fake) images.
        
        Returns:
        - Tensor: Calculated generator loss.
        """
        # Generate target labels of 1 (real) for the generator loss
        target_labels = torch.ones_like(fake_preds)
        
        # Calculate and return the generator loss
        loss = self.criterion(fake_preds, target_labels)
        print(f"Generator adversarial loss: {loss.item()}")
        return loss

    def discriminator_loss(self, real_preds, fake_preds):
        """
        Computes the discriminator's adversarial loss.

        Parameters:
        - real_preds (Tensor): Discriminator predictions for real images.
        - fake_preds (Tensor): Discriminator predictions for generated (fake) images.
        
        Returns:
        - Tensor: Calculated discriminator loss.
        """
        # Generate target labels (1 for real images, 0 for fake images)
        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)
        
        # Calculate discriminator loss on real and fake images
        real_loss = self.criterion(real_preds, real_labels)
        fake_loss = self.criterion(fake_preds, fake_labels)
        
        # Sum real and fake losses for total discriminator loss
        loss = (real_loss + fake_loss) / 2
        print(f"Discriminator adversarial loss: {loss.item()}")
        return loss
