import torch
from losses.adversarial_loss import AdversarialLoss

def test_adversarial_loss():
    """
    Test function for AdversarialLoss.
    Initializes the AdversarialLoss, computes the generator and discriminator
    losses with dummy predictions, and verifies the results.
    """
    # Create dummy predictions from the discriminator
    batch_size = 4
    fake_preds = torch.randn(batch_size, 1)  # Dummy predictions for generated (fake) images
    real_preds = torch.randn(batch_size, 1)  # Dummy predictions for real images
    
    # Initialize adversarial loss with BCE loss type
    adv_loss_fn = AdversarialLoss(loss_type="bce")
    
    # Test generator loss
    gen_loss = adv_loss_fn.generator_loss(fake_preds)
    print(f"Generator Loss: {gen_loss.item()}")

    # Test discriminator loss
    disc_loss = adv_loss_fn.discriminator_loss(real_preds, fake_preds)
    print(f"Discriminator Loss: {disc_loss.item()}")

    # Assertions to verify that losses are calculated
    assert gen_loss.item() >= 0, "Generator loss should be non-negative."
    assert disc_loss.item() >= 0, "Discriminator loss should be non-negative."

    print("AdversarialLoss test passed successfully.")

if __name__ == "__main__":
    test_adversarial_loss()
