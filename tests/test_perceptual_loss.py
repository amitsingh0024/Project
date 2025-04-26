import torch
from losses.perceptual_loss import PerceptualLoss

def test_perceptual_loss():
    """
    Test function for PerceptualLoss.
    Initializes the PerceptualLoss with both L1 and L2 loss types, computes the loss
    between dummy generated and target images, and verifies the results.
    """
    # Create dummy generated and target images (batch_size=1, 3 channels, 224x224 resolution)
    generated = torch.rand(1, 3, 224, 224)
    target = torch.rand(1, 3, 224, 224)
    
    # Test L1 perceptual loss
    l1_loss_fn = PerceptualLoss(loss_type="l1")
    l1_loss = l1_loss_fn(generated, target)
    
    # Debug: Print the calculated L1 perceptual loss
    print(f"L1 Perceptual Loss: {l1_loss.item()}")

    # Test L2 perceptual loss
    l2_loss_fn = PerceptualLoss(loss_type="l2")
    l2_loss = l2_loss_fn(generated, target)
    
    # Debug: Print the calculated L2 perceptual loss
    print(f"L2 Perceptual Loss: {l2_loss.item()}")

    # Assertions to verify loss values are positive
    assert l1_loss.item() >= 0, "L1 perceptual loss should be non-negative."
    assert l2_loss.item() >= 0, "L2 perceptual loss should be non-negative."

    print("PerceptualLoss test passed successfully.")

if __name__ == "__main__":
    test_perceptual_loss()
