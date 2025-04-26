import torch
from losses.pixel_loss import PixelLoss

def test_pixel_loss():
    """
    Test function for PixelLoss.
    Initializes the PixelLoss with both L1 and L2 loss types, computes the loss
    between dummy generated and target images, and verifies the results.
    """
    # Create dummy generated and target images (batch_size=1, 3 channels, 64x64 resolution)
    generated = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    # Test L1 (MAE) loss
    l1_loss_fn = PixelLoss(loss_type="l1")
    l1_loss = l1_loss_fn(generated, target)
    
    # Debug: Print the calculated L1 loss
    print(f"L1 Pixel Loss: {l1_loss.item()}")

    # Test L2 (MSE) loss
    l2_loss_fn = PixelLoss(loss_type="l2")
    l2_loss = l2_loss_fn(generated, target)
    
    # Debug: Print the calculated L2 loss
    print(f"L2 Pixel Loss: {l2_loss.item()}")

    # Assert that the loss values are positive
    assert l1_loss.item() >= 0, "L1 loss should be non-negative."
    assert l2_loss.item() >= 0, "L2 loss should be non-negative."

    print("PixelLoss test passed successfully.")

if __name__ == "__main__":
    test_pixel_loss()
