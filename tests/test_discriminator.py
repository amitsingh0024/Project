import torch
from models.discriminator import PatchDiscriminator

def test_patch_discriminator():
    """
    Test function for PatchDiscriminator.
    Initializes the discriminator, performs a forward pass with dummy input,
    and checks the output probability map dimensions.
    """
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the discriminator with configuration parameters
    in_channels = 3  # Input channels for RGB images
    discriminator = PatchDiscriminator(in_channels=in_channels)
    discriminator = discriminator.to(device)
    
    # Create a dummy high-resolution image tensor (batch_size=1, 3 color channels, 224x224 resolution)
    dummy_input = torch.rand(1, in_channels, 224, 224).to(device)
    
    # Forward pass to get the patch-based probability map
    with torch.no_grad():  # No need to track gradients for testing
        output_map = discriminator(dummy_input)
    
    # Debug: Print the shape of the output map
    print(f"Discriminator output map shape: {output_map.shape}")

    # Verify that the output is a single-channel patch-based probability map
    assert output_map.shape[1] == 1, "Output should have 1 channel for the probability map"
    print("PatchDiscriminator test passed successfully.")

if __name__ == "__main__":
    test_patch_discriminator()
