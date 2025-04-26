import torch
from models.generator import CustomGenerator

def test_custom_generator():
    """
    Test function for CustomGenerator.
    Initializes the generator, performs a forward pass with dummy input,
    and checks the output image dimensions.
    """
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the generator with configuration parameters
    in_channels = 1024  # Matches Swin Transformer feature map output channels
    upscale_factor = 4  # Upscaling factor (e.g., 4x for super-resolution)
    generator = CustomGenerator(in_channels=in_channels, upscale_factor=upscale_factor)
    generator = generator.to(device)
    
    # Create a dummy input tensor (batch_size=1, in_channels=1024, 7x7 spatial dimensions)
    dummy_input = torch.rand(1, in_channels, 7, 7).to(device)
    
    # Forward pass to generate high-resolution image
    with torch.no_grad():  # No need to track gradients for testing
        output_image = generator(dummy_input)
    
    # Expected output shape: (batch_size, 3, 224, 224) for 4x upscaling
    expected_height = 7 * upscale_factor
    expected_width = 7 * upscale_factor

    # Debug: Print the output shape
    print(f"Generated image shape: {output_image.shape}")

    # Assertions to verify output dimensions
    assert output_image.shape[1] == 3, "Output should have 3 color channels (RGB)"
    assert output_image.shape[2] == expected_height, f"Output height should be {expected_height}"
    assert output_image.shape[3] == expected_width, f"Output width should be {expected_width}"
    
    print("CustomGenerator test passed successfully.")

if __name__ == "__main__":
    test_custom_generator()
