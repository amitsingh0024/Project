import torch
from models.generator import CustomGenerator

def test_simple_generator():
    """
    A simplified test script to validate the input and output shape of the generator.
    """
    # Define the generator with an upscale factor of 4 (adjust if needed)
    generator = CustomGenerator(in_channels=3, upscale_factor=4)
    
    # Create a dummy input tensor with 3 channels (RGB) and size 64x64 (adjustable)
    lr_image = torch.randn(1, 3, 64, 64)  # Batch size of 1, RGB image of size 64x64

    # Pass the dummy input through the generator
    output_image = generator(lr_image)
    
    # Expected output shape: (1, 3, 256, 256) if the upscale factor is 4
    expected_channels = 3
    expected_height = lr_image.shape[2] * generator.upscale_factor
    expected_width = lr_image.shape[3] * generator.upscale_factor

    # Print out the shapes to understand what's being produced
    print(f"Input shape: {lr_image.shape}")
    print(f"Output shape: {output_image.shape}")
    
    # Assertions to confirm output shape matches expectations
    assert output_image.shape[1] == expected_channels, "Output should have 3 channels (RGB)"
    assert output_image.shape[2] == expected_height, f"Output height should be {expected_height}"
    assert output_image.shape[3] == expected_width, f"Output width should be {expected_width}"

    print("Generator output shape is as expected.")

if __name__ == "__main__":
    test_simple_generator()
