import torch
from models.swin_transformer import SwinFeatureExtractor

def test_swin_feature_extractor():
    """
    Test function for SwinFeatureExtractor.
    Initializes the feature extractor, performs a forward pass with dummy input,
    and checks the output feature map dimensions.
    """
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Swin feature extractor
    model = SwinFeatureExtractor(model_name="microsoft/swin-base-patch4-window7-224")
    model = model.to(device)
    
    # Create a dummy input tensor (batch_size=1, 3 color channels, 224x224 resolution)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    
    # Forward pass to extract feature map
    with torch.no_grad():  # No need to track gradients for testing
        feature_map = model(dummy_input)
    
    # Output the shape of the feature map
    print(f"Feature map shape: {feature_map.shape}")

    # Verify the expected output shape
    assert feature_map.dim() == 4, "Feature map should have 4 dimensions (batch, channels, height, width)"
    assert feature_map.shape[1] > 0, "Feature map should have non-zero channels"
    print("SwinFeatureExtractor test passed successfully.")

if __name__ == "__main__":
    test_swin_feature_extractor()
