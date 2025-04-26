from transformers import AutoModel
import torch
import torch.nn as nn

class SwinFeatureExtractor(nn.Module):
    """
    Swin Transformer feature extractor for super-resolution.
    This class loads a pre-trained Swin Transformer and configures it
    to output feature maps instead of classification predictions.
    """
    
    def __init__(self, model_name="microsoft/swin-base-patch4-window7-224", feature_layer=-1):
        """
        Initializes the SwinFeatureExtractor.

        Parameters:
        - model_name (str): Name of the pre-trained Swin Transformer model.
        - feature_layer (int): Layer from which to extract features. Default is -1 (last layer).
        """
        super(SwinFeatureExtractor, self).__init__()
        
        # Load pre-trained Swin Transformer model from Hugging Face
        self.swin = AutoModel.from_pretrained(model_name)
        
        # Disable gradients for the feature extractor (freeze layers)
        for param in self.swin.parameters():
            param.requires_grad = False

        # Set the layer to extract features from
        self.feature_layer = feature_layer

        # Debug: Output model configuration
        print(f"Initialized SwinFeatureExtractor with model '{model_name}'")
    
    def forward(self, x):
        """
        Forward pass to extract feature maps from the specified Swin Transformer layer.

        Parameters:
        - x (Tensor): Input image tensor with shape (batch_size, 3, height, width).
        
        Returns:
        - feature_map (Tensor): Extracted feature map tensor.
        """
        # Pass input through the Swin Transformer to get hidden states
        outputs = self.swin(x, output_hidden_states=True)
        
        # Extract the feature map from the specified layer
        feature_map = outputs.hidden_states[self.feature_layer]  # Shape: [batch, num_patches, channels]

        # Reshape to (batch, channels, height, width)
        batch_size, num_patches, channels = feature_map.shape
        height = width = int(num_patches ** 0.5)  # Assuming square arrangement of patches
        feature_map = feature_map.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        
        # Debug: Print shape of the extracted and reshaped feature map
        print(f"Extracted and reshaped feature map shape: {feature_map.shape}")
        
        return feature_map
