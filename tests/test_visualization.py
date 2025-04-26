import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import CustomGenerator  # Adjust this import based on your model structure

def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM between two images."""
    # Convert from torch tensors to numpy arrays
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    # Ensure both images have the same dimensions by resizing img2 to match img1
    if img1_np.shape != img2_np.shape:
        # Assuming images are in CxHxW format
        _, target_h, target_w = img1_np.shape
        img2 = transforms.Resize((target_h, target_w))(img2.unsqueeze(0)).squeeze(0)
        img2_np = img2.cpu().numpy()
    
    # Convert from CxHxW to HxWxC format for skimage
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))
    
    # Ensure the values are in the valid range [0, 1]
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # Calculate PSNR
    psnr_value = psnr(img1_np, img2_np, data_range=1.0)
    
    # Calculate SSIM
    ssim_value = ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)
    
    return psnr_value, ssim_value

def load_model(model_path):
    """Load the trained generator model."""
    model = CustomGenerator()  # Adjust parameters based on your model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def process_image(image_path, transform=None):
    """Load and preprocess the input image."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust size based on your model's requirements
            transforms.ToTensor(),
        ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def visualize_results(input_path, model_path, output_dir='test_outputs'):
    """Visualize input and output images side by side."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Process input image
    input_tensor = process_image(input_path)
    
    # Generate output
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert tensors to images for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show input image
    input_img = make_grid(input_tensor, normalize=True)
    ax1.imshow(input_img.permute(1, 2, 0))
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Show output image
    output_img = make_grid(output_tensor, normalize=True)
    ax2.imshow(output_img.permute(1, 2, 0))
    ax2.set_title('Output Image')
    ax2.axis('off')
    
    # Calculate PSNR and SSIM
    psnr_value, ssim_value = calculate_metrics(input_img, output_img)
    
    # Add metrics as text to the plot
    plt.figtext(0.5, 0.01, 
                f'PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    output_filename = os.path.join(output_dir, 
                                 f'comparison_{os.path.basename(input_path)}')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_filename}")
    print(f"Image Quality Metrics:")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

if __name__ == '__main__':
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Example usage with correct paths
    model_path = os.path.join(project_root, 'final_generator_model.pth')
    
    # You can place your test image in the data/test directory
    test_image_dir = os.path.join(project_root, 'data', 'test')
    os.makedirs(test_image_dir, exist_ok=True)
    
    # Update this path to your actual test image
    input_path = os.path.join(test_image_dir, '0004x2.png')
    
    if not os.path.exists(input_path):
        print(f"Please place a test image at: {input_path}")
    elif not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
    else:
        visualize_results(input_path, model_path) 