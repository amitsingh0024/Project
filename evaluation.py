import os
import torch
from torch.utils.data import DataLoader
from data.datasets import SuperResolutionDataset
from models.generator import CustomGenerator
from utils.metrics import calculate_psnr, calculate_ssim
from PIL import Image
import numpy as np
from config.config import Config

def evaluate():
    # Load configuration settings
    config = Config()
    device = config.device
    
    # Load the trained generator model
    generator = CustomGenerator(in_channels=3, upscale_factor=config.upscale_factor).to(device)
    generator.load_state_dict(torch.load("final_generator_model.pth", map_location=device))
    generator.eval()  # Set to evaluation mode
    
    # Dimensions for the evaluation based on dataset configurations
    lr_size = (64, 64)  # Replace with appropriate dimensions if different
    hr_size = (128, 128)  # Replace with appropriate dimensions if different

    # Load the test dataset with specified sizes
    test_dataset = SuperResolutionDataset(config.lr_image_path, config.hr_image_path, lr_size=lr_size, hr_size=hr_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
    
    # Initialize metrics
    psnr_total, ssim_total = 0, 0
    num_samples = len(test_loader)
    
    # Directory to save generated images for comparison
    output_dir = os.path.join(config.output_path, "generated_vs_ground_truth")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on each test sample
    with torch.no_grad():
        for idx, (lr_image, hr_image) in enumerate(test_loader):
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            
            # Generate high-resolution image from low-resolution input
            generated_hr = generator(lr_image)
            
            # Calculate PSNR and SSIM for the generated output
            psnr = calculate_psnr(generated_hr, hr_image)
            ssim = calculate_ssim(generated_hr, hr_image)
            psnr_total += psnr
            ssim_total += ssim
            
            # Save sample images for visual comparison
            save_sample_images(generated_hr, hr_image, lr_image, output_dir, idx, hr_size)

            # Optionally print progress
            if (idx + 1) % config.print_interval == 0:
                print(f"Processed {idx + 1}/{num_samples} images")
    
    # Calculate average PSNR and SSIM
    psnr_avg = psnr_total / num_samples
    ssim_avg = ssim_total / num_samples
    print(f"Average PSNR: {psnr_avg:.4f}, Average SSIM: {ssim_avg:.4f}")

def save_sample_images(generated_hr, hr_image, lr_image, output_dir, idx, hr_size):
    """
    Save a comparison of the generated and ground-truth high-resolution images.
    """
    # Convert tensors to PIL images
    generated_hr = tensor_to_image(generated_hr)
    hr_image = tensor_to_image(hr_image)
    lr_image = tensor_to_image(lr_image, upscale=True, target_size=hr_size)

    # Stack images horizontally for comparison and save
    comparison_image = np.hstack((lr_image, generated_hr, hr_image))
    comparison_pil = Image.fromarray(comparison_image)
    comparison_pil.save(os.path.join(output_dir, f"comparison_{idx}.png"))

def tensor_to_image(tensor, upscale=False, target_size=None):
    """
    Convert a tensor to a numpy array (image format) for saving.
    If upscale is True, the LR image will be resized to match HR dimensions for visual comparison.
    """
    image = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
    image = image.clip(0, 255).astype(np.uint8)
    if upscale and target_size:
        image = np.array(Image.fromarray(image).resize(target_size, Image.BICUBIC))
    return image

if __name__ == "__main__":
    evaluate()
