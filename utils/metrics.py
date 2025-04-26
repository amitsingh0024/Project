import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_pixel=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
    - img1 (Tensor): Generated high-resolution image.
    - img2 (Tensor): Target high-resolution image.
    - max_pixel (float): Maximum possible pixel value (1.0 for normalized images).
    
    Returns:
    - float: PSNR value.
    """
    mse = F.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Parameters:
    - img1 (Tensor): Generated high-resolution image.
    - img2 (Tensor): Target high-resolution image.
    - window_size (int): Size of the Gaussian kernel used for SSIM calculation.
    - size_average (bool): If True, average SSIM over batch; otherwise, return SSIM per image.
    
    Returns:
    - float: SSIM value.
    """
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")
    
    # Define constants for numerical stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Apply Gaussian filter to calculate mean
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variance
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map

