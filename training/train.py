import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.datasets import SuperResolutionDataset
from models.generator import CustomGenerator
from models.discriminator import PatchDiscriminator
from losses.pixel_loss import PixelLoss
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import AdversarialLoss
from utils.metrics import calculate_psnr, calculate_ssim
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from config.config import Config

def train():
    # Load configuration
    config = Config()
    
    # Set up device (GPU or CPU) and paths
    device = config.device
    
    # Initialize generator and discriminator models and move to the configured device
    generator = CustomGenerator(in_channels=3, upscale_factor=2).to(device)
    discriminator = PatchDiscriminator(in_channels=config.discriminator_in_channels).to(device)
    
    # Set up optimizers for generator and discriminator
    optimizer_g = Adam(generator.parameters(), lr=config.learning_rate_g, betas=(config.beta1, config.beta2))
    optimizer_d = Adam(discriminator.parameters(), lr=config.learning_rate_d, betas=(config.beta1, config.beta2))
    
    # Initialize loss functions
    pixel_loss_fn = PixelLoss(loss_type="l1").to(device)
    perceptual_loss_fn = PerceptualLoss(loss_type="l1").to(device)
    adversarial_loss_fn = AdversarialLoss(loss_type="bce").to(device)
    
    # Initialize logger for TensorBoard and experiment logging
    logger = Logger(log_dir=config.log_path)
    
    # Load dataset for training
    train_dataset = SuperResolutionDataset(config.lr_image_path, config.hr_image_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    # Load checkpoint if resuming training
    start_epoch = 1
    checkpoint_path = f"{config.checkpoint_path}/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(generator, optimizer_g, checkpoint_path) + 1
    
    # Main training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        generator.train()
        discriminator.train()
        
        for i, (lr_images, hr_images) in enumerate(train_loader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            
            # Debugging statement to check the shape of the input to the generator
            print(f"Shape of input to generator: {lr_images.shape}")  # Check input shape
            
            # Train discriminator
            optimizer_d.zero_grad()
            real_preds = discriminator(hr_images)
            fake_images = generator(lr_images)
            fake_preds = discriminator(fake_images.detach())
            
            # Compute discriminator loss using adversarial loss function
            d_loss = adversarial_loss_fn.discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            fake_preds = discriminator(fake_images)
            
            # Calculate generator loss components
            g_adv_loss = adversarial_loss_fn.generator_loss(fake_preds)
            g_pixel_loss = pixel_loss_fn(fake_images, hr_images)
            g_perceptual_loss = perceptual_loss_fn(fake_images, hr_images)
            
            # Combine generator losses based on specified weights in config
            g_loss = (config.pixel_loss_weight * g_pixel_loss +
                      config.perceptual_loss_weight * g_perceptual_loss +
                      config.adversarial_loss_weight * g_adv_loss)
            g_loss.backward()
            optimizer_g.step()
            
            # Log each loss separately for TensorBoard
            logger.log_loss("train/d_loss", epoch, d_loss.item())  # Log discriminator loss
            logger.log_loss("train/g_loss", epoch, g_loss.item())  # Log generator loss
        
        # Save checkpoint periodically based on save interval
        if epoch % config.save_interval == 0:
            save_checkpoint(generator, optimizer_g, epoch, checkpoint_dir=config.checkpoint_path)
        
        # Evaluate model performance at regular intervals
        if epoch % config.eval_interval == 0:
            generator.eval()
            with torch.no_grad():
                # Initialize metrics for PSNR and SSIM
                psnr_total, ssim_total = 0, 0
                for lr_images, hr_images in train_loader:
                    fake_images = generator(lr_images.to(device))
                    psnr_total += calculate_psnr(fake_images, hr_images.to(device))
                    ssim_total += calculate_ssim(fake_images, hr_images.to(device))
                
                # Calculate average PSNR and SSIM over the validation set
                psnr_avg = psnr_total / len(train_loader)
                ssim_avg = ssim_total / len(train_loader)
                
                # Log PSNR and SSIM metrics to TensorBoard
                logger.log_metrics("val", epoch, {"PSNR": psnr_avg, "SSIM": ssim_avg})

                # Log a sample batch of images
                sample_lr_images = lr_images[:4].to(device)
                sample_fake_images = generator(sample_lr_images)
                sample_hr_images = hr_images[:4].to(device)

                # Log images to TensorBoard for visualization
                logger.log_images("val", epoch, sample_lr_images, name="Low-Res")
                logger.log_images("val", epoch, sample_fake_images, name="Generated High-Res")
                logger.log_images("val", epoch, sample_hr_images, name="Ground Truth High-Res")
    
    # Save the final trained generator model after training
    torch.save(generator.state_dict(), "final_generator_model.pth")
    print("Final generator model saved as 'final_generator_model.pth'")

    # Close the logger to finish writing logs
    logger.close()

if __name__ == "__main__":
    train()
