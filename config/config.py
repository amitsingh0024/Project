import torch

class Config:
    """
    Configuration settings for super-resolution GAN training.
    """
    
    # Device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data settings
    lr_image_path = "data/lr_images/"        # Path to low-resolution images
    hr_image_path = "data/hr_images/"        # Path to high-resolution images
    batch_size = 16                          # Batch size for training and validation
    num_workers = 4                          # Number of workers for data loading

    # Model settings
    upscale_factor = 4                       # Upscale factor for generator output
    generator_in_channels = 1024             # Input channels for generator
    discriminator_in_channels = 3            # Input channels for discriminator
    swin_model_name = "microsoft/swin-base-patch4-window7-224"  # Swin Transformer model

    # Training hyperparameters
    num_epochs = 100                         # Number of epochs for training
    learning_rate_g = 5e-5                   # Lower learning rate for generator
    learning_rate_d = 5e-5                   # Lower learning rate for discriminator
    beta1 = 0.5                              # Beta1 for Adam optimizer (stabilizes GAN training)
    beta2 = 0.999                            # Beta2 for Adam optimizer

    # Loss weights
    pixel_loss_weight = 1.0                  # Weight for pixel (L1) loss
    perceptual_loss_weight = 0.1             # Weight for perceptual (VGG) loss
    adversarial_loss_weight = 0.01          # Reduced weight for adversarial loss

    # Checkpointing and logging
    save_interval = 10                       # Epoch interval for saving model checkpoints
    checkpoint_path = "experiments/checkpoints/"  # Path for model checkpoints
    log_path = "experiments/logs/"           # Path for TensorBoard or logging
    output_path = "experiments/outputs/"     # Path for generated image outputs during testing
    
    # Evaluation settings
    eval_interval = 5                        # Epoch interval for evaluation during training
    metrics_to_track = ["PSNR", "SSIM"]      # Metrics to evaluate and log
    
    # Debug settings
    print_interval = 100                     # Number of iterations after which to print training progress
