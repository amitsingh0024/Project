from torch.utils.tensorboard import SummaryWriter
import os

class Logger:
    """
    Logger class for logging training and validation metrics and images using TensorBoard.
    """
    
    def __init__(self, log_dir="experiments/logs"):
        """
        Initializes the Logger with a specified log directory.

        Parameters:
        - log_dir (str): Path to the directory where logs will be saved.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize the TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logger initialized. Writing logs to: {log_dir}")
    
    def log_metrics(self, phase, epoch, metrics):
        """
        Logs metrics (e.g., PSNR, SSIM) to TensorBoard.

        Parameters:
        - phase (str): Phase of training (e.g., 'train' or 'val').
        - epoch (int): Current epoch number.
        - metrics (dict): Dictionary of metric names and values.
        """
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{metric_name}", value, epoch)
            print(f"{phase} - {metric_name}: {value}")

    def log_loss(self, phase, epoch, loss):
        """
        Logs the loss value to TensorBoard.

        Parameters:
        - phase (str): Phase of training (e.g., 'train' or 'val').
        - epoch (int): Current epoch number.
        - loss (float): Loss value to log.
        """
        self.writer.add_scalar(f"{phase}/loss", loss, epoch)
        print(f"{phase} - Loss: {loss}")

    def log_images(self, phase, epoch, images, name="generated"):
        """
        Logs generated images to TensorBoard.

        Parameters:
        - phase (str): Phase of training (e.g., 'train' or 'val').
        - epoch (int): Current epoch number.
        - images (Tensor): Batch of images to log.
        - name (str): Name prefix for the images.
        """
        self.writer.add_images(f"{phase}/{name}", images, epoch)
        print(f"{phase} - {name} images logged at epoch {epoch}")

    def close(self):
        """
        Closes the TensorBoard SummaryWriter.
        """
        self.writer.close()
        print("Logger closed.")
