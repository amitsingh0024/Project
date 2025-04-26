import torch
import os

def save_checkpoint(model, optimizer, epoch, checkpoint_dir="experiments/checkpoints", filename="checkpoint.pth"):
    """
    Saves a model checkpoint to a specified directory.

    Parameters:
    - model (torch.nn.Module): Model to save.
    - optimizer (torch.optim.Optimizer): Optimizer state to save.
    - epoch (int): Current epoch number.
    - checkpoint_dir (str): Directory to save the checkpoint.
    - filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}' for epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Loads a model checkpoint from a specified file.

    Parameters:
    - model (torch.nn.Module): Model to load the state into.
    - optimizer (torch.optim.Optimizer): Optimizer to load the state into.
    - checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
    - int: The epoch at which the checkpoint was saved.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    print(f"Checkpoint loaded from '{checkpoint_path}' at epoch {epoch}")
    return epoch
