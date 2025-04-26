import os
from data.datasets import SuperResolutionDataset
from data.transforms import get_transforms
from torch.utils.data import DataLoader

# Define test directories (update these paths if necessary)
TEST_LR_DIR = "data/lr_images/"
TEST_HR_DIR = "data/hr_images/"

def test_data_loading():
    """
    Test that SuperResolutionDataset loads LR and HR images correctly
    and that each LR image has a matching HR image.
    """
    # Get transformations for testing
    lr_transform, hr_transform = get_transforms(lr_size=(64, 64), hr_size=(128, 128))

    # Initialize the dataset with test directories and transformations
    dataset = SuperResolutionDataset(
        lr_dir=TEST_LR_DIR, 
        hr_dir=TEST_HR_DIR, 
        lr_transform=lr_transform, 
        hr_transform=hr_transform
    )

    # Assert that dataset length matches the number of LR and HR images
    assert len(dataset) > 0, "Dataset should not be empty."
    print(f"Dataset contains {len(dataset)} image pairs.")

    # Load a sample from the dataset to verify pairing and transformations
    sample_idx = 0  # Index of the sample to load
    lr_image, hr_image = dataset[sample_idx]

    # Check if images are transformed to tensors
    assert lr_image is not None and hr_image is not None, "Images should not be None."
    assert lr_image.shape[0] == 3, "LR image should have 3 color channels (RGB)."
    assert hr_image.shape[0] == 3, "HR image should have 3 color channels (RGB)."
    
    # Debug: Print dimensions of the loaded images
    print(f"Sample {sample_idx} LR image shape: {lr_image.shape}")
    print(f"Sample {sample_idx} HR image shape: {hr_image.shape}")

def test_data_loader():
    """
    Test that SuperResolutionDataset works with DataLoader and batches are loaded correctly.
    """
    # Get transformations for testing
    lr_transform, hr_transform = get_transforms(lr_size=(64, 64), hr_size=(128, 128))

    # Initialize the dataset and dataloader
    dataset = SuperResolutionDataset(
        lr_dir=TEST_LR_DIR, 
        hr_dir=TEST_HR_DIR, 
        lr_transform=lr_transform, 
        hr_transform=hr_transform
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch one batch of data
    batch = next(iter(dataloader))
    lr_images, hr_images = batch

    # Check batch dimensions
    assert lr_images.shape[0] == 4, "Batch size should be 4."
    assert hr_images.shape[0] == 4, "Batch size should be 4."
    assert lr_images.shape[1] == 3, "LR images should have 3 color channels (RGB)."
    assert hr_images.shape[1] == 3, "HR images should have 3 color channels (RGB)."
    
    # Debug: Print batch shapes for verification
    print(f"Batch LR images shape: {lr_images.shape}")
    print(f"Batch HR images shape: {hr_images.shape}")

if __name__ == "__main__":
    # Run tests
    print("Testing data loading:")
    test_data_loading()
    
    print("\nTesting DataLoader compatibility:")
    test_data_loader()
    
    print("\nAll tests completed successfully.")
