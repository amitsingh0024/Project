import torch
from models.generator import CustomGenerator
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained generator model
generator = CustomGenerator(in_channels=3, upscale_factor=2).to(device)
generator.load_state_dict(torch.load("final_generator_model.pth"))
generator.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to low-resolution input size used during training
    transforms.ToTensor(),        # Convert to tensor
])

# Load a test image (low-resolution input)
# Replace 'path_to_image.jpg' with the path to your test low-resolution image
lr_image_path = "data/hr_images/0001.png"
lr_image = Image.open(lr_image_path).convert("RGB")
lr_image = transform(lr_image).unsqueeze(0).to(device)  # Add batch dimension and send to device

# Generate high-resolution image
with torch.no_grad():  # Disable gradient calculation for inference
    sr_image = generator(lr_image)

# Post-process the output (convert to image format)
sr_image = sr_image.squeeze(0).cpu().clamp(0, 1)  # Remove batch dimension and clamp values
sr_image = transforms.ToPILImage()(sr_image)  # Convert to PIL image for visualization

# Display the low-resolution and super-resolution images side-by-side
plt.figure(figsize=(12, 6))

# Display low-resolution input
plt.subplot(1, 2, 1)
plt.title("Low-Resolution Input")
plt.imshow(lr_image.squeeze(0).permute(1, 2, 0).cpu())  # Permute to convert to HWC format
plt.axis("off")

# Display generated high-resolution output
plt.subplot(1, 2, 2)
plt.title("Super-Resolution Output")
plt.imshow(sr_image)
plt.axis("off")

plt.show()

# Save the generated high-resolution image in experiments/outputs/
output_dir = "experiments/outputs"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
output_path = os.path.join(output_dir, "generated_sr_image.jpg")
sr_image.save(output_path)
print(f"Generated high-resolution image saved as '{output_path}'")
