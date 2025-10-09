import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Load and preprocess example images (replace with your own image paths)
image_dir ="dataset/MAD/pig" 

image_names = [
    os.path.join(image_dir, f"{i:03d}.png")
    for i in range(1, 210)  # 001 ~ 209
    if i % 2 == 1 and os.path.exists(os.path.join(image_dir, f"{i:03d}.png"))
]

print(f"Loaded {len(image_names)} images (odd-numbered).")

images = load_and_preprocess_images(image_names).to(device)
print("image shapeL: ", images[0].shape)

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
