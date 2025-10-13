import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"      # 🔥 强制禁用 Triton 内核生成
os.environ["TORCH_CUDA_FUSER_DISABLE"] = "1"  # 禁用 CUDA 图优化


import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# disable triton check
try:
    torch._inductor
    print("⚙️ TorchInductor on")
except AttributeError:
    print("✅ TorchInductor off")

print("Device capability:", torch.cuda.get_device_capability())

# load vggt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
cap_major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

allocated_after_load = torch.cuda.memory_allocated(device) / 1024**2
print(f"📦 vram usage after load model: {allocated_after_load:.2f} MB")

# Load and preprocess example images (replace with your own image paths)
image_names = ["SCENE_DIR/images/train_000.png",
               "SCENE_DIR/images/train_001.png"]  

target_size = (50, 50)
images_resized = [
    Image.open(path).resize(target_size, Image.Resampling.LANCZOS)
    for path in image_names
]

images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

max_allocated_during_inference = torch.cuda.max_memory_allocated(device) / 1024**2
print(f"🚀 max vram usage while inferencing: {max_allocated_during_inference:.2f} MB")

print(predictions.keys())