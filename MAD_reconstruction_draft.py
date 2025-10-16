import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
from PIL import Image

import argparse

import warnings
warnings.filterwarnings("ignore", message=".*bfloat16.*")

import logging

parser = argparse.ArgumentParser(description="VGGT for reconstruct the MAD dataset")
parser.add_argument(
    "--log", type=str, default="output/info_log.log", help="Path to log infomation"
)
parser.add_argument(
    "--data_path", type=str, default="data/", help="Path to dataset"
)
parser.add_argument(
    "--output", type=str, default="output/", help="Path to log infomation"
)
parser.add_argument(
    "--class_name", type=str, default=None, help="MAD dataset class name"
)

args = parser.parse_args()

# classname

if args.class_name is not None:
    classnames = [args.class_name]
else:
    classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
                "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
                "18Obesobeso", "19Bear", "20Puppy"]

# ============ Logging Setup ============

log_path = args.log
logging.basicConfig(
    format="🧭 [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, "w")     
    ]
)
logger = logging.getLogger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# ---------------------- Load Model ----------------------
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
logger.info("🚀 Loading VGGT model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
logger.info(f"🧠 Model loaded | VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


for name in classnames:
    logger.info(f"Start processing {name}")

    # Load and preprocess example images (replace with your own image paths)
    image_dir = os.path.join(args.data_path, f"{name}/train")

    image_names = [
        os.path.join(image_dir, f"train_{i:03d}.png")
        for i in range(0, 210)
        if os.path.exists(os.path.join(image_dir, f"train_{i:03d}.png"))
    ]

    #print(image_names)

    logger.info(f"📸 Found {len(image_names)} odd-numbered images in '{image_dir}'")

    # Optionally log a few example filenames
    if image_names:
        logger.info(f"🗂️  Example files: {', '.join(image_names[:3])} ...")

    # ============ Get Image Info ============
    if len(image_names) > 0:
        try:
            img = Image.open(image_names[0])
            w, h = img.size
            logger.info(f"🖼️  Image size: {w}X{h} pixels")
        except Exception as e:
            logger.warning(f"⚠️ Could not read image size: {e}")
    else:
        logger.warning("⚠️ No images found!")

    logger.info("🧩 Preprocessing images...")
    images = load_and_preprocess_images(image_names).to(device)
    logger.info(f"💾 After loading images | VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # ---------------------- Inference ----------------------
    logger.info("⚙️ Starting inference...")
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000

    logger.info(f"✅ Inference done.")
    logger.info(f"✅ Inference done in ⏱️ {elapsed_s:.2f} seconds (measured on GPU).")
    logger.info(f"📊 VRAM used (current): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.info(f"🚀 Peak VRAM during inference: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    logger.info(f"🔑 Predictions keys: {list(predictions.keys())}")
    output_path = os.path.join(args.output, f"predictions_{name}800x800.pt")
    torch.save(predictions, output_path)
    logger.info(f"💾 Saved predictions to {output_path}")

    # ---------------------- Clean up ----------------------
    torch.cuda.empty_cache()
    logger.info(f"🧹 VRAM after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
