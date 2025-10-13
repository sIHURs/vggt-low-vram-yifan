import torch
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

pt_path = "output/ampere/predictions_209_1_images_pig800x800.pt"

if not os.path.exists(pt_path):
    logger.error(f"❌ File not found: {pt_path}")
    exit(1)

logger.info(f"📦 Loading predictions from {pt_path} ...")
data = torch.load(pt_path, map_location="cpu")

if isinstance(data, dict):
    logger.info(f"🔑 Top-level keys: {list(data.keys())}")
    for k, v in data.items():
        if torch.is_tensor(v):
            logger.info(f"📊 {k}: Tensor | shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            logger.info(f"📚 {k}: {type(v).__name__} | len={len(v)}")
            if len(v) > 0:
                elem = v[0]
                if torch.is_tensor(elem):
                    logger.info(f"   ↳ Example element: Tensor | shape={tuple(elem.shape)}, dtype={elem.dtype}")
                else:
                    logger.info(f"   ↳ Example element type: {type(elem).__name__}")
        elif isinstance(v, dict):
            logger.info(f"🧩 {k}: dict | {len(v)} entries")
        else:
            logger.info(f"🔸 {k}: {type(v).__name__} | value={str(v)[:80]}")
else:
    logger.warning(f"⚠️ Loaded object is not a dict (type={type(data).__name__}).")
