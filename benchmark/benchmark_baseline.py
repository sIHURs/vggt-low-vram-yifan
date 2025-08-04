import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from time import perf_counter
import os
from typing import List


def main(image_list: List[str], plot: bool):
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # dtype = torch.float32

    print("Loading model")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    print(f"Loading {len(image_list)} images")
    images = load_and_preprocess_images(image_list).to(device)


    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / (1024**3)
    print(f"Current VRAM usage (model weights + images): {mem:.2f} GiB")

    torch.cuda.reset_peak_memory_stats()
    time0 = perf_counter()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak inference VRAM (including model/images): {mem:.2f} GiB")
    dt = perf_counter() - time0
    print(f"Inference time: {dt:.2f} s")


    if not plot:
        return

    from plot_recon import plot_recon

    plot_recon(
        extrinsic.float().cpu().numpy()[0],
        intrinsic.float().cpu().numpy()[0],
        predictions["world_points"].float().cpu().numpy()[0],
        images.float().cpu().numpy(),
        frustum_size=0.05,
        point_subsample=5000
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    images = [os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir))]
    main(images, args.plot)

