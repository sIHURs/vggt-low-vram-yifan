#!/bin/bash

datasets=(
  # warmup runs
  "./examples/single_cartoon/images"  # warmup run
  "./examples/room/images"  # warmup run
  "./examples/kitchen/images"  # warmup run

  # examples that comes with original repository
  "./examples/single_cartoon/images"  # 1 image
  "./examples/room/images"  # 8 images
  "./examples/kitchen/images"  # 25 images

  # larger public benchmark datasets
  "../vggt_low_vram_benchmark/360_v2_stump_images_4"  # 125 images
  "../vggt_low_vram_benchmark/tnt_family"  # 152 images
  "../vggt_low_vram_benchmark/360_v2_room_images_4"  # 311 images
  "../vggt_low_vram_benchmark/zipnerf_nyc_undistorted_images_2"  # 990 images
  "../vggt_low_vram_benchmark/imc_pt_brandenburg_gate"  # 1363 images
  "../vggt_low_vram_benchmark/zipnerf_london_undistorted_images_2"  # 1874 images
)

for dataset in "${datasets[@]}"; do
  echo "Running $dataset"
  python benchmark/benchmark.py --image_dir $dataset #--plot
  # python benchmark/benchmark_baseline.py --image_dir $dataset #--plot
  echo ""
done
