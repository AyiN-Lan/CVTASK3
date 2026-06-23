#!/usr/bin/env bash
# Garden background: scene reconstruction using 2DGS.
# The official 2DGS repository was used for reconstruction.
#
# Final asset used in fusion:
# /root/autodl-tmp/cv_final/final_assets/background_garden_2dgs/train/ours_30000/fuse_post.ply

cd /root/autodl-tmp/cv_final/repos/2d-gaussian-splatting

python train.py \
  -s /root/autodl-tmp/cv_final/final_assets/background_garden_colmap \
  -m /root/autodl-tmp/cv_final/final_assets/background_garden_2dgs \
  --iterations 30000
