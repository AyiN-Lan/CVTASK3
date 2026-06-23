#!/usr/bin/env bash
# Object A: real-object multi-view reconstruction using COLMAP + 2DGS.
# The official 2DGS repository was used for reconstruction.
# This repository only records the command and contains our post-processing scripts.
#
# Final asset used in fusion:
# /root/autodl-tmp/cv_final/final_assets/object_A_2dgs/train/ours_30000/fuse_post.ply

cd /root/autodl-tmp/cv_final/repos/2d-gaussian-splatting

python train.py \
  -s /root/autodl-tmp/cv_final/final_assets/object_A_colmap \
  -m /root/autodl-tmp/cv_final/final_assets/object_A_2dgs \
  --iterations 30000
