#!/usr/bin/env bash
# Object C: single-image-to-3D generation using Magic123.
# The official Magic123 repository was used.
# The input image was background-removed before optimization.
#
# Final exported assets:
# /root/autodl-tmp/cv_final/final_assets/object_C_magic123/mesh.obj
# /root/autodl-tmp/cv_final/final_assets/object_C_magic123/mesh.mtl
# /root/autodl-tmp/cv_final/final_assets/object_C_magic123/albedo.png

cd /root/autodl-tmp/cv_final/repos/Magic123

python launch.py \
  --config configs/magic123.yaml \
  --train \
  data.image_path=/root/autodl-tmp/cv_final/final_assets/object_C_input/input_rgba.png \
  trial_name=object_C_magic123
