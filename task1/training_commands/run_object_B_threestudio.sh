#!/usr/bin/env bash
# Object B: text-to-3D generation using threestudio.
# The official threestudio repository was used.
# This repository only records the command and contains our visualization/fusion scripts.
#
# Final exported asset:
# /root/autodl-tmp/cv_final/final_assets/object_B_threestudio/model.obj

cd /root/autodl-tmp/cv_final/repos/threestudio

python launch.py \
  --config configs/stable-dreamfusion.yaml \
  --train \
  system.prompt_processor.prompt="a 3D object" \
  trial_name=object_B_threestudio
