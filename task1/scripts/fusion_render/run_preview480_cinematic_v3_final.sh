#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/cv_final/final_assets

rm -rf /root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/preview480_cinematic_v3
mkdir -p /root/autodl-tmp/cv_final/logs

SCRIPT_PATH="/root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/final_submission/scripts/render_preview480_cinematic_v3_final.py"

TAG=preview480_cinematic_v3 \
RES_X=1280 RES_Y=720 SAMPLES=32 FRAMES=480 \
BG_UP=-1.10 BG_DEPTH=3.40 BG_TILT_X=-100 BG_SCALE=1.8 \
A_RIGHT=-2.9 B_RIGHT=-0.05 C_RIGHT=2.45 \
A_UP=0.50 B_UP=1.10 C_UP=0.36 \
A_DEPTH=2.50 B_DEPTH=7 C_DEPTH=2.50 \
A_SCALE=2.00 B_SCALE=1.2 C_SCALE=2.0 \
A_DROP_Z=1.9 B_DROP_Z=-0.15 C_DROP_Z=2.15 \
BACKDROP_DEPTH=8.0 BACKDROP_UP=0.10 BACKDROP_WIDTH=18.0 BACKDROP_HEIGHT=10.0 \
BACKDROP_R=0.40 BACKDROP_G=0.56 BACKDROP_B=0.30 \
CAM_SWAY_X=2.5 CAM_BREATH=0.4 CAM_BOB_Z=0.015 \
blender -b /root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/smoke_reframe_obja_scale2_bg_neg12_v4/strict_debug_cam_focus.blend \
-P "$SCRIPT_PATH" \
2>&1 | tee /root/autodl-tmp/cv_final/logs/preview480_cinematic_v3.log

ffmpeg -y -framerate 24 \
-i /root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/preview480_cinematic_v3/frames/frame_%04d.png \
-c:v libx264 -pix_fmt yuv420p -crf 18 \
/root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/preview480_cinematic_v3.mp4

mkdir -p /root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/final_submission

cp /root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/preview480_cinematic_v3.mp4 \
/root/autodl-tmp/cv_final/final_assets/fusion_scene_strict/final_submission/task1_final_fusion_render.mp4
