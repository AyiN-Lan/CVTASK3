# Task1: 3D Asset Generation and Fusion Rendering

## Overview

This task constructs a fused 3D scene using three foreground objects and one garden background.

- Object A: real multi-view object reconstructed using COLMAP + 2DGS.
- Object B: text-to-3D object generated using threestudio.
- Object C: single-image-to-3D object generated using Magic123.
- Background: garden scene reconstructed using 2DGS.

The final heterogeneous assets were imported into Blender and rendered as a multi-view roaming video.

## Training / Generation

The original training and generation were performed by calling the official implementations of 2DGS, threestudio, and Magic123. The full third-party repositories and large intermediate training outputs are not included in this submission.

Command records are provided in:

```text
scripts/training_commands/
