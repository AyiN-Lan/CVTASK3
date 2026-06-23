````markdown
# Task1 三维资产生成与融合渲染

本任务包含真实物体重建、文本生成三维、单图生成三维、背景重建和最终 Blender 融合渲染。

## 重要提醒

运行本任务前，需要先分别安装对应的第三方环境。本仓库不包含完整第三方源码。

需要提前准备：

- COLMAP
- 2D Gaussian Splatting / 2DGS 环境
- threestudio 环境
- Magic123 环境
- Blender
- ffmpeg

其中 Object A 和 Garden 背景依赖 2DGS；Object B 依赖 threestudio；Object C 依赖 Magic123；最终融合视频依赖 Blender 渲染。

## 资产来源

| 资产 | 方法 | 说明 |
|---|---|---|
| Object A | COLMAP + 2DGS | 自己拍摄视频后抽帧，进行多视角重建 |
| Object B | threestudio | 由文本 prompt 生成三维网格 |
| Object C | Magic123 | 由单张图片生成三维网格 |
| Garden 背景 | 2DGS | 使用 garden 多视角图像重建背景 |

## 目录说明

```text
task1/
├── scripts/
│   ├── asset_generation/    # 资产清理、检查和预览脚本
│   └── fusion_render/       # Blender 融合渲染脚本
├── training_commands/       # 训练/生成命令记录
├── logs/                    # 渲染或运行日志
└── README_TASK1.md
````

## 大文件说明

以下大文件不放在 GitHub，而是放在网盘中：

* Object A 的 `fuse_post.ply`
* Object B 的 `model.obj`
* Object C 的 `mesh.obj`、`mesh.mtl`、`albedo.png`
* Garden 背景的 `fuse_post.ply`
* 最终 Blender 场景 `.blend`
* 最终融合视频 `.mp4`

网盘链接见根目录 README。

```
```
