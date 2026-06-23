````markdown
# CVTASK3 计算机视觉期末项目

本仓库为计算机视觉期末项目代码与结果整理，包含两个任务：

- Task1：三维资产生成、重建、融合与渲染
- Task2：基于 CALVIN 数据集的 ACT 跨环境泛化实验

作者：王彦卿  
学号：23307110052

## 链接

GitHub 仓库：  
https://github.com/AyiN-Lan/CVTASK3

模型权重、大文件与最终结果：  
https://pan.baidu.com/s/1iv4XtFscweJ6HPuaivkRqw?pwd=43mw  
提取码：43mw

## 目录说明

```text
CVTASK3/
├── task1/      # 三维资产生成与融合渲染
└── task2/      # ACT 策略训练、评估与指标
````

## Task1 简介

Task1 使用多种方式生成或重建三维资产：

| 资产        | 输入        | 方法            | 输出                |
| --------- | --------- | ------------- | ----------------- |
| Object A  | 拍摄视频抽帧    | COLMAP + 2DGS | fuse_post.ply     |
| Object B  | 文本 prompt | threestudio   | model.obj         |
| Object C  | 单张图片      | Magic123      | mesh.obj / albedo |
| Garden 背景 | 多视角图像     | 2DGS          | fuse_post.ply     |

最终将不同格式的 PLY 和 OBJ 资产导入 Blender，调整尺度、位置和视角后渲染融合视频。

## Task2 简介

Task2 使用 CALVIN / LeRobot 数据训练 ACT policy，并比较两种训练设置在未见环境 splitD 上的泛化效果：

| 模型      | 训练环境                     | 评估环境   | Full-D Action L1 |
| ------- | ------------------------ | ------ | ---------------- |
| ACT-B   | splitB                   | splitD | 0.148686         |
| ACT-ABC | splitA + splitB + splitC | splitD | 0.128156         |

ACT-ABC 相比 ACT-B 在完整 splitD 上 Action L1 降低约 13.81%，说明多环境训练能提升未见环境泛化能力。

## 说明

本仓库主要保存代码、脚本、日志、配置和报告文件。
体积较大的模型权重、PLY/OBJ 资产、Blender 场景和最终视频放在网盘中。

```
```
