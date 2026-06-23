````markdown
# Task2 ACT 跨环境泛化实验````markdown
# Task2 ACT 跨环境泛化实验

本任务基于 CALVIN / LeRobot 数据训练 ACT policy，比较不同训练环境组合在未见环境 splitD 上的泛化效果。

## 重要提醒

运行本任务前，需要先安装机器人模仿学习相关环境。本仓库只保存实验配置、日志、指标和结果，不包含完整第三方数据集与官方库源码。

需要提前准备：

- PyTorch
- LeRobot 相关环境
- CALVIN / CALVIN-LeRobot 数据相关代码
- ACT policy 训练代码所需依赖
- wandb
- numpy / tqdm / matplotlib 等常用 Python 库

数据集路径需要根据本地或服务器实际位置修改。

## 实验设置

| 模型 | 训练环境 | 评估环境 |
|---|---|---|
| ACT-B | splitB | splitD |
| ACT-ABC | splitA + splitB + splitC | splitD |

## 主要超参数

| 参数 | 数值 |
|---|---|
| Policy | ACT |
| Batch Size | 32 |
| Chunk Size | 16 |
| Max Steps | 30000 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |
| Loss | Action L1 + KL regularization |
| KL Weight | 10.0 |

## 最终结果

| 模型 | Best Step | Full-D Action L1 |
|---|---:|---:|
| ACT-B | 27000 | 0.148686 |
| ACT-ABC | 28000 | 0.128156 |

ACT-ABC 相比 ACT-B 在完整 splitD 上 Action L1 降低约 13.81%。

## 文件说明

```text
task2/
├── configs/      # 训练配置
├── logs/         # 训练与评估日志
├── metrics/      # metrics.jsonl 等指标文件
├── results/      # 最终评估结果
└── scripts/      # 训练、评估或 WandB 上传脚本
````

模型权重和完整大文件结果放在网盘中，链接见根目录 README。

```
```


本任务基于 CALVIN / LeRobot 数据训练 ACT policy，比较不同训练环境组合在未见环境 splitD 上的泛化效果。

## 重要提醒

运行本任务前，需要先安装机器人模仿学习相关环境。本仓库只保存实验配置、日志、指标和结果，不包含完整第三方数据集与官方库源码。

需要提前准备：

- PyTorch
- LeRobot 相关环境
- CALVIN / CALVIN-LeRobot 数据相关代码
- ACT policy 训练代码所需依赖
- wandb
- numpy / tqdm / matplotlib 等常用 Python 库
或参考./env/内文件
数据集路径需要根据本地或服务器实际位置修改。

## 实验设置

| 模型 | 训练环境 | 评估环境 |
|---|---|---|
| ACT-B | splitB | splitD |
| ACT-ABC | splitA + splitB + splitC | splitD |

## 主要超参数

| 参数 | 数值 |
|---|---|
| Policy | ACT |
| Batch Size | 32 |
| Chunk Size | 16 |
| Max Steps | 30000 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |
| Loss | Action L1 + KL regularization |
| KL Weight | 10.0 |

## 最终结果

| 模型 | Best Step | Full-D Action L1 |
|---|---:|---:|
| ACT-B | 27000 | 0.148686 |
| ACT-ABC | 28000 | 0.128156 |

ACT-ABC 相比 ACT-B 在完整 splitD 上 Action L1 降低约 13.81%。

## 文件说明

```text
task2/
├── code/         # 训练与评估代码
├── env/          # 环境配置参考
├── eval/         # 最终评估结果与日志
└── runs/         # 训练日志
````

模型权重和完整大文件结果放在网盘中，链接见根目录 README。

```
```
