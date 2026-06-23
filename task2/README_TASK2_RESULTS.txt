Task2: LeRobot ACT cross-environment generalization

Experiments:
1. ACT-B
   Train: splitB
   Test: splitD
   Best checkpoint: weights/act_B_best.pt

2. ACT-ABC
   Train: splitA, splitB, splitC
   Test: splitD
   Best checkpoint: weights/act_ABC_best.pt

The model weights are provided in the cloud storage link:
- act_B_best.pt: best checkpoint of ACT-B
- act_ABC_best.pt: best checkpoint of ACT-ABC

Shared hyperparameters:
- Policy: LeRobot ACT
- chunk_size: 16
- batch_size: 32
- max_steps: 30000
- lr: 1e-5
- lr_backbone: 1e-5
- weight_decay: 1e-4
- kl_weight: 10.0
- dim_model: 512
- n_heads: 8
- n_encoder_layers: 4
- n_decoder_layers: 1

Full-D evaluation summary:
See eval/final_fullD_summary.json

Key result:
ACT-ABC improves full-D Action L1 over ACT-B.
