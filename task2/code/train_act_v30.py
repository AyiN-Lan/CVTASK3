import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


def parse_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_episode_parquets(split_root):
    files = sorted((Path(split_root) / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found under {split_root}/data")
    return files


def load_action_episode_arrays(split_root):
    """Load only actions and episode_index into memory for efficient ACT chunk construction."""
    actions_all = []
    eps_all = []

    files = find_episode_parquets(split_root)

    for p in files:
        df = pd.read_parquet(p, columns=["actions", "episode_index"])

        acts = []
        for x in df["actions"].to_list():
            acts.append(np.asarray(x, dtype=np.float32))
        acts = np.stack(acts, axis=0)

        eps = np.asarray(df["episode_index"].to_list(), dtype=np.int64)

        actions_all.append(acts)
        eps_all.append(eps)

    actions = np.concatenate(actions_all, axis=0).astype(np.float32)
    episodes = np.concatenate(eps_all, axis=0).astype(np.int64)

    return actions, episodes


def compute_episode_end_indices(episodes):
    n = len(episodes)
    ends = np.empty(n, dtype=np.int64)

    start = 0
    while start < n:
        ep = episodes[start]
        end = start + 1
        while end < n and episodes[end] == ep:
            end += 1
        ends[start:end] = end
        start = end

    return ends


class ACTChunkDataset(Dataset):
    def __init__(self, split_root, repo_id, chunk_size):
        self.split_root = Path(split_root)
        self.repo_id = repo_id
        self.chunk_size = int(chunk_size)

        self.base = LeRobotDataset(
            repo_id=repo_id,
            root=str(self.split_root),
            download_videos=False,
        )

        print(f"[LOAD ACTION META] {self.split_root}")
        self.actions, self.episodes = load_action_episode_arrays(self.split_root)
        self.episode_ends = compute_episode_end_indices(self.episodes)

        if len(self.base) != len(self.actions):
            raise RuntimeError(
                f"length mismatch: LeRobotDataset={len(self.base)}, actions={len(self.actions)}"
            )

        s0 = self.base[0]["actions"].detach().cpu().numpy()
        if not np.allclose(s0, self.actions[0], atol=1e-5):
            print("[WARN] base[0].actions does not exactly match loaded actions[0]")

        print(f"[DATASET READY] frames={len(self.base)} chunk_size={self.chunk_size}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        end = int(self.episode_ends[idx])
        valid_end = min(idx + self.chunk_size, end)
        valid_len = valid_end - idx

        action_chunk = np.zeros((self.chunk_size, self.actions.shape[-1]), dtype=np.float32)
        action_is_pad = np.ones((self.chunk_size,), dtype=bool)

        if valid_len > 0:
            action_chunk[:valid_len] = self.actions[idx:valid_end]
            action_is_pad[:valid_len] = False

        return {
            "observation.images.image": sample["image"],
            "observation.images.wrist_image": sample["wrist_image"],
            "observation.state": sample["state"],
            "action": torch.from_numpy(action_chunk),
            "action_is_pad": torch.from_numpy(action_is_pad),
        }


def make_dataset(data_root, splits, repo_id, chunk_size):
    datasets = []

    for s in splits:
        split_root = Path(data_root) / s
        ds = ACTChunkDataset(
            split_root=split_root,
            repo_id=repo_id,
            chunk_size=chunk_size,
        )
        print(f"[SPLIT] {s}: frames={len(ds)}")
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_act_policy(args):
    cfg = ACTConfig(
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 200, 200),
            ),
            "observation.images.wrist_image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 84, 84),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(15,),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(7,),
            ),
        },
    )

    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.chunk_size

    cfg.vision_backbone = "resnet18"
    cfg.pretrained_backbone_weights = None
    cfg.dim_model = args.dim_model
    cfg.n_heads = args.n_heads
    cfg.dim_feedforward = args.dim_feedforward
    cfg.n_encoder_layers = args.n_encoder_layers
    cfg.n_decoder_layers = args.n_decoder_layers
    cfg.dropout = args.dropout

    cfg.use_vae = True
    cfg.latent_dim = args.latent_dim
    cfg.n_vae_encoder_layers = args.n_vae_encoder_layers
    cfg.kl_weight = args.kl_weight

    cfg.optimizer_lr = args.lr
    cfg.optimizer_weight_decay = args.weight_decay
    cfg.optimizer_lr_backbone = args.lr_backbone

    cfg.device = args.device
    cfg.use_amp = args.amp
    cfg.push_to_hub = False

    policy = ACTPolicy(cfg).to(args.device)

    return policy, cfg


@torch.no_grad()
def evaluate(policy, loader, device, max_batches):
    was_training = policy.training
    policy.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_kld = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = move_batch(batch, device)
        loss, info = policy.forward(batch)

        total_loss += float(loss.detach().cpu())
        total_l1 += float(info.get("l1_loss", 0.0))
        total_kld += float(info.get("kld_loss", 0.0))
        n_batches += 1

    if not was_training:
        policy.eval()

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "l1_loss": total_l1 / n,
        "kld_loss": total_kld / n,
    }


def save_ckpt(path, policy, optimizer, scaler, step, best_metric, cfg, args):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "best_metric": best_metric,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "act_config_repr": repr(cfg),
            "args": vars(args),
        },
        path,
    )


def maybe_init_wandb(args):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except Exception as e:
        print("[WARN] wandb import failed:", repr(e))
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
    )
    return run


def maybe_log_wandb(run, data, step):
    if run is not None:
        run.log(data, step=step)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/cv_final_task2/data/calvin-lerobot")
    parser.add_argument("--repo_id", type=str, default="xiaoma26/calvin-lerobot")

    parser.add_argument("--train_splits", type=str, required=True)
    parser.add_argument("--eval_splits", type=str, default="splitD")

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/root/autodl-tmp/cv_final_task2/runs_act")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--dim_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_vae_encoder_layers", type=int, default=4)
    parser.add_argument("--kl_weight", type=float, default=10.0)

    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--max_eval_batches", type=int, default=100)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cv_final_task2_act")

    args = parser.parse_args()

    seed_everything(args.seed)

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_splits = parse_csv(args.train_splits)
    eval_splits = parse_csv(args.eval_splits)

    print("[TRAIN SPLITS]", train_splits)
    print("[EVAL SPLITS]", eval_splits)

    train_set = make_dataset(args.data_root, train_splits, args.repo_id, args.chunk_size)
    eval_set = make_dataset(args.data_root, eval_splits, args.repo_id, args.chunk_size)

    print("[TRAIN FRAMES]", len(train_set))
    print("[EVAL FRAMES]", len(eval_set))

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    eval_loader = make_loader(eval_set, args.batch_size, False, args.num_workers)

    policy, cfg = build_act_policy(args)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.startswith("cuda")))

    wandb_run = maybe_init_wandb(args)

    train_iter = iter(train_loader)
    policy.train()

    best_l1 = float("inf")
    t0 = time.time()

    metrics_path = run_dir / "metrics.jsonl"

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = move_batch(batch, args.device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(args.amp and args.device.startswith("cuda"))):
            loss, info = policy.forward(batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss = float(loss.detach().cpu())
        train_l1 = float(info.get("l1_loss", 0.0))
        train_kld = float(info.get("kld_loss", 0.0))

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            steps_per_s = step / max(elapsed, 1e-6)

            print(
                f"[TRAIN] step={step:06d} "
                f"loss={train_loss:.6f} "
                f"l1={train_l1:.6f} "
                f"kld={train_kld:.6f} "
                f"grad_norm={float(grad_norm):.3f} "
                f"steps/s={steps_per_s:.3f}",
                flush=True,
            )

            maybe_log_wandb(
                wandb_run,
                {
                    "train/loss": train_loss,
                    "train/l1_loss": train_l1,
                    "train/kld_loss": train_kld,
                    "train/grad_norm": float(grad_norm),
                    "train/steps_per_s": steps_per_s,
                },
                step,
            )

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_metrics = evaluate(
                policy,
                eval_loader,
                args.device,
                max_batches=args.max_eval_batches,
            )

            print(
                f"[EVAL] step={step:06d} "
                f"eval_loss={eval_metrics['loss']:.6f} "
                f"eval_l1={eval_metrics['l1_loss']:.6f} "
                f"eval_kld={eval_metrics['kld_loss']:.6f}",
                flush=True,
            )

            row = {
                "step": step,
                "train_loss": train_loss,
                "train_l1_loss": train_l1,
                "train_kld_loss": train_kld,
                "eval_loss": eval_metrics["loss"],
                "eval_l1_loss": eval_metrics["l1_loss"],
                "eval_kld_loss": eval_metrics["kld_loss"],
            }

            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

            maybe_log_wandb(
                wandb_run,
                {
                    "eval/loss": eval_metrics["loss"],
                    "eval/l1_loss": eval_metrics["l1_loss"],
                    "eval/kld_loss": eval_metrics["kld_loss"],
                },
                step,
            )

            if eval_metrics["l1_loss"] < best_l1:
                best_l1 = eval_metrics["l1_loss"]
                save_ckpt(run_dir / "best.pt", policy, optimizer, scaler, step, best_l1, cfg, args)
                print(f"[SAVE] best.pt step={step} best_eval_l1={best_l1:.6f}")

        if step % args.save_every == 0 or step == args.max_steps:
            save_ckpt(run_dir / "latest.pt", policy, optimizer, scaler, step, best_l1, cfg, args)
            save_ckpt(run_dir / f"step_{step:06d}.pt", policy, optimizer, scaler, step, best_l1, cfg, args)
            print(f"[SAVE] latest.pt step={step}")

    if wandb_run is not None:
        wandb_run.finish()

    print("[DONE]")
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
