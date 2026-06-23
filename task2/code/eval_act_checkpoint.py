import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_act_v30 import (
    make_dataset,
    make_loader,
    build_act_policy,
    move_batch,
)


@torch.no_grad()
def eval_checkpoint(policy, loader, device, max_batches):
    policy.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_kld = 0.0
    total_batches = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        batch = move_batch(batch, device)
        loss, info = policy.forward(batch)

        total_loss += float(loss.detach().cpu())
        total_l1 += float(info.get("l1_loss", 0.0))
        total_kld += float(info.get("kld_loss", 0.0))
        total_batches += 1

    n = max(total_batches, 1)
    return {
        "num_batches": total_batches,
        "loss": total_loss / n,
        "l1_loss": total_l1 / n,
        "kld_loss": total_kld / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval_splits", default="splitD")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    run_args = SimpleNamespace(**ckpt["args"])
    run_args.device = args.device

    print("[CKPT]", ckpt_path)
    print("[STEP]", ckpt.get("step"))
    print("[BEST_METRIC_IN_CKPT]", ckpt.get("best_metric"))
    print("[TRAIN_SPLITS]", run_args.train_splits)
    print("[EVAL_SPLITS]", args.eval_splits)
    print("[CHUNK_SIZE]", run_args.chunk_size)

    eval_splits = [x.strip() for x in args.eval_splits.split(",") if x.strip()]
    eval_set = make_dataset(
        run_args.data_root,
        eval_splits,
        run_args.repo_id,
        run_args.chunk_size,
    )

    loader = make_loader(
        eval_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    policy, cfg = build_act_policy(run_args)
    missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
    print("[LOAD] missing:", missing)
    print("[LOAD] unexpected:", unexpected)

    metrics = eval_checkpoint(
        policy,
        loader,
        args.device,
        max_batches=args.max_batches,
    )

    result = {
        "ckpt": str(ckpt_path),
        "step": ckpt.get("step"),
        "best_metric_in_ckpt": ckpt.get("best_metric"),
        "train_splits": run_args.train_splits,
        "eval_splits": args.eval_splits,
        "chunk_size": run_args.chunk_size,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        **metrics,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n[RESULT]")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
