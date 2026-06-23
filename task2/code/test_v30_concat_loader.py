import torch
from torch.utils.data import ConcatDataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = "/root/autodl-tmp/cv_final_task2/data/calvin-lerobot"

datasets = []
for s in ["splitA", "splitB", "splitC", "splitD"]:
    ds = LeRobotDataset(
        repo_id="xiaoma26/calvin-lerobot",
        root=f"{ROOT}/{s}",
        download_videos=False,
    )
    print(s, len(ds))
    datasets.append(ds)

full = ConcatDataset(datasets)
print("total:", len(full))

def collate_fn(batch):
    out = {}
    for k in ["image", "wrist_image", "state", "actions"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

loader = DataLoader(
    full,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

batch = next(iter(loader))
for k, v in batch.items():
    print(k, v.shape, v.dtype)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

print("device:", device)
print("cuda batch ok:")
for k, v in batch.items():
    print(k, v.shape, v.device)
