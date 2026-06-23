import gc
import json
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
import lerobot.policies.act.modeling_act as act_m
from lerobot.configs.types import FeatureType, PolicyFeature


ROOT = Path("/root/autodl-tmp/cv_final_task2/data/calvin-lerobot")
SPLIT = "splitB"
SPLIT_ROOT = ROOT / SPLIT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def short_tb():
    return "".join(traceback.format_exc().splitlines(True)[-18:])


def load_raw_batch():
    ds = LeRobotDataset(
        repo_id="xiaoma26/calvin-lerobot",
        root=str(SPLIT_ROOT),
        download_videos=False,
    )

    def collate_fn(batch):
        return {
            "image": torch.stack([b["image"] for b in batch], 0),
            "wrist_image": torch.stack([b["wrist_image"] for b in batch], 0),
            "state": torch.stack([b["state"] for b in batch], 0),
            "actions": torch.stack([b["actions"] for b in batch], 0),
        }

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    return next(iter(loader))


def make_feature(key):
    if "wrist" in key:
        return PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
    if "image" in key:
        return PolicyFeature(type=FeatureType.VISUAL, shape=(3, 200, 200))
    if "state" in key:
        return PolicyFeature(type=FeatureType.STATE, shape=(15,))
    raise ValueError(f"unknown input feature key: {key}")


def make_cfg(input_keys, output_key):
    input_features = {k: make_feature(k) for k in input_keys}

    cfg = ACTConfig(
        input_features=input_features,
        output_features={
            output_key: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )

    cfg.pretrained_backbone_weights = None
    cfg.device = DEVICE
    cfg.use_amp = False
    cfg.push_to_hub = False
    return cfg


def map_raw_to_batch(raw, batch_keys, action_key="action", include_pad=True, chunk_size=100):
    batch = {}

    for k in batch_keys:
        if k.endswith("wrist_image") or "wrist" in k:
            batch[k] = raw["wrist_image"].to(DEVICE)
        elif k.endswith("image") or "image" in k:
            batch[k] = raw["image"].to(DEVICE)
        elif k.endswith("state") or "state" in k:
            batch[k] = raw["state"].to(DEVICE)
        else:
            raise ValueError(f"cannot map key: {k}")

    action_chunk = raw["actions"].unsqueeze(1).repeat(1, chunk_size, 1).to(DEVICE)
    batch[action_key] = action_chunk

    if include_pad:
        batch["action_is_pad"] = torch.zeros(
            action_chunk.shape[0],
            action_chunk.shape[1],
            dtype=torch.bool,
            device=DEVICE,
        )

    return batch


def summarize_out(out):
    print("forward output type:", type(out))

    if isinstance(out, tuple):
        print("tuple len:", len(out))
        for i, x in enumerate(out):
            print(f"  tuple[{i}]:", type(x))
            if torch.is_tensor(x):
                print("    tensor shape:", tuple(x.shape), "mean:", x.detach().float().mean().item())
            elif isinstance(x, dict):
                print("    dict keys:", list(x.keys()))
                for kk, vv in x.items():
                    if torch.is_tensor(vv):
                        print("     ", kk, tuple(vv.shape), vv.detach().float().mean().item())
                    else:
                        print("     ", kk, type(vv), vv)
            else:
                print("   ", x)

    elif isinstance(out, dict):
        print("dict keys:", list(out.keys()))
        for kk, vv in out.items():
            print(" ", kk, type(vv), getattr(vv, "shape", None))
    else:
        print(out)


def try_case(raw, name, input_keys, output_key, batch_action_key, include_pad):
    print_section(f"TRY CASE: {name}")

    print("input_keys:", input_keys)
    print("output_key:", output_key)
    print("batch_action_key:", batch_action_key)
    print("include_pad:", include_pad)

    policy = None

    try:
        cfg = make_cfg(input_keys, output_key)

        print("\nACTConfig derived features:")
        print(" state_feature:", getattr(cfg, "state_feature", None))
        print(" action_feature:", getattr(cfg, "action_feature", None))
        print(" image_features:", getattr(cfg, "image_features", None))

        policy = ACTPolicy(cfg).to(DEVICE)
        policy.train()

        batch = map_raw_to_batch(
            raw,
            batch_keys=input_keys,
            action_key=batch_action_key,
            include_pad=include_pad,
            chunk_size=cfg.chunk_size,
        )

        print("\nforward batch keys/shapes:")
        for k, v in batch.items():
            print(" ", k, tuple(v.shape), v.dtype, v.device)

        out = policy.forward(batch)
        print("\n[FORWARD OK]")
        summarize_out(out)

        print("\nselect_action smoke test:")
        policy.eval()
        policy.reset()

        obs_batch = {k: batch[k] for k in input_keys}
        with torch.no_grad():
            action = policy.select_action(obs_batch)

        print("[SELECT_ACTION OK]", type(action), tuple(action.shape), action.dtype, action.device)

        return True

    except Exception:
        print("[FAIL]")
        print(short_tb())
        return False

    finally:
        if policy is not None:
            del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print_section("ENV")
    print("lerobot file:", lerobot.__file__)
    try:
        import importlib.metadata as md
        print("lerobot version:", md.version("lerobot"))
    except Exception as e:
        print("version unknown:", repr(e))
    print("device:", DEVICE)

    print_section("DATASET INFO")
    info = json.loads((SPLIT_ROOT / "meta" / "info.json").read_text())
    print("split:", SPLIT)
    print("codebase_version:", info.get("codebase_version"))
    print("total_episodes:", info.get("total_episodes"))
    print("total_frames:", info.get("total_frames"))
    print("feature keys:", list(info.get("features", {}).keys()))

    raw = load_raw_batch()

    print("\nraw dataset batch:")
    for k, v in raw.items():
        print(" ", k, tuple(v.shape), v.dtype)

    print_section("MODELING_ACT CONSTANTS")
    for k, v in vars(act_m).items():
        if (
            k.isupper()
            or "OBS" in k
            or "ACTION" in k
            or "STATE" in k
            or "IMAGE" in k
            or "PAD" in k
        ):
            if isinstance(v, (str, int, float, bool, tuple, list, dict, type(None))):
                print(k, "=", repr(v))

    print_section("ACTConfig DEFAULT FEATURE PROPERTIES")
    empty_cfg = ACTConfig()
    print("default input_features:", empty_cfg.input_features)
    print("default output_features:", empty_cfg.output_features)
    for attr in ["state_feature", "action_feature", "image_features"]:
        try:
            print(attr, "=", getattr(empty_cfg, attr))
        except Exception as e:
            print(attr, "ERROR:", repr(e))

    cases = [
        {
            "name": "standard_two_images_state_action_pad",
            "input_keys": [
                "observation.images.image",
                "observation.images.wrist_image",
                "observation.state",
            ],
            "output_key": "action",
            "batch_action_key": "action",
            "include_pad": True,
        },
        {
            "name": "standard_two_images_state_action_no_pad",
            "input_keys": [
                "observation.images.image",
                "observation.images.wrist_image",
                "observation.state",
            ],
            "output_key": "action",
            "batch_action_key": "action",
            "include_pad": False,
        },
        {
            "name": "dataset_keys_output_action",
            "input_keys": [
                "image",
                "wrist_image",
                "state",
            ],
            "output_key": "action",
            "batch_action_key": "action",
            "include_pad": True,
        },
        {
            "name": "plain_observation_keys_output_action",
            "input_keys": [
                "observation.image",
                "observation.wrist_image",
                "observation.state",
            ],
            "output_key": "action",
            "batch_action_key": "action",
            "include_pad": True,
        },
        {
            "name": "standard_input_output_actions_plural",
            "input_keys": [
                "observation.images.image",
                "observation.images.wrist_image",
                "observation.state",
            ],
            "output_key": "actions",
            "batch_action_key": "actions",
            "include_pad": True,
        },
        {
            "name": "state_only_action",
            "input_keys": [
                "observation.state",
            ],
            "output_key": "action",
            "batch_action_key": "action",
            "include_pad": True,
        },
        {
            "name": "main_image_state_only_action",
            "input_keys": [
                "observation.images.image",
                "observation.state",
            ],
            "output_key": "action",
            "batch_action_key": "action",
            "include_pad": True,
        },
    ]

    results = {}

    for case in cases:
        ok = try_case(raw, **case)
        results[case["name"]] = ok

    print_section("SUMMARY")
    for name, ok in results.items():
        print(f"{name}: {'OK' if ok else 'FAIL'}")

    ok_cases = [name for name, ok in results.items() if ok]
    print("\nOK_CASES:", ok_cases)

    if ok_cases:
        print("\nRecommended profile:", ok_cases[0])
    else:
        print("\nNo working profile found. Need to inspect ACT source around failing key usage.")


if __name__ == "__main__":
    main()
