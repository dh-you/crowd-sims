"""Lean Stage A gate for the autonomous airplane expert and BC policy."""

import argparse
import csv
import os
import random
import shutil
import sys

import numpy as np
import torch
from pygame.math import Vector2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.physics import advance_velocity_agents
from scenes import airplane as scene
from train_bc import BCPolicy, split_indices


CSV_FIELDS = (
    "variant",
    "seed",
    "count",
    "complete",
    "frames",
    "contacts",
    "max_penetration",
    "max_row_drift",
    "unresolved_overlaps",
    "stalled",
)
MERGE_ZONE_TOLERANCE = 0.2


def checkpoint_hidden_dims(checkpoint):
    weight_keys = sorted(
        (
            key
            for key in checkpoint
            if key.startswith("trunk.") and key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    return tuple(checkpoint[key].shape[0] for key in weight_keys[:-1])


def load_candidate(device):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airplane")
    data = np.load(os.path.join(data_dir, "expert_data.candidate.npz"))
    stats = np.load(
        os.path.join(data_dir, "norm_stats.candidate.npy"),
        allow_pickle=True,
    ).item()
    checkpoint = torch.load(
        os.path.join(data_dir, "bc_policy.candidate.pt"),
        map_location=device,
        weights_only=True,
    )
    hidden_dims = checkpoint_hidden_dims(checkpoint)
    model = BCPolicy(scene.STATE_DIM, scene.ACTION_DIM, hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return data_dir, data, stats, model


def validate_dataset(data, stats):
    states = data["states"]
    episode_ids = data["episode_ids"]
    if states.ndim != 2 or states.shape[1] != scene.STATE_DIM:
        raise AssertionError(f"expected four-input states, got {states.shape}")
    if "phases" in data.files:
        raise AssertionError("autonomous airplane data must not contain phases")
    if np.any(states[:, 0] < -1.0) or np.any(states[:, 0] > 1.0):
        raise AssertionError("aisle offsets are outside [-1, 1]")
    if np.any(states[:, 1:] < 0.0) or np.any(states[:, 1:] > 1.0):
        raise AssertionError("clearances are outside [0, 1]")
    if not np.array_equal(stats["state_mean"], np.zeros(scene.STATE_DIM)):
        raise AssertionError("airplane state normalization mean must be identity")
    if not np.array_equal(stats["state_std"], np.ones(scene.STATE_DIM)):
        raise AssertionError("airplane state normalization scale must be identity")

    train_indices, val_indices, _ = split_indices(
        episode_ids,
        len(states),
        0.2,
        42,
    )
    overlap = np.intersect1d(
        np.unique(episode_ids[train_indices]),
        np.unique(episode_ids[val_indices]),
    )
    if overlap.size:
        raise AssertionError("training and validation episodes overlap")


def policy_actions(model, stats, device, states, force_row_clear=False):
    inputs = np.array(states, dtype=np.float32, copy=True)
    if force_row_clear:
        inputs[:, 1] = 1.0
    normalized = (inputs - stats["state_mean"]) / stats["state_std"]
    with torch.no_grad():
        outputs = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
    predictions = outputs * stats["action_std"] + stats["action_mean"]
    stop_threshold = getattr(scene, "POLICY_STOP_THRESHOLD", 0.5)
    predictions[np.linalg.norm(predictions, axis=1) < stop_threshold] = 0.0
    return predictions


def rollout(variant, count, seed, max_frames, model=None, stats=None, device=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    walls = scene.build_walls()
    agents, _ = scene.build_agents(count=count)
    rng = random.Random(seed)
    for agent in agents:
        agent.position += Vector2(rng.uniform(-1, 1), rng.uniform(-1, 1))

    contacts = 0
    max_penetration = 0.0
    max_row_drift = 0.0
    unresolved_overlaps = 0
    last_motion_frame = 0
    stalled = False
    entered_aisle = set()

    for frame in range(1, max_frames + 1):
        states = np.asarray(
            [scene.extract_state(agent, agents) for agent in agents],
            dtype=np.float32,
        )
        if variant == "expert":
            actions = [scene.expert_action(state) for state in states]
        else:
            actions = policy_actions(
                model,
                stats,
                device,
                states,
                force_row_clear=variant == "row_clearance_1",
            )

        before = [Vector2(agent.position) for agent in agents]
        diagnostics = advance_velocity_agents(
            agents,
            actions,
            walls,
            scene.TIMESTEP,
        )
        contacts += diagnostics["contacts"]
        max_penetration = max(
            max_penetration,
            diagnostics["max_penetration"],
        )
        unresolved_overlaps = max(
            unresolved_overlaps,
            diagnostics["unresolved_overlaps"],
        )

        motion = sum(
            old.distance_to(agent.position)
            for old, agent in zip(before, agents)
        )
        if motion > 1e-3:
            last_motion_frame = frame

        for agent in agents:
            inside_merge_zone = (
                abs(agent.position.x - scene.AISLE_X)
                <= scene.AISLE_HALF_WIDTH + agent.radius + MERGE_ZONE_TOLERANCE
            )
            if inside_merge_zone:
                entered_aisle.add(agent.id)
            elif agent.id not in entered_aisle:
                max_row_drift = max(
                    max_row_drift,
                    abs(agent.position.y - agent.getData("start_pos").y),
                )

        agents = [agent for agent in agents if scene.filter_agent(agent)]
        if not agents:
            break
        if frame - last_motion_frame >= 1000:
            stalled = True
            break

    return {
        "variant": variant,
        "seed": seed,
        "count": count,
        "complete": not agents,
        "frames": frame,
        "contacts": contacts,
        "max_penetration": round(max_penetration, 6),
        "max_row_drift": round(max_row_drift, 6),
        "unresolved_overlaps": unresolved_overlaps,
        "stalled": stalled,
    }


def row_passes_gate(row, max_frames, max_row_drift):
    return (
        row["complete"]
        and row["frames"] <= max_frames
        and row["unresolved_overlaps"] == 0
        and row["max_row_drift"] <= max_row_drift
        and not row["stalled"]
    )


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def promote(data_dir):
    for stem, extension in (
        ("expert_data", ".npz"),
        ("norm_stats", ".npy"),
        ("bc_policy", ".pt"),
    ):
        source = os.path.join(data_dir, f"{stem}.candidate{extension}")
        destination = os.path.join(data_dir, f"{stem}{extension}")
        temporary = destination + ".promoting"
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    print("Candidate artifacts promoted to production.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="24,60,102")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument("--max-row-drift", type=float, default=6.0)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    counts = [int(value) for value in args.counts.split(",")]
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airplane")
    csv_path = args.csv or os.path.join(data_dir, "evaluation.csv")

    model = stats = None
    variants = ("expert",) if args.expert_only else ("normal", "row_clearance_1")
    if not args.expert_only:
        data_dir, data, stats, model = load_candidate(device)
        validate_dataset(data, stats)

    rows = []
    failures = []
    for variant in variants:
        for count in counts:
            for seed in range(args.seeds):
                row = rollout(
                    variant,
                    count,
                    seed,
                    args.max_frames,
                    model,
                    stats,
                    device,
                )
                rows.append(row)
                print(
                    f"{variant:16s} count={count:3d} seed={seed} "
                    f"complete={row['complete']} frames={row['frames']:5d} "
                    f"contacts={row['contacts']:5d} "
                    f"penetration={row['max_penetration']:.4f} "
                    f"drift={row['max_row_drift']:.4f} "
                    f"unresolved={row['unresolved_overlaps']} "
                    f"stalled={row['stalled']}"
                )
                if variant in ("expert", "normal") and not row_passes_gate(
                    row,
                    args.max_frames,
                    args.max_row_drift,
                ):
                    failures.append(row)

    write_csv(csv_path, rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    if failures:
        raise SystemExit(f"Gate failed for {len(failures)} rollout(s)")

    print("Required rollout variants passed the Stage A gate.")
    if args.promote:
        if args.expert_only:
            raise SystemExit("--promote requires candidate policy evaluation")
        promote(data_dir)


if __name__ == "__main__":
    main()
