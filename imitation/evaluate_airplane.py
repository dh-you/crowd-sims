"""Headless acceptance test and gated promotion for an airplane BC candidate."""

import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
from pygame.math import Vector2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenes import airplane as scene
from train_bc import BCPolicy, split_indices


def load_candidate(device):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airplane")
    data = np.load(os.path.join(data_dir, "expert_data.candidate.npz"))
    stats = np.load(
        os.path.join(data_dir, "norm_stats.candidate.npy"), allow_pickle=True
    ).item()
    checkpoint = torch.load(
        os.path.join(data_dir, "bc_policy.candidate.pt"),
        map_location=device,
        weights_only=True,
    )
    hidden = checkpoint["trunk.0.weight"].shape[0]
    model = BCPolicy(scene.STATE_DIM, scene.ACTION_DIM, hidden).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return data_dir, data, stats, model


def validate_dataset(data):
    states = data["states"]
    actions = data["actions"]
    phases = data["phases"]
    episode_ids = data["episode_ids"]
    waiting = phases == scene.PHASE_WAITING
    to_aisle = phases == scene.PHASE_TO_AISLE
    waiting_max = float(np.abs(actions[waiting]).max(initial=0.0))
    aisle_vertical_max = float(np.abs(actions[to_aisle, 1]).max(initial=0.0))
    train_indices, val_indices, _ = split_indices(episode_ids, len(states), 0.2, 42)
    overlap = np.intersect1d(
        np.unique(episode_ids[train_indices]),
        np.unique(episode_ids[val_indices]),
    )
    print(
        f"Dataset: samples={len(states)}, episodes={len(np.unique(episode_ids))}, "
        f"waiting_action_max={waiting_max:.8f}, "
        f"aisle_vertical_max={aisle_vertical_max:.8f}"
    )
    if waiting_max > 1e-6:
        raise AssertionError("WAITING expert actions are not zero")
    if aisle_vertical_max > 1e-6:
        raise AssertionError("TO_AISLE expert vertical actions are not zero")
    if overlap.size:
        raise AssertionError("training and validation episodes overlap")


def report_open_loop(data, stats, model, device):
    states = data["states"]
    phases = data["phases"]
    predictions = []
    for offset in range(0, len(states), 65536):
        normalized = (states[offset:offset + 65536] - stats["state_mean"]) / stats[
            "state_std"
        ]
        with torch.no_grad():
            output = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        predictions.append(output * stats["action_std"] + stats["action_mean"])
    predictions = np.concatenate(predictions)
    waiting = phases == scene.PHASE_WAITING
    to_aisle = phases == scene.PHASE_TO_AISLE
    waiting_speeds = np.linalg.norm(predictions[waiting], axis=1)
    aisle_vertical = np.abs(predictions[to_aisle, 1])
    print(
        "Open loop: "
        f"waiting_mean_speed={waiting_speeds.mean():.5f}, "
        f"waiting_false_motion={np.mean(waiting_speeds >= 0.5):.3%}, "
        f"aisle_vertical_mean={aisle_vertical.mean():.5f}, "
        f"aisle_vertical_p99={np.percentile(aisle_vertical, 99):.5f}"
    )


def rollout(model, stats, device, count, seed, max_frames):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    walls = scene.build_walls()
    agents, meta = scene.build_agents(count=count)
    controller = scene.SceneController(agents, meta)
    initial_agents = list(agents)
    entered_aisle = set()
    max_row_drift = 0.0
    worst_agent = None

    for frame in range(1, max_frames + 1):
        controller.update(agents)
        agents = [agent for agent in agents if scene.filter_agent(agent)]
        if not agents:
            return {
                "complete": True,
                "frames": frame - 1,
                "max_row_drift": max_row_drift,
                "worst_agent": worst_agent,
            }

        for agent in agents:
            if agent.id not in entered_aisle:
                drift = abs(agent.position.y - agent.getData("start_pos").y)
                if drift > max_row_drift:
                    max_row_drift = drift
                    worst_agent = agent.id
                if scene.training_phase(agent) == scene.PHASE_EXITING:
                    entered_aisle.add(agent.id)

        states = np.array(
            [scene.extract_state(agent, agents) for agent in agents], dtype=np.float32
        )
        normalized = (states - stats["state_mean"]) / stats["state_std"]
        with torch.no_grad():
            normalized_actions = model(
                torch.from_numpy(normalized).to(device)
            ).cpu().numpy()
        predictions = normalized_actions * stats["action_std"] + stats["action_mean"]

        for agent, prediction in zip(agents, predictions):
            velocity = Vector2(float(prediction[0]), float(prediction[1]))
            if velocity.length() < 0.5:
                velocity.update(0, 0)
            if velocity.length() > agent.max_speed:
                velocity.scale_to_length(agent.max_speed)
            agent.velocity = velocity
            agent.position += velocity * scene.TIMESTEP

        for agent in agents:
            for wall in walls:
                wall.collision_resolve(agent)

        for agent in agents:
            if agent.id not in entered_aisle:
                drift = abs(agent.position.y - agent.getData("start_pos").y)
                if drift > max_row_drift:
                    max_row_drift = drift
                    worst_agent = agent.id

    remaining = {agent.id for agent in initial_agents if agent in agents}
    return {
        "complete": False,
        "frames": max_frames,
        "max_row_drift": max_row_drift,
        "worst_agent": worst_agent,
        "remaining": sorted(remaining),
    }


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
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument("--max-row-drift", type=float, default=1.0)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="CPU is faster for the tiny per-frame policy batches.")
    args = parser.parse_args()
    counts = [int(value) for value in args.counts.split(",")]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device(args.device)
    data_dir, data, stats, model = load_candidate(device)
    validate_dataset(data)
    report_open_loop(data, stats, model, device)

    failures = []
    for count in counts:
        for seed in range(args.seeds):
            result = rollout(model, stats, device, count, seed, args.max_frames)
            print(
                f"count={count:3d} seed={seed:2d} frames={result['frames']:5d} "
                f"complete={result['complete']} drift={result['max_row_drift']:.4f} "
                f"worst_agent={result['worst_agent']}"
            )
            if not result["complete"] or result["max_row_drift"] > args.max_row_drift:
                failures.append((count, seed, result))

    if failures:
        raise SystemExit(
            f"Candidate rejected: {len(failures)} rollout(s) failed completion or "
            f"the {args.max_row_drift}px seat-row limit. Production artifacts unchanged."
        )
    print("Candidate passed every rollout acceptance check.")
    if args.promote:
        promote(data_dir)


if __name__ == "__main__":
    main()
