"""
Run a trained BC policy in a scene.

Usage:
    python inference.py --scene airplane
    python inference.py --scene queue
    python inference.py --scene airplane --alpha 0.7
    python inference.py --scene airplane --export-json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pygame
from pygame.math import Vector2

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Scene name")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="Velocity blending factor. 1.0 = use raw policy output. "
                         "Lower values smooth toward the previous velocity. Start at "
                         "1.0 to see what the network actually does.")
parser.add_argument(
    "--hidden",
    type=int,
    default=None,
    help="Hidden width override. By default it is detected from the checkpoint.",
)
parser.add_argument(
    "--count",
    type=int,
    default=None,
    help="Number of agents to test (defaults to the scene's configured count).",
)
parser.add_argument(
    "--export-json",
    nargs="?",
    const="",
    default=None,
    metavar="PATH",
    help="Run without a window and export a web replay. If PATH is omitted, "
         "writes data/replays/<scene>.json.",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=0,
    help="Optional export safety limit. Zero runs until every agent exits.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used to build a reproducible scene (default: 42).",
)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

scene = importlib.import_module(f"scenes.{args.scene}")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scene)
ALPHA = args.alpha
EXPORTING = args.export_json is not None


class BCPolicy(nn.Module):
    """Must match training architecture exactly (incl. tanh on output)."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.trunk(x))


# Load model and normalisation stats
stats = np.load(os.path.join(DATA_DIR, "norm_stats.npy"), allow_pickle=True).item()
state_mean  = stats["state_mean"]
state_std   = stats["state_std"]
action_mean = stats["action_mean"]
action_std  = stats["action_std"]

STATE_DIM  = len(state_mean)
ACTION_DIM = len(action_mean)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    os.path.join(DATA_DIR, "bc_policy.pt"),
    map_location=device,
    weights_only=True,
)

# Older checkpoints named the Sequential module "net" instead of "trunk".
if any(key.startswith("net.") for key in checkpoint):
    checkpoint = {
        key.replace("net.", "trunk.", 1): value
        for key, value in checkpoint.items()
    }

detected_hidden = checkpoint["trunk.0.weight"].shape[0]
HIDDEN_DIM = args.hidden or detected_hidden
if HIDDEN_DIM != detected_hidden:
    raise ValueError(
        f"--hidden={HIDDEN_DIM} does not match checkpoint width "
        f"{detected_hidden}."
    )

model = BCPolicy(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(checkpoint)
model.eval()
print(f"Model loaded  (STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, "
      f"hidden={HIDDEN_DIM}, alpha={ALPHA})")

# Run
screen = None
clock = None
if not EXPORTING:
    pygame.init()
    screen = pygame.display.set_mode((scene.SCREEN_WIDTH, scene.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption(f"BC Inference: {args.scene}")

walls = scene.build_walls()
agents, meta = scene.build_agents(count=args.count)
controller = scene.SceneController(agents, meta)
initial_agent_count = len(agents)

prev_velocities = {}
pause = False
running = True
frame_count = 0
replay_frames = []

while running:
    if not EXPORTING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause

    if not pause:
        controller.update(agents)
        agents = [agent for agent in agents if scene.filter_agent(agent)]

        if agents:
            states = np.array(
                [scene.extract_state(agent, agents) for agent in agents],
                dtype=np.float32,
            )
            normalized_states = (states - state_mean) / state_std
            state_tensor = torch.from_numpy(normalized_states).to(
                device,
                non_blocking=True,
            )

            with torch.no_grad():
                normalized_actions = model(state_tensor).cpu().numpy()

            predictions = normalized_actions * action_std + action_mean

            for agent, prediction in zip(agents, predictions):
                previous = prev_velocities.get(agent.id, prediction)
                blended = ALPHA * prediction + (1.0 - ALPHA) * previous
                prev_velocities[agent.id] = blended

                velocity = Vector2(float(blended[0]), float(blended[1]))

                # Match the stationary threshold used by train_bc.py.
                if velocity.length() < 0.5:
                    velocity = Vector2(0, 0)
                if velocity.length() > agent.max_speed:
                    velocity = velocity.normalize() * agent.max_speed

                agent.velocity = velocity
                agent.position += agent.velocity * scene.TIMESTEP

        for agent in agents:
            for wall in walls:
                wall.collision_resolve(agent)

        frame_count += 1

        if EXPORTING:
            replay_frames.append([
                [
                    agent.id,
                    round(agent.position.x, 3),
                    round(agent.position.y, 3),
                    round(agent.radius, 3),
                ]
                for agent in agents
            ])
            reached_safety_limit = (
                args.max_frames > 0 and frame_count >= args.max_frames
            )
            if not agents or reached_safety_limit:
                running = False

    if not EXPORTING:
        screen.fill((30, 30, 30))
        for wall in walls:
            pygame.draw.rect(screen, (200, 200, 200), wall.rect)
        for agent in agents:
            pygame.draw.circle(
                screen,
                (255, 100, 0),
                (int(agent.position.x), int(agent.position.y)),
                int(agent.radius),
            )
        pygame.display.flip()
        clock.tick(60)

if EXPORTING:
    output_path = args.export_json
    using_default_path = not output_path
    if not output_path:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(
            repo_root,
            "data",
            "replays",
            f"{args.scene}-{initial_agent_count}.json",
        )

    complete = len(agents) == 0
    replay = {
        "version": 1,
        "scene": args.scene,
        "agent_count": initial_agent_count,
        "width": scene.SCREEN_WIDTH,
        "height": scene.SCREEN_HEIGHT,
        "fps": 60,
        "complete": complete,
        "walls": [
            [wall.rect.left, wall.rect.top, wall.rect.width, wall.rect.height]
            for wall in walls
        ],
        "frames": replay_frames,
    }

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(replay, output_file, separators=(",", ":"))

    if using_default_path and complete:
        manifest_path = os.path.join(os.path.dirname(output_path), "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        else:
            manifest = {"version": 1, "replays": []}

        entry = {
            "scene": args.scene,
            "count": initial_agent_count,
            "file": os.path.basename(output_path),
            "frames": len(replay_frames),
        }
        manifest["replays"] = [
            item
            for item in manifest["replays"]
            if (item["scene"], item["count"])
            != (entry["scene"], entry["count"])
        ]
        manifest["replays"].append(entry)
        manifest["replays"].sort(key=lambda item: (item["scene"], item["count"]))

        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")

    status = "complete" if complete else f"incomplete ({len(agents)} agents remain)"
    print(
        f"Exported {len(replay_frames)} frames to {output_path} [{status}]"
    )
    if not complete:
        raise SystemExit(
            "Replay hit --max-frames before every agent exited; "
            "increase the limit or inspect the stalled policy."
        )
else:
    pygame.quit()
