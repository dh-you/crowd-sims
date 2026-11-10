"""
Run a trained BC policy in a scene.

Usage:
    python inference.py --scene airplane
    python inference.py --scene queue
    python inference.py --scene airplane --alpha 0.7
    python inference.py --scene airplane --candidate
    python inference.py --scene airplane --export-json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import inspect
import math
import random
import numpy as np
import torch
import torch.nn as nn
import pygame
from pygame.math import Vector2

from core.physics import advance_velocity_agents
from replay_export import capture_frame, write_web_replay

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
         "writes data/replays/<scene>-<layout>-<count>.json for airplanes "
         "or data/replays/<scene>-<count>.json for other scenes.",
)
parser.add_argument(
    "--frame-stride", type=int, default=1,
    help="Record every N simulation steps while preserving playback duration.",
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
parser.add_argument(
    "--device",
    choices=("cpu", "cuda", "auto"),
    default="cpu",
    help="Inference device. CPU is fastest for these small per-frame batches.",
)
parser.add_argument(
    "--candidate",
    action="store_true",
    help="Load unpromoted candidate artifacts for visual inspection.",
)
parser.add_argument(
    "--artifact-tag",
    default=None,
    help="Load a tagged candidate such as 'heterogeneous'; requires --candidate.",
)
parser.add_argument(
    "--layout",
    default=None,
    help="Airplane layout such as 3x3, 3x4, or 3x4x3 (airplane scene only).",
)
parser.add_argument(
    "--occupancy",
    type=float,
    default=None,
    help="Seeded airplane occupancy fraction in (0, 1]; cannot be combined with --count.",
)
parser.add_argument("--preferred-speed-min", type=float, default=None,
                    help="Airplane preferred-speed lower bound; provide both bounds.")
parser.add_argument("--preferred-speed-max", type=float, default=None,
                    help="Airplane preferred-speed upper bound; provide both bounds.")
args = parser.parse_args()
if args.artifact_tag and not args.candidate:
    parser.error("--artifact-tag requires --candidate")
if args.frame_stride < 1:
    parser.error("--frame-stride must be at least 1")
if args.max_frames < 0:
    parser.error("--max-frames must be nonnegative")
speed_bounds = (args.preferred_speed_min, args.preferred_speed_max)
if any(value is not None for value in speed_bounds):
    if args.scene != "airplane":
        parser.error("preferred-speed bounds are available only for airplane")
    if any(value is None for value in speed_bounds):
        parser.error("provide both --preferred-speed-min and --preferred-speed-max")
    if (not all(math.isfinite(value) and value > 0 for value in speed_bounds)
            or speed_bounds[0] > speed_bounds[1]):
        parser.error("preferred-speed bounds must be finite, positive, and ordered")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

scene = importlib.import_module(f"scenes.{args.scene}")
if args.preferred_speed_min is not None:
    scene.PREFERRED_SPEED_MIN, scene.PREFERRED_SPEED_MAX = speed_bounds

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scene)
ARTIFACT_SUFFIX = ".candidate" if args.candidate else ""
if args.artifact_tag:
    ARTIFACT_SUFFIX += f".{args.artifact_tag}"
ALPHA = args.alpha
EXPORTING = args.export_json is not None


class BCPolicy(nn.Module):
    """Must match training architecture exactly (incl. tanh on output)."""
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.trunk(x))


# Load model and normalisation stats
stats = np.load(
    os.path.join(DATA_DIR, f"norm_stats{ARTIFACT_SUFFIX}.npy"),
    allow_pickle=True,
).item()
state_mean  = stats["state_mean"]
state_std   = stats["state_std"]
action_mean = stats["action_mean"]
action_std  = stats["action_std"]

STATE_DIM  = len(state_mean)
ACTION_DIM = len(action_mean)

if args.device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but is not available")
device = torch.device(
    "cuda"
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    else "cpu"
)
if device.type == "cpu":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
checkpoint = torch.load(
    os.path.join(DATA_DIR, f"bc_policy{ARTIFACT_SUFFIX}.pt"),
    map_location=device,
    weights_only=True,
)

# Older checkpoints named the Sequential module "net" instead of "trunk".
if any(key.startswith("net.") for key in checkpoint):
    checkpoint = {
        key.replace("net.", "trunk.", 1): value
        for key, value in checkpoint.items()
    }

weight_keys = sorted(
    (key for key in checkpoint if key.startswith("trunk.") and key.endswith(".weight")),
    key=lambda key: int(key.split(".")[1]),
)
detected_hidden_dims = tuple(checkpoint[key].shape[0] for key in weight_keys[:-1])
hidden_dims = (
    (args.hidden,) * len(detected_hidden_dims)
    if args.hidden is not None else detected_hidden_dims
)
if hidden_dims != detected_hidden_dims:
    raise ValueError(
        f"--hidden={args.hidden} does not match checkpoint dimensions "
        f"{detected_hidden_dims}."
    )

model = BCPolicy(STATE_DIM, ACTION_DIM, hidden_dims).to(device)
model.load_state_dict(checkpoint)
model.eval()
print(f"Model loaded  (STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, "
      f"hidden={hidden_dims}, alpha={ALPHA})")

# Run
screen = None
clock = None
if not EXPORTING:
    pygame.init()
    screen = pygame.display.set_mode((scene.SCREEN_WIDTH, scene.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption(f"BC Inference: {args.scene}")

if (args.layout is not None or args.occupancy is not None) and args.scene != "airplane":
    raise SystemExit("--layout and --occupancy are available only for airplane")
if args.layout is not None or args.occupancy is not None:
    walls = scene.build_walls(args.layout)
    agents, meta = scene.build_agents(
        count=args.count,
        layout=args.layout,
        occupancy=args.occupancy,
        seed=args.seed,
    )
else:
    walls = scene.build_walls()
    build_kwargs = {"count": args.count}
    if "seed" in inspect.signature(scene.build_agents).parameters:
        build_kwargs["seed"] = args.seed
    agents, meta = scene.build_agents(**build_kwargs)
controller_factory = getattr(scene, "build_controller", None)
controller_class = getattr(scene, "SceneController", None)
if controller_factory is not None:
    controller = controller_factory(agents, meta)
elif controller_class is not None:
    controller = controller_class(agents, meta)
else:
    controller = None
initial_agent_count = len(agents)

prev_velocities = {}
pause = False
running = True
frame_count = 0
replay_frames = [capture_frame(agents)] if EXPORTING else []

while running:
    if not EXPORTING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause

    if not pause:
        if controller is not None:
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

            velocities = []
            action_cleaner = getattr(scene, "clean_policy_action", None)
            action_scaler = getattr(scene, "scale_action_for_agent", None)
            for agent, state, prediction in zip(agents, states, predictions):
                previous = prev_velocities.get(agent.id, prediction)
                blended = ALPHA * prediction + (1.0 - ALPHA) * previous
                prev_velocities[agent.id] = blended

                velocity = Vector2(float(blended[0]), float(blended[1]))

                if action_cleaner is not None:
                    cleaned = action_cleaner(state, velocity)
                    velocity = Vector2(float(cleaned[0]), float(cleaned[1]))
                else:
                    stop_threshold = getattr(scene, "POLICY_STOP_THRESHOLD", 0.5)
                    if velocity.length() < stop_threshold:
                        velocity = Vector2(0, 0)
                if action_scaler is not None:
                    scaled = action_scaler(agent, velocity)
                    velocity = Vector2(float(scaled[0]), float(scaled[1]))
                if velocity.length() > agent.max_speed:
                    velocity = velocity.normalize() * agent.max_speed
                velocities.append(velocity)

            advance_velocity_agents(
                agents,
                velocities,
                walls,
                scene.TIMESTEP,
            )
            agents = [agent for agent in agents if scene.filter_agent(agent)]

        frame_count += 1

        if EXPORTING:
            reached_safety_limit = (
                args.max_frames > 0 and frame_count >= args.max_frames
            )
            if frame_count % args.frame_stride == 0 or not agents or reached_safety_limit:
                replay_frames.append(capture_frame(agents))
            if frame_count % 1000 == 0:
                print(f"  frame {frame_count}: {len(agents)} agents remain", flush=True)
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
    complete = len(agents) == 0
    replay = {
        "version": 2,
        "scene": args.scene,
        "agent_count": initial_agent_count,
        "width": scene.SCREEN_WIDTH,
        "height": scene.SCREEN_HEIGHT,
        "fps": 60 / args.frame_stride,
        "frame_stride": args.frame_stride,
        "simulation_frames": frame_count,
        "complete": complete,
        "checkpoint": f"bc_policy{ARTIFACT_SUFFIX}.pt",
        "seed": args.seed,
        "walls": [
            [wall.rect.left, wall.rect.top, wall.rect.width, wall.rect.height]
            for wall in walls
        ],
        "frames": replay_frames,
    }
    if args.scene == "airplane":
        replay["layout"] = meta["layout"].label
        replay["preferred_speed_range"] = [scene.PREFERRED_SPEED_MIN, scene.PREFERRED_SPEED_MAX]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = write_web_replay(replay, repo_root, args.export_json or None)

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
