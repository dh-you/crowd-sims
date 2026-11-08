"""Render accelerated GIFs of candidate-policy airplane rollouts."""

import argparse
import os
import random
import sys

import numpy as np
import pygame
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.physics import advance_velocity_agents
from evaluate_airplane import checkpoint_hidden_dims, policy_actions
from scenes import airplane as scene
from train_bc import BCPolicy


CROP_RECT = pygame.Rect(210, 135, 180, 305)
OUTPUT_SIZE = (320, 528)
VIEW_SIZE = (288, 488)
VIEW_OFFSET = (16, 32)
BACKGROUND = (30, 30, 30)
WALL_COLOR = (200, 200, 200)
AGENT_COLOR = (255, 100, 0)
TEXT_COLOR = (235, 235, 235)


def draw_frame(
    layout,
    walls,
    agents,
    frame_number,
    initial_count,
    font,
    ablation_name=None,
    display_label=None,
):
    """Draw one cropped, labelled airplane frame as a palette image."""
    full_surface = pygame.Surface((scene.SCREEN_WIDTH, scene.SCREEN_HEIGHT))
    full_surface.fill(BACKGROUND)
    for wall in walls:
        pygame.draw.rect(full_surface, WALL_COLOR, wall.rect)
    for agent in agents:
        pygame.draw.circle(
            full_surface,
            AGENT_COLOR,
            (round(agent.position.x), round(agent.position.y)),
            round(agent.radius),
        )

    view = pygame.transform.smoothscale(
        full_surface.subsurface(CROP_RECT),
        VIEW_SIZE,
    )
    output = pygame.Surface(OUTPUT_SIZE)
    output.fill(BACKGROUND)
    output.blit(view, VIEW_OFFSET)
    ablation_labels = {
        "aisle_offset": "no-aisle",
        "row_clearance": "no-row",
        "opposing_clearance": "no-opposing",
        "forward_clearance": "no-forward",
    }
    policy_label = display_label or ablation_labels.get(ablation_name, "")
    label = f"{layout.label} {policy_label}  f{frame_number:,}  {len(agents)}/{initial_count}"
    output.blit(font.render(label, True, TEXT_COLOR), (16, 8))

    rgb = pygame.image.tobytes(output, "RGB")
    image = Image.frombytes("RGB", OUTPUT_SIZE, rgb)
    return image.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)


def render_layout(
    layout,
    model,
    stats,
    device,
    output_path,
    stride,
    frame_duration_ms,
    max_frames,
    ablation_name=None,
    seed=42,
    display_label=None,
):
    walls = scene.build_walls(layout)
    agents, _ = scene.build_agents(layout=layout, seed=seed)
    initial_count = len(agents)
    font = pygame.font.Font(None, 20)
    frames = [
        draw_frame(
            layout,
            walls,
            agents,
            0,
            initial_count,
            font,
            ablation_name,
            display_label,
        )
    ]
    complete = False

    for frame_number in range(1, max_frames + 1):
        full_states = np.asarray(
            [scene.extract_state(agent, agents) for agent in agents],
            dtype=np.float32,
        )
        states = scene.project_state_observations(full_states, ablation_name)
        actions = policy_actions(
            model,
            stats,
            device,
            states,
            cleanup_states=full_states,
        )
        action_scaler = getattr(scene, "scale_action_for_agent", None)
        physical_actions = (
            [
                action_scaler(agent, action)
                for agent, action in zip(agents, actions)
            ]
            if action_scaler is not None else actions
        )
        advance_velocity_agents(agents, physical_actions, walls, scene.TIMESTEP)
        agents = [agent for agent in agents if scene.filter_agent(agent)]

        if frame_number % stride == 0 or not agents:
            frames.append(
                draw_frame(
                    layout,
                    walls,
                    agents,
                    frame_number,
                    initial_count,
                    font,
                    ablation_name,
                    display_label,
                )
            )
        if not agents:
            complete = True
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    durations = [1000] + [frame_duration_ms] * (len(frames) - 2) + [1500]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    status = "complete" if complete else f"capped with {len(agents)} aboard"
    print(
        f"Rendered {layout.label} {ablation_name or 'full'}: "
        f"{frame_number} simulation frames ({status}), "
        f"{len(frames)} GIF frames -> {output_path}",
        flush=True,
    )


def load_policy(device, ablation_name=None, artifact_tag=None):
    suffix = ".candidate"
    if artifact_tag:
        suffix += f".{artifact_tag}"
    input_dim = scene.STATE_DIM
    if ablation_name is not None:
        if ablation_name not in scene.STATE_NAMES:
            raise ValueError(f"unknown ablation {ablation_name!r}")
        suffix += f".no_{ablation_name}"
        input_dim -= 1
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airplane")
    stats = np.load(
        os.path.join(data_dir, f"norm_stats{suffix}.npy"),
        allow_pickle=True,
    ).item()
    checkpoint = torch.load(
        os.path.join(data_dir, f"bc_policy{suffix}.pt"),
        map_location=device,
        weights_only=True,
    )
    model = BCPolicy(input_dim, scene.ACTION_DIM, checkpoint_hidden_dims(checkpoint))
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return stats, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layouts",
        default="2x1,2x2,3x2,3x4,3x4x3",
        help="Comma-separated airplane layouts.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Tagged candidate suffix, such as 'heterogeneous'.",
    )
    parser.add_argument(
        "--filename-suffix",
        default=None,
        help="Optional output filename suffix.",
    )
    parser.add_argument("--output-dir", default="imitation/airplane/gifs")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--frame-duration-ms", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--ablations",
        default=None,
        help="Comma-separated omitted observations; default renders the full candidate.",
    )
    args = parser.parse_args()
    if args.stride < 1 or args.frame_duration_ms < 1:
        raise SystemExit("stride and frame duration must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    pygame.font.init()

    layouts = [
        scene.parse_layout(value.strip())
        for value in args.layouts.split(",")
        if value.strip()
    ]
    ablations = (
        [value.strip() for value in args.ablations.split(",") if value.strip()]
        if args.ablations is not None else [None]
    )
    for ablation_name in ablations:
        stats, model = load_policy(device, ablation_name, args.artifact_tag)
        for layout in layouts:
            filename = f"airplane_{layout.label}"
            if ablation_name is not None:
                filename += f"_without_{ablation_name}"
            if args.filename_suffix:
                filename += f"_{args.filename_suffix}"
            output_path = os.path.join(args.output_dir, filename + ".gif")
            print(
                f"Simulating {layout.label} {ablation_name or 'full'}...",
                flush=True,
            )
            render_layout(
                layout,
                model,
                stats,
                device,
                output_path,
                args.stride,
                args.frame_duration_ms,
                args.max_frames,
                ablation_name,
                args.seed,
                args.artifact_tag,
            )


if __name__ == "__main__":
    main()
