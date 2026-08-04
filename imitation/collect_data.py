"""
Collect behavioral cloning data for a given scene.

Usage:
    python collect_data.py --scene airplane
    python collect_data.py --scene queue
    python collect_data.py --scene airplane --episodes 8 --headless

Each scene module must export:
    STATE_DIM, ACTION_DIM
    extract_state(agent, agents) -> np.ndarray
    extract_action(agent)        -> np.ndarray
    filter_agent(agent)          -> bool
    build_walls(), build_agents(), TIMESTEP,
    SCREEN_WIDTH, SCREEN_HEIGHT

Scenes may optionally export:
    SceneController or build_controller(agents, meta)
    expert_action(state) for direct-velocity demonstrations
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import numpy as np
import pygame
from core.physics import advance_velocity_agents, update_agent

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Scene name (e.g. airplane, queue)")
parser.add_argument("--episodes", type=int, default=8,
                    help="Number of complete episodes to collect.")
parser.add_argument("--frames", type=int, default=0,
                    help="Optional total-frame safety limit. Zero means unlimited.")
parser.add_argument("--episode_cap", type=int, default=None,
                    help="Per-episode safety limit. Defaults to the scene recommendation.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--candidate", action="store_true",
                    help="Write expert_data.candidate.npz instead of replacing production data.")
parser.add_argument("--headless", action="store_true",
                    help="Disable the collection window and frame-rate limit.")
args = parser.parse_args()

scene = importlib.import_module(f"scenes.{args.scene}")
random.seed(args.seed)
np.random.seed(args.seed)
episode_cap = args.episode_cap or getattr(scene, "TRAINING_EPISODE_CAP", 1500)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scene)
os.makedirs(DATA_DIR, exist_ok=True)

screen = None
clock = None
if not args.headless:
    pygame.init()
    screen = pygame.display.set_mode((scene.SCREEN_WIDTH, scene.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption(f"Collecting data: {args.scene}")

states_list = []
actions_list = []
episode_ids_list = []
phases_list = []
collect_phases = hasattr(scene, "training_phase")
frame = 0
episode = 0
running = True
print(f"Collecting data for '{args.scene}' "
      f"(STATE_DIM={scene.STATE_DIM}, ACTION_DIM={scene.ACTION_DIM})...")

# --- EPISODE LOOP ---
# The airplane scene has one source of variation: the random seat-order shuffle
# inside build_agents(). Running multiple short episodes draws fresh shuffles,
# which is the whole point — same total frames, far more diversity.
while running and episode < args.episodes and (args.frames <= 0 or frame < args.frames):
    episode += 1
    walls = scene.build_walls()
    agents, meta = scene.build_agents()
    initialize_episode = getattr(scene, "initialize_training_episode", None)
    if initialize_episode is not None:
        initialize_episode(agents, random)
    controller_factory = getattr(scene, "build_controller", None)
    controller_class = getattr(scene, "SceneController", None)
    if controller_factory is not None:
        controller = controller_factory(agents, meta)
    elif controller_class is not None:
        controller = controller_class(agents, meta)
    else:
        controller = None

    episode_frame = 0
    episode_states = []
    episode_actions = []
    episode_phases = []

    while (running
           and (args.frames <= 0 or frame < args.frames)
           and episode_frame < episode_cap
           and len(agents) > 0):

        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # 5% chance to nudge the expert (covariate-shift coverage). Scenes may
        # constrain this so noise cannot create physically invalid labels.
        for agent in agents:
            if random.random() < 0.05:
                perturb = getattr(scene, "perturb_for_training", None)
                if perturb is not None:
                    perturb(agent, random)
                else:
                    agent.position.x += random.uniform(-2, 2)
                    agent.position.y += random.uniform(-2, 2)

        if controller is not None:
            controller.update(agents)

        # Capture state BEFORE physics — this matches inference, where
        # extract_state is called before the agent moves. Capturing after
        # physics would let the network learn a trivial velocity-copy shortcut.
        pre_states = {
            agent.id: scene.extract_state(agent, agents)
            for agent in agents if scene.filter_agent(agent)
        }
        pre_phases = (
            {
                agent.id: scene.training_phase(agent)
                for agent in agents if scene.filter_agent(agent)
            }
            if collect_phases else {}
        )

        direct_expert = getattr(scene, "expert_action", None)
        if direct_expert is not None:
            active_agents = [agent for agent in agents if agent.id in pre_states]
            actions = [direct_expert(pre_states[agent.id]) for agent in active_agents]
            advance_velocity_agents(
                active_agents,
                actions,
                walls,
                scene.TIMESTEP,
            )
        else:
            for agent in agents:
                should_move = getattr(scene, "expert_should_move", None)
                if should_move is None or should_move(agent):
                    update_agent(agent, agents, scene.TIMESTEP)
                clean_motion = getattr(scene, "clean_expert_motion", None)
                if clean_motion is not None:
                    clean_motion(agent)
            for agent in agents:
                for wall in walls:
                    wall.collision_resolve(agent)

        # Log samples BEFORE despawning, so a state captured this frame always
        # has its matching action recorded.
        for agent in agents:
            if agent.id in pre_states:
                episode_states.append(pre_states[agent.id])
                episode_actions.append(scene.extract_action(agent))
                if collect_phases:
                    episode_phases.append(pre_phases[agent.id])

        # Drop dead agents from the active list. Using filter_agent (the
        # scene's own "is this agent still in play?" predicate) instead of
        # a y-coordinate hack — works for any scene.
        agents = [a for a in agents if scene.filter_agent(a)]

        # Render
        if not args.headless:
            screen.fill((30, 30, 30))
            for wall in walls:
                pygame.draw.rect(screen, (200, 200, 200), wall.rect)
            for agent in agents:
                pygame.draw.circle(screen, (0, 255, 0),
                                   (int(agent.position.x), int(agent.position.y)),
                                   int(agent.radius))
            pygame.display.flip()
            clock.tick(60)

        frame += 1
        episode_frame += 1

        if frame % 100 == 0:
            frame_target = args.frames if args.frames > 0 else "unlimited"
            print(f"  frame {frame}/{frame_target}  "
                  f"(episode {episode}, ep-frame {episode_frame}, "
                  f"samples so far: {len(states_list) + len(episode_states)})")

    complete = len(agents) == 0
    if complete:
        states_list.extend(episode_states)
        actions_list.extend(episode_actions)
        if collect_phases:
            phases_list.extend(episode_phases)
        episode_ids_list.extend([episode - 1] * len(episode_states))
    ep_samples = len(episode_states)
    print(f"Episode {episode} ended after {episode_frame} frames "
          f"(agents remaining: {len(agents)}, samples this episode: {ep_samples}, "
          f"{'kept' if complete else 'discarded'})")
    if running and not complete:
        raise RuntimeError(
            f"Episode {episode} did not complete within {episode_cap} frames. "
            "Increase --episode_cap; incomplete episodes are never saved."
        )

if not args.headless:
    pygame.quit()

if not states_list:
    raise RuntimeError("No complete episodes were collected; no dataset was written.")

filename = "expert_data.candidate.npz" if args.candidate else "expert_data.npz"
out_path = os.path.join(DATA_DIR, filename)
payload = dict(
    states=np.array(states_list, dtype=np.float32),
    actions=np.array(actions_list, dtype=np.float32),
    episode_ids=np.array(episode_ids_list, dtype=np.int32),
)
if collect_phases:
    payload["phases"] = np.array(phases_list, dtype=np.int8)
np.savez(out_path, **payload)
print(f"\nDone. {episode} complete episodes, {len(states_list)} samples -> {out_path}")
