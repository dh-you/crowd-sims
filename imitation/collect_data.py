"""
Collect behavioral cloning data for a given scene.

Usage:
    python collect_data.py --scene airplane
    python collect_data.py --scene queue
    python collect_data.py --scene airplane --frames 1100

Each scene module must export:
    STATE_DIM, ACTION_DIM
    extract_state(agent, agents) -> np.ndarray
    extract_action(agent)        -> np.ndarray
    filter_agent(agent)          -> bool
    build_walls(), build_agents(), SceneController, TIMESTEP,
    SCREEN_WIDTH, SCREEN_HEIGHT
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import numpy as np
import pygame
from core.physics import update_agent

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="Scene name (e.g. airplane, queue)")
parser.add_argument("--frames", type=int, default=5500, help="Total frames to collect across all episodes")
parser.add_argument("--episode_cap", type=int, default=1500,
                    help="Max frames per episode before forcing a reset. Set higher than "
                         "a normal deboarding (~1100 frames for airplane) so episodes end "
                         "naturally when len(agents)==0, not by hitting this cap.")
args = parser.parse_args()

scene = importlib.import_module(f"scenes.{args.scene}")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scene)
os.makedirs(DATA_DIR, exist_ok=True)

pygame.init()
screen = pygame.display.set_mode((scene.SCREEN_WIDTH, scene.SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption(f"Collecting data: {args.scene}")

states_list = []
actions_list = []
frame = 0
episode = 0
running = True
print(f"Collecting data for '{args.scene}' "
      f"(STATE_DIM={scene.STATE_DIM}, ACTION_DIM={scene.ACTION_DIM})...")

# --- EPISODE LOOP ---
# The airplane scene has one source of variation: the random seat-order shuffle
# inside build_agents(). Running multiple short episodes draws fresh shuffles,
# which is the whole point — same total frames, far more diversity.
while running and frame < args.frames:
    episode += 1
    walls = scene.build_walls()
    agents, meta = scene.build_agents()
    controller = scene.SceneController(agents, meta)

    episode_frame = 0
    samples_at_episode_start = len(states_list)

    while (running
           and frame < args.frames
           and episode_frame < args.episode_cap
           and len(agents) > 0):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 5% chance to nudge the expert (covariate-shift coverage)
        for agent in agents:
            if random.random() < 0.05:
                agent.position.x += random.uniform(-2, 2)
                agent.position.y += random.uniform(-2, 2)

        controller.update(agents)

        # Capture state BEFORE physics — this matches inference, where
        # extract_state is called before the agent moves. Capturing after
        # physics would let the network learn a trivial velocity-copy shortcut.
        pre_states = {
            agent.id: scene.extract_state(agent, agents)
            for agent in agents if scene.filter_agent(agent)
        }

        for agent in agents:
            update_agent(agent, agents, scene.TIMESTEP)
        for agent in agents:
            for wall in walls:
                wall.collision_resolve(agent)

        # Log samples BEFORE despawning, so a state captured this frame always
        # has its matching action recorded.
        for agent in agents:
            if scene.filter_agent(agent) and agent.id in pre_states:
                states_list.append(pre_states[agent.id])
                actions_list.append(scene.extract_action(agent))

        # Drop dead agents from the active list. Using filter_agent (the
        # scene's own "is this agent still in play?" predicate) instead of
        # a y-coordinate hack — works for any scene.
        agents = [a for a in agents if scene.filter_agent(a)]

        # Render
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
            print(f"  frame {frame}/{args.frames}  "
                  f"(episode {episode}, ep-frame {episode_frame}, "
                  f"samples so far: {len(states_list)})")

    ep_samples = len(states_list) - samples_at_episode_start
    print(f"Episode {episode} ended after {episode_frame} frames "
          f"(agents remaining: {len(agents)}, samples this episode: {ep_samples})")

pygame.quit()

out_path = os.path.join(DATA_DIR, "expert_data.npz")
np.savez(
    out_path,
    states=np.array(states_list, dtype=np.float32),
    actions=np.array(actions_list, dtype=np.float32),
)
print(f"\nDone. {episode} episodes, {len(states_list)} samples -> {out_path}")
