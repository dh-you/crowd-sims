import inspect
import os
import random
import sys
import unittest

import numpy as np
from pygame.math import Vector2

IMITATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if IMITATION_ROOT not in sys.path:
    sys.path.insert(0, IMITATION_ROOT)

from core.agent import Agent
from core.physics import advance_velocity_agents
from core.wall import Wall
from evaluate_airplane import (
    action_matches_expert_regime,
    candidate_boundary_cases,
    count_blocked_departure_starts,
)
from scenes import airplane
from train_bc import BCPolicy


def make_agent(agent_id, x, y):
    return Agent(agent_id, Vector2(x, y), 3, 5, 30, 5, 2, 10, 5)


class AirplaneObservationTests(unittest.TestCase):
    def test_same_row_forward_sensing_starts_only_after_aisle_occupancy(self):
        agents, _ = airplane.build_agents(count=6, seed=11)
        follower = agents[2]
        leader = agents[5]
        leader.position.x = airplane.AISLE_X
        leader.position.y = follower.position.y + 12.0

        outside_state = airplane.extract_state(follower, [follower, leader])
        follower.position.x = airplane.AISLE_X + airplane.AISLE_OCCUPANCY_HALF_WIDTH
        inside_state = airplane.extract_state(follower, [follower, leader])

        self.assertAlmostEqual(float(outside_state[3]), 1.0)
        self.assertAlmostEqual(float(inside_state[3]), 0.12)

    def test_named_three_input_projections_preserve_order(self):
        state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        expected = {
            "aisle_offset": [0.2, 0.3, 0.4],
            "row_clearance": [0.1, 0.3, 0.4],
            "opposing_clearance": [0.1, 0.2, 0.4],
            "forward_clearance": [0.1, 0.2, 0.3],
        }

        self.assertEqual(tuple(expected), airplane.STATE_NAMES)
        for omitted_name, remaining in expected.items():
            np.testing.assert_allclose(
                airplane.project_state_observations(state, omitted_name),
                remaining,
            )

    def test_3x4x3_builds_two_assigned_aisles(self):
        layout = airplane.parse_layout("3x4x3")
        agents, meta = airplane.build_agents(layout=layout)

        self.assertIsInstance(layout, airplane.WidebodyLayout)
        self.assertEqual(layout.capacity, 170)
        self.assertEqual(len(agents), 170)
        self.assertEqual(meta["aisle_positions"], {0: 270.0, 1: 330.0})
        self.assertEqual(
            [sum(agent.getData("start_aisle") == index for agent in agents)
             for index in (0, 1)],
            [85, 85],
        )

        first_row = agents[:10]
        self.assertEqual(
            [agent.position.x for agent in first_row],
            [234.0, 246.0, 258.0, 282.0, 294.0,
             306.0, 318.0, 366.0, 354.0, 342.0],
        )

    def test_3x4x3_offsets_are_relative_to_each_assigned_aisle(self):
        agents, _ = airplane.build_agents(layout="3x4x3")

        outer_left = airplane.extract_state(agents[0], agents)
        middle_left_half = airplane.extract_state(agents[3], agents)
        middle_right_half = airplane.extract_state(agents[5], agents)
        outer_right = airplane.extract_state(agents[7], agents)

        self.assertAlmostEqual(float(outer_left[0]), 0.75)
        self.assertAlmostEqual(float(middle_left_half[0]), -0.25)
        self.assertAlmostEqual(float(middle_right_half[0]), 0.5)
        self.assertAlmostEqual(float(outer_right[0]), -0.75)

    def test_3x4x3_sensors_do_not_cross_between_aisles(self):
        agents, _ = airplane.build_agents(layout="3x4x3")
        left_aisle_agent = agents[3]
        other_aisle_agent = agents[15]

        # Put a different-row passenger directly ahead in the left corridor,
        # but retain its immutable right-aisle assignment.
        other_aisle_agent.position.x = left_aisle_agent.getData("aisle_x")
        other_aisle_agent.position.y = left_aisle_agent.position.y + 12.0
        state = airplane.extract_state(left_aisle_agent, [left_aisle_agent, other_aisle_agent])

        self.assertEqual(left_aisle_agent.getData("start_aisle"), 0)
        self.assertEqual(other_aisle_agent.getData("start_aisle"), 1)
        self.assertAlmostEqual(float(state[3]), 1.0)

    def test_3x4x3_front_walls_leave_two_aisle_gaps(self):
        walls = airplane.build_walls("3x4x3")

        self.assertEqual(len(walls), 6)
        front_walls = sorted(
            (wall.rect for wall in walls if wall.rect.centery == 420),
            key=lambda rect: rect.left,
        )
        self.assertEqual(
            [
                front_walls[index + 1].left - front_walls[index].right
                for index in range(len(front_walls) - 1)
            ],
            [18, 18],
        )
        for aisle_x in (270.0, 330.0):
            self.assertTrue(all(not wall.rect.collidepoint(aisle_x, 420.0) for wall in walls))

    def test_fixed_48_pixel_scale_represents_outer_3x4_seat(self):
        layout = airplane.CabinLayout(3, 4)
        agents, _ = airplane.build_agents(layout=layout)
        outer_right = agents[0]

        state = airplane.extract_state(outer_right, agents)

        self.assertEqual(len(agents), 119)
        self.assertAlmostEqual(outer_right.position.x, 348.0)
        self.assertAlmostEqual(float(state[0]), -1.0)
        self.assertAlmostEqual(airplane.AISLE_EPS, 6.0 / 48.0)

    def test_partial_layout_occupancy_is_seeded_and_not_front_filled(self):
        layout = airplane.CabinLayout(2, 1)
        first, _ = airplane.build_agents(layout=layout, occupancy=0.5, seed=7)
        repeated, _ = airplane.build_agents(layout=layout, occupancy=0.5, seed=7)

        self.assertEqual(len(first), round(layout.capacity * 0.5))
        self.assertEqual(
            [(agent.position.x, agent.position.y) for agent in first],
            [(agent.position.x, agent.position.y) for agent in repeated],
        )
        self.assertGreater(len({agent.getData("start_row") for agent in first}), 2)

    def test_signs_and_clearance_responsibilities(self):
        agents, _ = airplane.build_agents(count=6)
        right_middle = agents[1]
        right_aisle = agents[2]
        left_aisle = agents[5]

        middle_state = airplane.extract_state(right_middle, agents)
        aisle_state = airplane.extract_state(right_aisle, agents)
        left_state = airplane.extract_state(left_aisle, agents)

        self.assertLess(aisle_state[0], 0)
        self.assertGreater(left_state[0], 0)
        self.assertAlmostEqual(float(middle_state[1]), 0.12, places=5)
        self.assertAlmostEqual(float(aisle_state[1]), 1.0, places=5)
        self.assertAlmostEqual(float(aisle_state[2]), 0.24, places=5)
        self.assertAlmostEqual(float(aisle_state[3]), 1.0, places=5)
        self.assertTrue(np.all(aisle_state >= -1.0))
        self.assertTrue(np.all(aisle_state <= 1.0))

    def test_same_row_vertical_jitter_never_becomes_forward_clearance(self):
        agents, _ = airplane.build_agents(count=6)
        right_aisle = agents[2]
        agents[5].position.y += 7.0
        state = airplane.extract_state(right_aisle, agents)
        self.assertAlmostEqual(float(state[3]), 1.0, places=5)

    def test_different_row_aisle_seat_is_forward_clearance(self):
        agents, _ = airplane.build_agents(count=12)
        behind = agents[2]
        ahead = agents[8]

        initial_state = airplane.extract_state(behind, agents)
        ahead.position.x -= 10.0
        shifted_state = airplane.extract_state(behind, agents)

        self.assertAlmostEqual(float(initial_state[3]), 0.15, places=5)
        self.assertAlmostEqual(float(shifted_state[3]), 0.15, places=5)


class AirplaneExpertTests(unittest.TestCase):
    def test_expert_is_state_only(self):
        parameters = inspect.signature(airplane.expert_action).parameters
        self.assertEqual(list(parameters), ["state"])
        state = np.array([0.5, 1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(
            airplane.expert_action(state),
            airplane.expert_action(state.copy()),
        )

    def test_expert_decisions(self):
        np.testing.assert_array_equal(
            airplane.expert_action([0.5, 1.0, 1.0, 1.0]),
            [airplane.CONFIG["MAXSPEED"], 0.0],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([-0.5, 1.0, 1.0, 1.0]),
            [-airplane.CONFIG["MAXSPEED"], 0.0],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([-0.5, 1.0, 0.2, 1.0]),
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([0.5, 0.1, 1.0, 1.0]),
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([0.0, 1.0, 1.0, 1.0]),
            [0.0, airplane.CONFIG["MAXSPEED"]],
        )

    def test_expert_centers_within_two_pixels_before_forward_motion(self):
        outside_center = 3.0 / airplane.MAX_SEAT_DISTANCE
        inside_center = 2.0 / airplane.MAX_SEAT_DISTANCE

        np.testing.assert_array_equal(
            airplane.expert_action([outside_center, 1.0, 1.0, 1.0]),
            [airplane.CONFIG["MAXSPEED"], 0.0],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([inside_center, 1.0, 1.0, 1.0]),
            [0.0, airplane.CONFIG["MAXSPEED"]],
        )

    def test_expert_uses_proportional_24_to_40_pixel_following(self):
        self.assertAlmostEqual(airplane.FOLLOW_STOP, 0.24)
        self.assertAlmostEqual(airplane.FOLLOW_FULL, 0.40)
        np.testing.assert_array_equal(
            airplane.expert_action([0.0, 1.0, 1.0, 0.24]),
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([0.0, 1.0, 1.0, 0.32]),
            [0.0, 2.5],
        )
        np.testing.assert_array_equal(
            airplane.expert_action([0.0, 1.0, 1.0, 0.40]),
            [0.0, airplane.CONFIG["MAXSPEED"]],
        )

    def test_preferred_speeds_are_seeded_bounded_and_diverse(self):
        first, _ = airplane.build_agents(count=24, seed=19)
        repeated, _ = airplane.build_agents(count=24, seed=19)
        first_speeds = [agent.getData("preferred_speed") for agent in first]
        repeated_speeds = [agent.getData("preferred_speed") for agent in repeated]

        self.assertEqual(first_speeds, repeated_speeds)
        self.assertGreater(len(set(first_speeds)), 20)
        self.assertTrue(
            all(
                airplane.PREFERRED_SPEED_MIN <= speed <= airplane.PREFERRED_SPEED_MAX
                for speed in first_speeds
            )
        )

    def test_preferred_speed_uses_seeded_uniform_sample(self):
        seed = 19
        agents, _ = airplane.build_agents(count=1, seed=seed)
        expected = random.Random(seed).uniform(
            airplane.PREFERRED_SPEED_MIN,
            airplane.PREFERRED_SPEED_MAX,
        )

        self.assertAlmostEqual(agents[0].getData("preferred_speed"), expected)

    def test_physical_speed_scaling_preserves_canonical_labels(self):
        agents, _ = airplane.build_agents(count=1, seed=4)
        agent = agents[0]
        canonical = np.array([5.0, 0.0], dtype=np.float32)
        physical = airplane.scale_action_for_agent(agent, canonical)
        agent.velocity = Vector2(float(physical[0]), float(physical[1]))

        self.assertAlmostEqual(np.linalg.norm(physical), agent.max_speed, places=5)
        np.testing.assert_allclose(airplane.extract_action(agent), canonical)

    def test_cross_axis_cleanup_keeps_slow_following(self):
        centered = [0.0, 1.0, 1.0, 0.25]
        cleaned = airplane.clean_policy_action(centered, [3.1, 0.35])
        np.testing.assert_allclose(cleaned, [0.0, 0.35])

    def test_lateral_cleanup_keeps_binary_merge_threshold(self):
        outside_center = [0.25, 1.0, 1.0, 1.0]

        np.testing.assert_allclose(
            airplane.clean_policy_action(outside_center, [2.49, 0.0]),
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            airplane.clean_policy_action(outside_center, [2.51, 3.1]),
            [2.51, 0.0],
        )

    def test_lateral_cleanup_allows_slow_final_centering(self):
        inside_occupancy = [
            5.0 / airplane.MAX_SEAT_DISTANCE,
            1.0,
            1.0,
            1.0,
        ]

        np.testing.assert_allclose(
            airplane.clean_policy_action(inside_occupancy, [0.31, 4.0]),
            [0.31, 0.0],
        )

    def test_cleanup_rejects_motion_away_from_aisle_and_backward(self):
        outside_center = [0.25, 1.0, 1.0, 1.0]
        centered = [0.0, 1.0, 1.0, 1.0]

        np.testing.assert_allclose(
            airplane.clean_policy_action(outside_center, [-4.0, 0.0]),
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            airplane.clean_policy_action(centered, [0.0, -4.0]),
            [0.0, 0.0],
        )

    def test_airplane_has_no_controller_or_state_flags(self):
        self.assertFalse(hasattr(airplane, "SceneController"))
        agents, _ = airplane.build_agents(count=1)
        self.assertFalse(hasattr(agents[0], "state"))


class SimultaneousPhysicsTests(unittest.TestCase):
    def test_wall_inside_resolution_uses_minimum_translation(self):
        horizontal = Wall(40, True, Vector2(100, 100))
        agent = make_agent(1, 100, 100)
        agent.velocity = Vector2(0, 5)

        horizontal.collision_resolve(agent)

        self.assertAlmostEqual(agent.position.x, 100.0)
        self.assertAlmostEqual(agent.position.y, horizontal.rect.top - agent.radius)

    def test_exit_divider_does_not_invent_rightward_motion(self):
        divider = airplane.build_walls("3x3")[2]
        agent = make_agent(1, divider.rect.centerx, divider.rect.centery)
        agent.velocity = Vector2(0, 5)

        divider.collision_resolve(agent)

        self.assertAlmostEqual(agent.position.x, divider.rect.centerx)
        self.assertLess(agent.position.y, divider.rect.top)

    def _head_on_result(self, reverse):
        left = make_agent(10, 0, 0)
        right = make_agent(20, 2, 0)
        pairs = [(left, Vector2(1, 0)), (right, Vector2(-1, 0))]
        if reverse:
            pairs.reverse()
        diagnostics = advance_velocity_agents(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            [],
            1.0,
        )
        positions = {
            agent.id: (agent.position.x, agent.position.y)
            for agent, _ in pairs
        }
        return positions, diagnostics

    def test_overlap_resolution_is_independent_of_list_order(self):
        forward, forward_diagnostics = self._head_on_result(False)
        reverse, reverse_diagnostics = self._head_on_result(True)
        for agent_id in forward:
            self.assertAlmostEqual(forward[agent_id][0], reverse[agent_id][0])
            self.assertAlmostEqual(forward[agent_id][1], reverse[agent_id][1])
        self.assertEqual(forward_diagnostics["unresolved_overlaps"], 0)
        self.assertEqual(reverse_diagnostics["unresolved_overlaps"], 0)


class AirplanePolicyTests(unittest.TestCase):
    def test_policy_has_exact_mvp_topology(self):
        model = BCPolicy(4, 2, (32, 32))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(parameter_count, 1282)
        self.assertEqual(tuple(model.trunk[0].weight.shape), (32, 4))
        self.assertEqual(tuple(model.trunk[2].weight.shape), (32, 32))
        self.assertEqual(tuple(model.trunk[4].weight.shape), (2, 32))

    def test_three_input_ablation_topology_has_1250_parameters(self):
        model = BCPolicy(3, 2, (32, 32))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(parameter_count, 1250)
        self.assertEqual(tuple(model.trunk[0].weight.shape), (32, 3))
        self.assertEqual(tuple(model.trunk[2].weight.shape), (32, 32))
        self.assertEqual(tuple(model.trunk[4].weight.shape), (2, 32))

    def test_airplane_uses_safety_first_stationary_weight(self):
        self.assertEqual(airplane.WAITING_WEIGHT, 10.0)

    def test_boundary_training_examples_are_state_only_expert_labels(self):
        states, actions = airplane.sample_boundary_training_examples(
            samples_per_side=3,
            bottom_row_repeats=2,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(states.shape, (42, airplane.STATE_DIM))
        self.assertEqual(actions.shape, (42, airplane.ACTION_DIM))
        for state, action in zip(states, actions):
            np.testing.assert_array_equal(action, airplane.expert_action(state))


class AirplaneEvaluationTests(unittest.TestCase):
    def test_boundary_cases_match_expert_and_bottom_row_yield_order(self):
        cases = dict(candidate_boundary_cases())
        for state in cases.values():
            self.assertTrue(
                action_matches_expert_regime(state, airplane.expert_action(state))
            )

        for agent_id in range(5):
            state = cases[f"bottom_row_agent_{agent_id}"]
            self.assertTrue(
                np.allclose(airplane.expert_action(state), [0.0, 0.0])
            )
        left_aisle = cases["bottom_row_agent_5"]
        np.testing.assert_array_equal(
            airplane.expert_action(left_aisle),
            [airplane.CONFIG["MAXSPEED"], 0.0],
        )

    def test_boundary_matcher_rejects_premature_right_merge(self):
        right_merge = dict(candidate_boundary_cases())["right_merge_yields"]
        self.assertFalse(
            action_matches_expert_regime(
                right_merge,
                [-airplane.CONFIG["MAXSPEED"], 0.0],
            )
        )

    def test_blocked_departure_uses_true_state_and_is_counted_once(self):
        agents, _ = airplane.build_agents(count=3)
        states = np.asarray(
            [airplane.extract_state(agent, agents) for agent in agents],
            dtype=np.float32,
        )
        intervened_states = states.copy()
        intervened_states[:, 1] = 1.0
        actions = np.asarray(
            [airplane.expert_action(state) for state in intervened_states],
            dtype=np.float32,
        )
        started_departures = set()

        first_count = count_blocked_departure_starts(
            agents,
            states,
            actions,
            started_departures,
        )
        repeated_count = count_blocked_departure_starts(
            agents,
            states,
            actions,
            started_departures,
        )

        self.assertEqual(first_count, 2)
        self.assertEqual(repeated_count, 0)

    def test_subthreshold_lateral_command_does_not_start_departure(self):
        agents, _ = airplane.build_agents(count=1)
        states = np.asarray([airplane.extract_state(agents[0], agents)])
        actions = np.asarray([[-0.24, 0.0]], dtype=np.float32)
        started_departures = set()

        count = count_blocked_departure_starts(
            agents,
            states,
            actions,
            started_departures,
        )

        self.assertEqual(count, 0)
        self.assertEqual(started_departures, set())


if __name__ == "__main__":
    unittest.main()
