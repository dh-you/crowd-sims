import inspect
import os
import sys
import unittest

import numpy as np
from pygame.math import Vector2

IMITATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if IMITATION_ROOT not in sys.path:
    sys.path.insert(0, IMITATION_ROOT)

from core.agent import Agent
from core.physics import advance_velocity_agents
from scenes import airplane
from train_bc import BCPolicy


def make_agent(agent_id, x, y):
    return Agent(agent_id, Vector2(x, y), 3, 5, 30, 5, 2, 10, 5)


class AirplaneObservationTests(unittest.TestCase):
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
        state = airplane.extract_state(agents[2], agents)
        self.assertAlmostEqual(float(state[3]), 0.15, places=5)


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

    def test_airplane_has_no_controller_or_state_flags(self):
        self.assertFalse(hasattr(airplane, "SceneController"))
        agents, _ = airplane.build_agents(count=1)
        self.assertFalse(hasattr(agents[0], "state"))


class SimultaneousPhysicsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
