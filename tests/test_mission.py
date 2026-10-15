"""
Unit and Integration Test Suite for WALL-E & EVE's Signal.
Verifies POMDP environment isolation, Bayesian belief updates,
A* pathfinding, and battery fail-safe guarantees.
"""

import unittest
from pathlib import Path
import numpy as np

from src.core.environment import WastelandEnvironment
from src.core.belief import SpatialBeliefEngine
from src.core.planner import MissionPlanner
from src.core.agent import AutonomousWallEAgent


class TestWalleAutonomousSystem(unittest.TestCase):

    def setUp(self):
        self.config_path = "configs/mission_wasteland.json"
        self.env = WastelandEnvironment(self.config_path)

    def test_environment_initialization_and_pomdp_isolation(self):
        """Test that environment initializes correctly and isolates ground truth."""
        obs = self.env.reset()
        self.assertEqual(obs["agent_position"], (1, 1))
        self.assertEqual(obs["battery"], 100)
        self.assertFalse(self.env.is_done)
        
        # Ensure agent cannot peek at ground truth in observation dictionary
        self.assertNotIn("true_position", obs)
        self.assertNotIn("_true_eve_pos", obs)
        self.assertNotIn("_true_eve_status", obs)

    def test_bayesian_belief_entropy_reduction(self):
        """Test that Bayesian updates properly reduce uncertainty (Shannon Entropy)."""
        belief = SpatialBeliefEngine(width=10, height=10)
        initial_entropy = belief.get_shannon_entropy()
        self.assertGreater(initial_entropy, 5.0)

        # Apply an RF signal observation with bearing towards south-east (8, 8)
        belief.update_with_rf_signal(wall_e_pos=(1, 1), rssi=0.28, bearing_deg=135.0)
        signal_entropy = belief.get_shannon_entropy()
        self.assertLess(signal_entropy, initial_entropy)

        # Apply clue evidence restricting area to X >= 6 and Y >= 6
        belief.update_with_clue_evidence({
            "type": "ION_TRAIL",
            "evidence": {"x_min": 6, "y_min": 6}
        })
        clue_entropy = belief.get_shannon_entropy()
        self.assertLess(clue_entropy, signal_entropy)

        # The peak belief coordinate should now be in the south-east quadrant
        top_x, top_y = belief.get_highest_belief_coordinate()
        self.assertGreaterEqual(top_x, 6)
        self.assertGreaterEqual(top_y, 6)

    def test_astar_pathfinding_and_obstacle_avoidance(self):
        """Test that A* correctly routes around obstacles without diagonal clipping."""
        obstacles = {(1, 0), (1, 1), (1, 2)}  # Vertical wall
        planner = MissionPlanner(width=5, height=5, obstacles=obstacles, solar_stations=set(), hazards=set())
        
        start = (0, 1)
        goal = (2, 1)
        path = planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        
        # Verify no path node intersects with obstacles
        for coord, _ in path:
            self.assertNotIn(coord, obstacles)

    def test_battery_critical_failsafe(self):
        """Test that planner accurately flags critical battery and computes return path."""
        solar_stations = {(0, 0)}
        planner = MissionPlanner(width=10, height=10, obstacles=set(), solar_stations=solar_stations, hazards=set())
        
        # Agent at (4, 4), distance to (0, 0) is 8 steps. Cost = 8 * 2 = 16 battery.
        # With safety margin of 8, critical threshold = 24.
        
        # Battery = 20 (below threshold) -> Must be critical
        is_crit, cost, station, path = planner.is_battery_critical((4, 4), current_battery=20, safety_margin=8)
        self.assertTrue(is_crit)
        self.assertEqual(station, (0, 0))
        self.assertEqual(cost, 16)
        self.assertIsNotNone(path)

        # Battery = 50 (well above threshold) -> Not critical
        is_crit_safe, _, _, _ = planner.is_battery_critical((4, 4), current_battery=50, safety_margin=8)
        self.assertFalse(is_crit_safe)

    def test_full_mission_convergence(self):
        """Test that autonomous agent successfully converges on EVE without depleting battery."""
        obs = self.env.reset()
        agent = AutonomousWallEAgent(obs, self.env.costs)

        done = False
        step = 0
        while not done and step < 130:
            step += 1
            action, kwargs, reasoning = agent.decide_next_action()
            obs, cost, done, info = self.env.step(action, **kwargs)
            agent.process_observation(obs, info)

        self.assertTrue(self.env.eve_found or self.env.get_ground_truth_for_evaluation()["success"])
        self.assertGreater(self.env.battery, 0)


if __name__ == "__main__":
    unittest.main()
