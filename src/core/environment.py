"""
WALL-E and EVE's Signal - Environment Simulator (POMDP Gridworld)
Handles map representation, resource tracking, sensor physics, and clue interaction.
Strictly isolates ground truth from the autonomous agent.
"""

from typing import Dict, Any, Tuple, List, Optional
import json
import math
import random
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class WastelandEnvironment:
    """
    Simulated 2D grid world for WALL-E's search mission.
    Acts as a Partially Observable Markov Decision Process (POMDP).
    The agent receives only local observations and noisy sensor readings;
    it never has direct access to the environment's internal ground truth.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Grid parameters
        self.width = self.config["grid"]["width"]
        self.height = self.config["grid"]["height"]
        self.start_pos = tuple(self.config["grid"]["wall_e_start"])
        self.obstacles = set(tuple(p) for p in self.config["grid"].get("obstacles", []))
        self.solar_stations = set(tuple(p) for p in self.config["grid"].get("solar_recharge_stations", []))
        self.hazards = set(tuple(p) for p in self.config["grid"].get("hazards", []))
        
        # Ground truth targets (strictly private)
        self._true_eve_pos = tuple(self.config["eve_target"]["true_position"])
        self._true_eve_status = self.config["eve_target"].get("true_status", "operational")
        self._signal_signature = self.config["eve_target"].get("signal_signature", "EVE-STANDARD-PING")
        
        # Decoys / Spoof signals (if any)
        self._decoys = self.config.get("decoys", [])
        
        # Clues placed on the grid
        self._clue_map = {tuple(c["position"]): c for c in self.config.get("clues", [])}
        self.discovered_clues = set()
        
        # Resource budgets & costs
        res = self.config.get("resources", {})
        self.max_battery = res.get("max_battery", 100)
        self.initial_battery = res.get("initial_battery", 100)
        self.max_steps = res.get("max_steps", 100)
        self.costs = res.get("action_costs", {
            "move": 2,
            "move_hazard_penalty": 3,
            "local_scan": 3,
            "long_range_ping": 5,
            "analyze_clue": 4,
            "recharge_gain": 25,
            "wait": 1
        })
        
        # Noise settings
        init_sig = self.config.get("initial_signal", {})
        self.signal_noise_std = init_sig.get("noise_std", 0.05)
        
        # Seed initialization
        seed = self.config.get("seed", 42)
        self.rng = random.Random(seed)
        
        # Dynamic state
        self.agent_pos = self.start_pos
        self.battery = self.initial_battery
        self.step_count = 0
        self.is_done = False
        self.eve_found = False

    def _load_config(self, filepath: str) -> Dict[str, Any]:
        """Loads configuration from either JSON or YAML format."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Mission config file not found: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to parse .yaml config files.")
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def reset(self) -> Dict[str, Any]:
        """Resets the simulation and returns the initial observation."""
        seed = self.config.get("seed", 42)
        self.rng = random.Random(seed)
        self.agent_pos = self.start_pos
        self.battery = self.initial_battery
        self.step_count = 0
        self.is_done = False
        self.eve_found = False
        self.discovered_clues = set()

        initial_obs = {
            "agent_position": self.agent_pos,
            "battery": self.battery,
            "max_battery": self.max_battery,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "grid_dimensions": (self.width, self.height),
            "is_solar_station": self.agent_pos in self.solar_stations,
            "solar_stations": list(self.solar_stations),
            "initial_signal": self.config.get("initial_signal", {}),
            "visible_local": self._get_local_scan(radius=1)
        }
        return initial_obs

    def _get_local_scan(self, radius: int = 1) -> Dict[str, Any]:
        """Scans the immediate neighborhood of WALL-E."""
        ax, ay = self.agent_pos
        scanned_cells = []
        clue_detected = None
        eve_detected = False

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell_type = "empty"
                    if (nx, ny) in self.obstacles:
                        cell_type = "obstacle"
                    elif (nx, ny) in self.solar_stations:
                        cell_type = "solar_pad"
                    elif (nx, ny) in self.hazards:
                        cell_type = "hazard"

                    cell_info = {
                        "coord": (nx, ny),
                        "type": cell_type,
                        "has_clue": (nx, ny) in self._clue_map and (nx, ny) not in self.discovered_clues
                    }
                    scanned_cells.append(cell_info)

                    if (nx, ny) == self._true_eve_pos:
                        eve_detected = True
                    
                    if (nx, ny) in self._clue_map and (nx, ny) not in self.discovered_clues:
                        clue_detected = {
                            "coord": (nx, ny),
                            "id": self._clue_map[(nx, ny)]["id"]
                        }

        return {
            "radius": radius,
            "cells": scanned_cells,
            "clue_nearby": clue_detected,
            "eve_in_sight": eve_detected
        }

    def _calculate_rf_signal(self) -> Dict[str, Any]:
        """
        Physics-based signal calculation:
        RSSI attenuation: S = 1 / (1 + 0.12 * distance) + Gaussian noise
        Angle bearing: True angle in radians + Gaussian angular perturbation
        Also returns decoy readings if present to simulate multi-beacon/spoof environments.
        """
        ax, ay = self.agent_pos
        ex, ey = self._true_eve_pos
        
        # Euclidean distance to EVE
        dist = math.sqrt((ex - ax) ** 2 + (ey - ay) ** 2)
        base_rssi = 1.0 / (1.0 + 0.12 * dist)
        noisy_rssi = max(0.01, min(1.0, base_rssi + self.rng.gauss(0, self.signal_noise_std)))
        
        # Bearing angle to EVE in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        dx = ex - ax
        dy = ey - ay
        true_angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        angular_noise = self.rng.gauss(0, 8.0) # +/- 8 degrees standard deviation
        noisy_angle = (true_angle + angular_noise) % 360

        reading = {
            "rssi": round(noisy_rssi, 4),
            "bearing_deg": round(noisy_angle, 2),
            "estimated_distance_bracket": [max(0.0, dist - 2.5), dist + 2.5],
            "signature": self._signal_signature
        }

        # Check for spoofed/decoy beacons
        decoy_readings = []
        for decoy in self._decoys:
            d_pos = tuple(decoy["position"])
            d_dist = math.sqrt((d_pos[0] - ax) ** 2 + (d_pos[1] - ay) ** 2)
            d_base = 1.0 / (1.0 + 0.12 * d_dist)
            d_noisy = max(0.01, min(1.0, d_base + self.rng.gauss(0, decoy.get("noise_std", 0.08))))
            decoy_readings.append({
                "id": decoy["id"],
                "rssi": round(d_noisy, 4),
                "signature": decoy.get("signature", "UNKNOWN-RF-BEACON")
            })
        
        if decoy_readings:
            reading["decoy_signals"] = decoy_readings

        return reading

    def step(self, action: str, **kwargs) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """
        Executes an agent action in the environment.
        Returns: (observation, cost, done, info)
        """
        if self.is_done:
            raise RuntimeError("Mission has already terminated. Call reset() first.")

        self.step_count += 1
        cost = 0
        info = {"status": "ok", "message": ""}
        obs: Dict[str, Any] = {}

        action = action.upper()

        if action == "MOVE":
            direction = kwargs.get("direction", "NORTH").upper()
            dir_map = {
                "NORTH": (0, -1),
                "SOUTH": (0, 1),
                "EAST": (1, 0),
                "WEST": (-1, 0),
                "NORTH_EAST": (1, -1),
                "NORTH_WEST": (-1, -1),
                "SOUTH_EAST": (1, 1),
                "SOUTH_WEST": (-1, 1),
            }
            dx, dy = dir_map.get(direction, (0, 0))
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy

            cost = self.costs["move"]

            # Check boundary and obstacles
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                info["status"] = "blocked_boundary"
                info["blocked_coord"] = (nx, ny)
                info["message"] = f"WALL-E bumped into wasteland perimeter at ({nx}, {ny})."
            elif (nx, ny) in self.obstacles:
                info["status"] = "blocked_obstacle"
                info["blocked_coord"] = (nx, ny)
                info["message"] = f"Scrap obstacle blocked path at ({nx}, {ny})."
            else:
                self.agent_pos = (nx, ny)
                if (nx, ny) in self.hazards:
                    hazard_cost = self.costs.get("move_hazard_penalty", 3)
                    cost += hazard_cost
                    info["status"] = "hazard_incurred"
                    info["hazard_coord"] = (nx, ny)
                    info["message"] = f"WALL-E traversed sandstorm hazard at ({nx}, {ny}), costing +{hazard_cost} battery."
                else:
                    info["message"] = f"WALL-E moved {direction} to ({nx}, {ny})."

        elif action == "LOCAL_SCAN":
            cost = self.costs["local_scan"]
            radius = kwargs.get("radius", 2)
            obs["scan_result"] = self._get_local_scan(radius=radius)
            info["message"] = f"High-fidelity sensor sweep completed across radius {radius}."

        elif action == "LONG_RANGE_PING":
            cost = self.costs["long_range_ping"]
            obs["signal_reading"] = self._calculate_rf_signal()
            info["message"] = f"Sub-band radio pulse emitted. Received RSSI: {obs['signal_reading']['rssi']}."

        elif action == "ANALYZE_CLUE":
            cost = self.costs["analyze_clue"]
            if self.agent_pos in self._clue_map:
                clue = self._clue_map[self.agent_pos]
                self.discovered_clues.add(self.agent_pos)
                obs["clue_data"] = {
                    "id": clue["id"],
                    "type": clue["type"],
                    "description": clue["description"],
                    "evidence": clue.get("evidence", {})
                }
                info["message"] = f"Analyzed clue '{clue['id']}': {clue['description']}"
            else:
                info["status"] = "no_clue_here"
                info["message"] = f"No clue artifact present at current position {self.agent_pos}."

        elif action == "RECHARGE":
            if self.agent_pos in self.solar_stations:
                gain = self.costs.get("recharge_gain", 25)
                prev_bat = self.battery
                self.battery = min(self.max_battery, self.battery + gain)
                cost = 0  # Recharging restores energy rather than consuming it
                info["message"] = f"Solar panels deployed. Recharged from {prev_bat}% to {self.battery}%."
            else:
                cost = self.costs.get("wait", 1)
                info["status"] = "invalid_solar_spot"
                info["message"] = "Cannot recharge here: no solar concentration array."

        elif action == "WAIT":
            cost = self.costs.get("wait", 1)
            info["message"] = "WALL-E entered low-power idle standby."

        else:
            cost = 1
            info["status"] = "unknown_action"
            info["message"] = f"Unrecognized action '{action}'."

        # Apply battery deduction (unless recharge)
        if action != "RECHARGE":
            self.battery = max(0, self.battery - cost)

        # Check if WALL-E has reached EVE directly
        if self.agent_pos == self._true_eve_pos:
            self.eve_found = True
            info["eve_located"] = True
            info["message"] += " EVE HAS BEEN DIRECTLY LOCATED!"

        # Check termination conditions
        if self.battery <= 0:
            self.is_done = True
            info["termination_reason"] = "BATTERY_DEPLETED"
            info["message"] += " CRITICAL: Battery drained to 0%. WALL-E halted."
        elif self.step_count >= self.max_steps:
            self.is_done = True
            info["termination_reason"] = "MAX_STEPS_REACHED"
        elif self.eve_found:
            self.is_done = True
            info["termination_reason"] = "EVE_DISCOVERED"

        # Construct step observation
        obs.update({
            "agent_position": self.agent_pos,
            "battery": self.battery,
            "is_solar_station": self.agent_pos in self.solar_stations,
            "step_count": self.step_count,
            "is_done": self.is_done,
            "clues_collected_count": len(self.discovered_clues)
        })

        return obs, cost, self.is_done, info

    def get_ground_truth_for_evaluation(self) -> Dict[str, Any]:
        """
        Used ONLY by the evaluation/reporting engine after mission completion
        to grade how accurately WALL-E's belief matched reality.
        NEVER exposed to the agent during execution!
        """
        return {
            "eve_position": self._true_eve_pos,
            "eve_status": self._true_eve_status,
            "total_clues": len(self._clue_map),
            "clues_discovered": len(self.discovered_clues),
            "final_battery": self.battery,
            "final_steps": self.step_count,
            "success": self.eve_found
        }
