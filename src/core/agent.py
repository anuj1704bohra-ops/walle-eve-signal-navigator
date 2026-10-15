"""
Autonomous WALL-E Agent Implementation.
Fuses Bayesian belief updating, A* path planning, resource fail-safes,
and deliberate reasoning traces to locate EVE.
"""

from typing import Dict, Any, Tuple, List, Optional
import math
from src.core.belief import SpatialBeliefEngine
from src.core.planner import MissionPlanner


class AutonomousWallEAgent:
    """
    Autonomous agent controlling WALL-E's decisions, movements, and observations.
    Operates under partial observability and maintains its own internal belief model.
    """

    def __init__(self, initial_observation: Dict[str, Any], costs: Dict[str, int]):
        self.width, self.height = initial_observation["grid_dimensions"]
        self.current_pos = initial_observation["agent_position"]
        self.battery = initial_observation["battery"]
        self.max_battery = initial_observation["max_battery"]
        self.costs = costs
        
        # Knowledge discovered by WALL-E during mission
        self.known_obstacles = set()
        self.known_solar_stations = set()
        for s in initial_observation.get("solar_stations", []):
            self.known_solar_stations.add(tuple(s))
        if initial_observation.get("is_solar_station"):
            self.known_solar_stations.add(self.current_pos)
            
        self.known_hazards = set()
        self.known_clues_pending = []  # list of (x, y) coordinates
        self.analyzed_clues = []
        
        # Internal AI components
        self.belief = SpatialBeliefEngine(self.width, self.height, self.known_obstacles)
        self.planner = MissionPlanner(
            self.width, self.height, 
            self.known_obstacles, self.known_solar_stations, self.known_hazards
        )
        
        # Apply initial hints
        init_sig = initial_observation.get("initial_signal", {})
        dir_hint = init_sig.get("direction_hint")
        if dir_hint:
            self.belief.apply_initial_hint(dir_hint)

        # Agent state
        self.state_mode = "INITIALIZING"
        self.current_path = []
        self.target_goal = None
        self.steps_since_last_ping = 0
        self.ping_interval = 4
        self.eve_confirmed_pos = None

    def process_observation(self, observation: Dict[str, Any], info: Dict[str, Any]):
        """Updates agent's internal state, knowledge map, and Bayesian belief."""
        self.current_pos = observation["agent_position"]
        self.battery = observation["battery"]
        
        # Check solar pad status
        if observation.get("is_solar_station"):
            self.known_solar_stations.add(self.current_pos)
            self.planner.solar_stations.add(self.current_pos)

        # Process local scan results if available
        if "scan_result" in observation:
            scan = observation["scan_result"]
            scanned_coords = []
            for cell in scan.get("cells", []):
                coord = tuple(cell["coord"])
                scanned_coords.append(coord)
                
                if cell["type"] == "obstacle":
                    self.known_obstacles.add(coord)
                    self.planner.add_obstacle(coord)
                elif cell["type"] == "solar_pad":
                    self.known_solar_stations.add(coord)
                    self.planner.solar_stations.add(coord)
                elif cell["type"] == "hazard":
                    self.known_hazards.add(coord)
                    self.planner.hazards.add(coord)

                if cell.get("has_clue") and coord not in self.known_clues_pending and coord not in [c["coord"] for c in self.analyzed_clues]:
                    self.known_clues_pending.append(coord)

            # Negative observation update (if EVE was not seen in this local scan)
            if not scan.get("eve_in_sight"):
                self.belief.update_negative_observation(scanned_coords)
            else:
                self.eve_confirmed_pos = self.current_pos

        # Process radio signal ping
        if "signal_reading" in observation:
            reading = observation["signal_reading"]
            self.belief.update_with_rf_signal(
                self.current_pos,
                rssi=reading["rssi"],
                bearing_deg=reading["bearing_deg"]
            )
            self.steps_since_last_ping = 0

        # Process analyzed clue
        if "clue_data" in observation:
            clue = observation["clue_data"]
            clue["coord"] = self.current_pos
            self.analyzed_clues.append(clue)
            if self.current_pos in self.known_clues_pending:
                self.known_clues_pending.remove(self.current_pos)
            self.belief.update_with_clue_evidence(clue)

        # Handle collision or hazard detection from move outcome
        if info.get("status") == "blocked_obstacle" and "blocked_coord" in info:
            b_coord = tuple(info["blocked_coord"])
            self.known_obstacles.add(b_coord)
            self.planner.add_obstacle(b_coord)
            if 0 <= b_coord[0] < self.width and 0 <= b_coord[1] < self.height:
                self.belief.grid[b_coord[1], b_coord[0]] = 0.0
                self.belief._normalize()

        if info.get("status") == "hazard_incurred" and "hazard_coord" in info:
            h_coord = tuple(info["hazard_coord"])
            self.known_hazards.add(h_coord)
            self.planner.hazards.add(h_coord)

        # Direct EVE detection
        if info.get("eve_located") or observation.get("is_done") and info.get("termination_reason") == "EVE_DISCOVERED":
            self.eve_confirmed_pos = self.current_pos
            self.belief.set_direct_detection(self.current_pos)

    def decide_next_action(self) -> Tuple[str, Dict[str, Any], str]:
        """
        Core decision engine.
        Returns: (action_name, action_kwargs, reasoning_text)
        """
        self.steps_since_last_ping += 1
        
        # 1. FAIL-SAFE: Battery Safety Check
        is_critical, return_cost, nearest_station, return_path = self.planner.is_battery_critical(
            self.current_pos, self.battery, safety_margin=8,
            move_cost=self.costs.get("move", 2),
            hazard_penalty=self.costs.get("move_hazard_penalty", 3)
        )

        if is_critical and nearest_station:
            self.state_mode = "EMERGENCY_RECHARGE"
            if return_path and len(return_path) > 0:
                next_coord, direction = return_path[0]
                reasoning = (
                    f"CRITICAL BATTERY LEVEL ({self.battery}%). Return cost to solar pad at {nearest_station} "
                    f"is {return_cost}% (plus 8% safety buffer). Aborting exploration and backtracking {direction} to recharge."
                )
                return "MOVE", {"direction": direction}, reasoning

        # 2. RECHARGE IF AT SOLAR PAD AND BATTERY < 90%
        if self.current_pos in self.known_solar_stations and self.battery <= self.max_battery - 10:
            self.state_mode = "RECHARGING"
            reasoning = (
                f"Currently docked at solar charging pad {self.current_pos} with battery at {self.battery}%. "
                f"Deploying solar arrays to replenish +{self.costs.get('recharge_gain', 25)}% energy."
            )
            return "RECHARGE", {}, reasoning

        # 3. INTERACT WITH CLUE IF SITTING ON ONE
        if self.current_pos in self.known_clues_pending:
            self.state_mode = "ANALYZING_CLUE"
            reasoning = (
                f"Unexamined environmental artifact detected directly under chassis at {self.current_pos}. "
                f"Executing deep sensor spectrometry to extract EVE coordinates."
            )
            return "ANALYZE_CLUE", {}, reasoning

        # 4. PERIODIC LONG-RANGE RF TRIANGULATION (Only if not critically low)
        if (self.steps_since_last_ping >= self.ping_interval or self.state_mode == "INITIALIZING") and self.battery > 25:
            self.state_mode = "SCANNING_BEACON"
            entropy = self.belief.get_shannon_entropy()
            reasoning = (
                f"Triangulation cycle active (step count {self.steps_since_last_ping}). Current belief entropy is {entropy} bits. "
                f"Emitting long-range RF pulse to refine distance & bearing vectors."
            )
            return "LONG_RANGE_PING", {}, reasoning

        # 5. INVESTIGATE KNOWN CLUES NEARBY
        if self.known_clues_pending:
            target_clue = self.known_clues_pending[0]
            path = self.planner.find_path(self.current_pos, target_clue)
            if path and len(path) > 0:
                self.state_mode = "APPROACHING_CLUE"
                next_coord, direction = path[0]
                reasoning = (
                    f"Investigating undiscovered clue beacon at {target_clue} ({len(path)} steps away). "
                    f"Navigating {direction} towards clue site."
                )
                return "MOVE", {"direction": direction}, reasoning

        # 6. LOCAL SCAN IF ENTERING UNCHARTED REGION OR HIGH BELIEF CELL
        top_coord = self.belief.get_highest_belief_coordinate()
        dist_to_top = abs(self.current_pos[0] - top_coord[0]) + abs(self.current_pos[1] - top_coord[1])
        if dist_to_top <= 2 and self.steps_since_last_ping > 1:
            self.state_mode = "LOCAL_RADAR_SWEEP"
            reasoning = (
                f"In close proximity (distance {dist_to_top}) to highest probability hypothesis {top_coord}. "
                f"Deploying high-resolution local scan to spot EVE or ground markers."
            )
            return "LOCAL_SCAN", {"radius": 2}, reasoning

        # 7. CONVERGE TOWARDS BAYESIAN MAP (MAXIMUM A POSTERIORI) TARGET
        best_coord = self.belief.get_highest_belief_coordinate()
        path_to_target = self.planner.find_path(self.current_pos, best_coord)

        if path_to_target and len(path_to_target) > 0:
            self.state_mode = "TRACKING_EVE"
            next_coord, direction = path_to_target[0]
            entropy = self.belief.get_shannon_entropy()
            top_hyp = self.belief.get_top_hypotheses(1)[0]
            reasoning = (
                f"Bayesian hypothesis points to target sector {best_coord} (confidence: {top_hyp['confidence_percent']}%, entropy: {entropy} bits). "
                f"Advancing {direction} along A* calculated trajectory."
            )
            return "MOVE", {"direction": direction}, reasoning

        # Fallback: Perform local sweep or small exploratory move
        self.state_mode = "EXPLORATION_SWEEP"
        reasoning = "Executing wide-area perimeter scan to discover path around obstacles."
        return "LOCAL_SCAN", {"radius": 1}, reasoning
