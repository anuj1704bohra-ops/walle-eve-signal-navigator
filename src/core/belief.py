"""
Bayesian Belief Updating Engine for WALL-E.
Maintains a 2D spatial posterior distribution P(EVE at (x,y))
and a categorical distribution over EVE's operational status.
Computes Shannon Entropy to quantify uncertainty reduction.
"""

from typing import Dict, Any, Tuple, List, Optional
import math
import numpy as np


class SpatialBeliefEngine:
    """
    Bayesian Grid Filter representing WALL-E's probabilistic belief
    about EVE's spatial location across the wasteland grid.
    """

    def __init__(self, width: int, height: int, obstacles: Optional[set] = None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
        
        # Initialize 2D grid probability matrix
        self.grid = np.zeros((height, width), dtype=np.float64)
        
        # Categorical belief over EVE's status
        self.status_hypotheses = {
            "operational": 0.25,
            "standby_beacon_lock": 0.35,
            "low_power_hibernation": 0.25,
            "rebooting_diagnostic": 0.15
        }
        
        self._init_uniform_prior()

    def _init_uniform_prior(self):
        """Initializes uniform prior over all non-obstacle cells."""
        valid_cells = 0
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.obstacles:
                    self.grid[y, x] = 1.0
                    valid_cells += 1
                else:
                    self.grid[y, x] = 0.0

        if valid_cells > 0:
            self.grid /= np.sum(self.grid)

    def apply_initial_hint(self, hint: str):
        """
        Applies a weak initial prior bias based on coarse telemetry
        (e.g. SOUTH_EAST, NORTH_WEST).
        """
        if not hint:
            return

        cx, cy = self.width / 2.0, self.height / 2.0
        hint = hint.upper()

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.obstacles:
                    continue
                weight = 1.0
                if "SOUTH" in hint and y >= cy:
                    weight *= 2.0
                elif "NORTH" in hint and y < cy:
                    weight *= 2.0
                
                if "EAST" in hint and x >= cx:
                    weight *= 2.0
                elif "WEST" in hint and x < cx:
                    weight *= 2.0

                self.grid[y, x] *= weight

        self._normalize()

    def update_with_rf_signal(self, wall_e_pos: Tuple[int, int], rssi: float, bearing_deg: float, noise_std: float = 0.08):
        """
        Bayesian update given an RF signal ping.
        P(EVE=(x,y) | z) proportional to P(z_rssi | x,y) * P(z_bearing | x,y) * P(EVE=(x,y))
        """
        wx, wy = wall_e_pos
        likelihood = np.zeros_like(self.grid)

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.obstacles or (x, y) == (wx, wy):
                    continue

                dist = math.sqrt((x - wx) ** 2 + (y - wy) ** 2)
                expected_rssi = 1.0 / (1.0 + 0.12 * dist)
                rssi_likelihood = math.exp(-0.5 * ((rssi - expected_rssi) / noise_std) ** 2)

                # Expected bearing from WALL-E to candidate (x, y)
                dx = x - wx
                dy = y - wy
                expected_bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                angle_diff = abs(bearing_deg - expected_bearing)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Wrapped normal likelihood for bearing (std ~ 25 degrees)
                bearing_likelihood = math.exp(-0.5 * (angle_diff / 25.0) ** 2)

                likelihood[y, x] = rssi_likelihood * (0.6 * bearing_likelihood + 0.4)

        self.grid *= likelihood
        self._normalize()

    def update_negative_observation(self, scanned_cells: List[Tuple[int, int]]):
        """
        Negative evidence update: if WALL-E scanned a cell and EVE was not visible,
        drop the probability of that cell to near zero.
        """
        for (x, y) in scanned_cells:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] *= 0.001  # extremely low probability
        self._normalize()

    def update_with_clue_evidence(self, clue_data: Dict[str, Any]):
        """
        Fuses structured environmental clues into the spatial belief.
        Handles coordinate boundaries, proximity radii, and decoy warnings.
        """
        evidence = clue_data.get("evidence", {})
        x_min = evidence.get("x_min")
        x_max = evidence.get("x_max")
        y_min = evidence.get("y_min")
        y_max = evidence.get("y_max")
        prox_radius = evidence.get("proximity_radius")
        disqualify_region = evidence.get("disqualify_region")

        for y in range(self.height):
            for x in range(self.width):
                multiplier = 1.0

                if x_min is not None and x < x_min:
                    multiplier *= 0.01
                if x_max is not None and x > x_max:
                    multiplier *= 0.01
                if y_min is not None and y < y_min:
                    multiplier *= 0.01
                if y_max is not None and y > y_max:
                    multiplier *= 0.01

                if prox_radius is not None:
                    # Clue location is the center of the proximity sphere
                    c_pos = clue_data.get("position")
                    if c_pos:
                        d = math.sqrt((x - c_pos[0]) ** 2 + (y - c_pos[1]) ** 2)
                        if d <= prox_radius:
                            multiplier *= 3.0
                        else:
                            multiplier *= 0.2

                if disqualify_region:
                    dx_max = disqualify_region.get("x_max", self.width)
                    dy_min = disqualify_region.get("y_min", 0)
                    if x <= dx_max and y >= dy_min:
                        multiplier *= 0.02  # heavily penalize decoy sector

                self.grid[y, x] *= multiplier

        # Update status hypotheses based on clue type
        clue_type = clue_data.get("type", "")
        if "ION" in clue_type or "HOVER" in clue_type:
            self.status_hypotheses["standby_beacon_lock"] += 0.25
        elif "PLANT" in clue_type:
            self.status_hypotheses["operational"] += 0.35
        elif "CORRUPT" in clue_type or "HIBERNATION" in clue_type:
            self.status_hypotheses["low_power_hibernation"] += 0.30

        self._normalize_status()
        self._normalize()

    def set_direct_detection(self, eve_pos: Tuple[int, int]):
        """Pinpoints EVE's location with 100% certainty when found."""
        self.grid.fill(0.0)
        self.grid[eve_pos[1], eve_pos[0]] = 1.0

    def _normalize(self):
        """Normalizes grid so total probability mass equals 1.0."""
        total = np.sum(self.grid)
        if total > 1e-12:
            self.grid /= total
        else:
            self._init_uniform_prior()

    def _normalize_status(self):
        """Normalizes status categorical distribution."""
        total = sum(self.status_hypotheses.values())
        if total > 0:
            for k in self.status_hypotheses:
                self.status_hypotheses[k] = round(self.status_hypotheses[k] / total, 4)

    def get_shannon_entropy(self) -> float:
        """
        Computes Shannon Information Entropy H(B) in bits.
        H(B) = - sum( P(x,y) * log2(P(x,y)) )
        Represents uncertainty: drops sharply as clues & pings eliminate hypotheses.
        """
        flattened = self.grid.flatten()
        non_zero = flattened[flattened > 1e-10]
        entropy = -float(np.sum(non_zero * np.log2(non_zero)))
        return round(entropy, 3)

    def get_top_hypotheses(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Returns the top K coordinates with the highest posterior probability."""
        flat_indices = np.argsort(self.grid.ravel())[::-1][:top_k]
        results = []
        for idx in flat_indices:
            y, x = np.unravel_index(idx, self.grid.shape)
            prob = float(self.grid[y, x])
            results.append({
                "coordinate": (int(x), int(y)),
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100, 2)
            })
        return results

    def get_highest_belief_coordinate(self) -> Tuple[int, int]:
        """Returns (x, y) coordinate with maximum probability."""
        idx = np.argmax(self.grid)
        y, x = np.unravel_index(idx, self.grid.shape)
        return int(x), int(y)

    def get_confidence_radius(self, center: Tuple[int, int], radius: float = 1.5) -> float:
        """
        Calculates cumulative probability mass within radius of a target point.
        Used to calculate confidence % in the final mission report.
        """
        cx, cy = center
        mass = 0.0
        for y in range(self.height):
            for x in range(self.width):
                if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius:
                    mass += self.grid[y, x]
        return round(float(mass) * 100.0, 2)

    def get_inferred_status(self) -> Tuple[str, float]:
        """Returns (most_likely_status, confidence)."""
        best_status = max(self.status_hypotheses.items(), key=lambda item: item[1])
        return best_status[0], round(best_status[1] * 100.0, 1)

    def get_heatmap_matrix(self) -> List[List[float]]:
        """Returns 2D grid as standard Python float matrix for serialization/UI."""
        return [[round(float(val), 5) for val in row] for row in self.grid]
