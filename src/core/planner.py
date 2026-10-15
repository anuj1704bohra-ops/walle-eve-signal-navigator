"""
A* Pathfinding & Resource-Safe Planner for WALL-E.
Calculates optimal routes, avoids obstacles/hazards, guarantees battery return safety,
and selects high-utility exploration targets.
"""

from typing import Dict, Any, Tuple, List, Optional, Set
import heapq
import math


class Node:
    """A* Search Node."""
    def __init__(self, coord: Tuple[int, int], g: float, h: float, parent=None):
        self.coord = coord
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f


class MissionPlanner:
    """
    Handles pathfinding, safety margins for battery management,
    and exploration target selection.
    """

    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]], 
                 solar_stations: Set[Tuple[int, int]], hazards: Set[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.solar_stations = set(solar_stations)
        self.hazards = set(hazards)

    def add_obstacle(self, coord: Tuple[int, int]):
        """Dynamically add newly discovered obstacle."""
        self.obstacles.add(coord)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible for 4-directional grid)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, coord: Tuple[int, int]) -> List[Tuple[Tuple[int, int], str]]:
        """Returns valid adjacent 4-directional neighboring cells and direction strings."""
        x, y = coord
        directions = [
            ((x, y - 1), "NORTH"),
            ((x, y + 1), "SOUTH"),
            ((x + 1, y), "EAST"),
            ((x - 1, y), "WEST"),
        ]
        valid = []
        for n_coord, dir_name in directions:
            nx, ny = n_coord
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if n_coord not in self.obstacles:
                    valid.append((n_coord, dir_name))
        return valid

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  move_cost: int = 2, hazard_penalty: int = 3) -> Optional[List[Tuple[Tuple[int, int], str]]]:
        """
        A* algorithm to find lowest-battery cost path from start to goal.
        Returns list of ((next_x, next_y), direction_name).
        """
        if start == goal:
            return []

        open_set: List[Node] = []
        start_node = Node(start, g=0.0, h=self.heuristic(start, goal))
        heapq.heappush(open_set, start_node)

        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}

        while open_set:
            current = heapq.heappop(open_set)

            if current.coord == goal:
                # Reconstruct path
                path = []
                curr = goal
                while curr in came_from:
                    prev_coord, dir_name = came_from[curr]
                    path.append((curr, dir_name))
                    curr = prev_coord
                path.reverse()
                return path

            for neighbor_coord, dir_name in self.get_neighbors(current.coord):
                step_cost = move_cost
                if neighbor_coord in self.hazards:
                    step_cost += hazard_penalty

                new_cost = current.g + step_cost
                if neighbor_coord not in cost_so_far or new_cost < cost_so_far[neighbor_coord]:
                    cost_so_far[neighbor_coord] = new_cost
                    neighbor_node = Node(neighbor_coord, g=new_cost, h=self.heuristic(neighbor_coord, goal), parent=current)
                    heapq.heappush(open_set, neighbor_node)
                    came_from[neighbor_coord] = (current.coord, dir_name)

        return None  # No path found

    def estimate_path_battery_cost(self, path: List[Tuple[Tuple[int, int], str]], 
                                   move_cost: int = 2, hazard_penalty: int = 3) -> int:
        """Calculates total battery units required to traverse path."""
        total = 0
        for coord, _ in path:
            total += move_cost
            if coord in self.hazards:
                total += hazard_penalty
        return total

    def find_nearest_solar_station(self, current_pos: Tuple[int, int], 
                                  move_cost: int = 2, hazard_penalty: int = 3) -> Tuple[Optional[Tuple[int, int]], int, Optional[List[Tuple[Tuple[int, int], str]]]]:
        """
        Finds the closest accessible solar station and its battery return cost.
        Returns: (station_coord, battery_cost, path)
        """
        best_station = None
        min_cost = float('inf')
        best_path = None

        for station in self.solar_stations:
            path = self.find_path(current_pos, station, move_cost, hazard_penalty)
            if path is not None:
                cost = self.estimate_path_battery_cost(path, move_cost, hazard_penalty)
                if cost < min_cost:
                    min_cost = cost
                    best_station = station
                    best_path = path

        if best_station is None:
            return None, 999, None

        return best_station, int(min_cost), best_path

    def is_battery_critical(self, current_pos: Tuple[int, int], current_battery: int, 
                            safety_margin: int = 8, move_cost: int = 2, hazard_penalty: int = 3) -> Tuple[bool, int, Optional[Tuple[int, int]], Optional[List[Tuple[Tuple[int, int], str]]]]:
        """
        FAIL-SAFE CHECK: Evaluates if WALL-E must turn back to solar pad immediately.
        Returns: (is_critical, return_cost, nearest_station, path_to_station)
        """
        nearest_station, return_cost, path = self.find_nearest_solar_station(current_pos, move_cost, hazard_penalty)
        
        # If already at a solar station and battery is not full
        if current_pos in self.solar_stations:
            return False, 0, current_pos, []

        required_reserve = return_cost + safety_margin
        is_critical = (current_battery <= required_reserve)
        return is_critical, return_cost, nearest_station, path
