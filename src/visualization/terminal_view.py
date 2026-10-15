"""
Terminal ASCII / ANSI Visualizer for WALL-E's search mission.
Renders colorful grid, telemetry status bars, and real-time step observations.
"""

from typing import Dict, Any, Tuple, Set


class TerminalVisualizer:
    """ASCII Terminal renderer for WALL-E's mission."""

    # ANSI Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"

    @staticmethod
    def render_grid(
        width: int,
        height: int,
        agent_pos: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        solar_stations: Set[Tuple[int, int]],
        hazards: Set[Tuple[int, int]],
        clues_discovered: Set[Tuple[int, int]],
        top_hypothesis: Tuple[int, int],
        eve_found: bool = False
    ) -> str:
        """Constructs ANSI ASCII grid representation."""
        lines = []
        # Header
        col_indices = "   " + "".join(f"{x:2d}" for x in range(width))
        lines.append(f"{TerminalVisualizer.GRAY}{col_indices}{TerminalVisualizer.RESET}")

        for y in range(height):
            row_str = f"{TerminalVisualizer.GRAY}{y:2d}{TerminalVisualizer.RESET} "
            for x in range(width):
                coord = (x, y)
                if coord == agent_pos:
                    row_str += f"{TerminalVisualizer.BOLD}{TerminalVisualizer.YELLOW}🤖{TerminalVisualizer.RESET}"
                elif coord == top_hypothesis and eve_found:
                    row_str += f"{TerminalVisualizer.BOLD}{TerminalVisualizer.GREEN}✨{TerminalVisualizer.RESET}"
                elif coord == top_hypothesis:
                    row_str += f"{TerminalVisualizer.BOLD}{TerminalVisualizer.MAGENTA}🎯{TerminalVisualizer.RESET}"
                elif coord in clues_discovered:
                    row_str += f"{TerminalVisualizer.CYAN}🔍{TerminalVisualizer.RESET}"
                elif coord in solar_stations:
                    row_str += f"{TerminalVisualizer.GREEN}⚡{TerminalVisualizer.RESET}"
                elif coord in obstacles:
                    row_str += f"{TerminalVisualizer.GRAY}🧱{TerminalVisualizer.RESET}"
                elif coord in hazards:
                    row_str += f"{TerminalVisualizer.RED}💨{TerminalVisualizer.RESET}"
                else:
                    row_str += f"{TerminalVisualizer.GRAY} ·{TerminalVisualizer.RESET}"
            lines.append(row_str)

        legend = (
            f"\n{TerminalVisualizer.BOLD}Legend:{TerminalVisualizer.RESET} "
            f"{TerminalVisualizer.YELLOW}🤖 WALL-E{TerminalVisualizer.RESET} | "
            f"{TerminalVisualizer.MAGENTA}🎯 Top Hypothesis{TerminalVisualizer.RESET} | "
            f"{TerminalVisualizer.GREEN}⚡ Solar Pad{TerminalVisualizer.RESET} | "
            f"{TerminalVisualizer.CYAN}🔍 Clue{TerminalVisualizer.RESET} | "
            f"{TerminalVisualizer.GRAY}🧱 Scrap Obstacle{TerminalVisualizer.RESET} | "
            f"{TerminalVisualizer.RED}💨 Hazard{TerminalVisualizer.RESET}"
        )
        lines.append(legend)
        return "\n".join(lines)

    @staticmethod
    def render_telemetry(step: int, battery: int, max_battery: int, mode: str, 
                         entropy: float, action: str, reasoning: str) -> str:
        """Renders battery bar, step status, and agent thinking string."""
        pct = max(0, min(100, int((battery / max_battery) * 100)))
        bar_len = 20
        filled = int((pct / 100) * bar_len)
        empty = bar_len - filled
        
        color = TerminalVisualizer.GREEN
        if pct < 25:
            color = TerminalVisualizer.RED
        elif pct < 50:
            color = TerminalVisualizer.YELLOW

        bar = f"{color}[{'█' * filled}{'░' * empty}] {battery}/{max_battery} ({pct}%){TerminalVisualizer.RESET}"

        status_box = [
            f"┌{'─' * 76}┐",
            f"│ {TerminalVisualizer.BOLD}Step: {step:3d}{TerminalVisualizer.RESET} | Battery: {bar} | Entropy: {TerminalVisualizer.CYAN}{entropy:.2f} bits{TerminalVisualizer.RESET} │",
            f"│ Mode: {TerminalVisualizer.MAGENTA}{mode:20s}{TerminalVisualizer.RESET} | Action: {TerminalVisualizer.BOLD}{action:25s}{TerminalVisualizer.RESET} │",
            f"├{'─' * 76}┤",
            f"│ {TerminalVisualizer.CYAN}Thought:{TerminalVisualizer.RESET} {reasoning[:66]:66s} │",
        ]
        if len(reasoning) > 66:
            status_box.append(f"│         {reasoning[66:132]:66s} │")
        status_box.append(f"└{'─' * 76}┘")

        return "\n".join(status_box)
