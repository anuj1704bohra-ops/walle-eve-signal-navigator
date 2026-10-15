"""
Trace Logger for WALL-E.
Maintains comprehensive, timestamped logs of every action, observation,
battery transition, and internal Bayesian reasoning step.
Outputs clean JSON and Markdown files.
"""

from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
from datetime import datetime


class TraceLogger:
    """Logs full POMDP action-observation traces for evaluation and visualization."""

    def __init__(self, mission_name: str, output_dir: str = "outputs/traces"):
        self.mission_name = mission_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traces: List[Dict[str, Any]] = []

    def record_step(
        self,
        step_number: int,
        agent_pos: tuple,
        battery: int,
        action: str,
        action_kwargs: dict,
        cost: int,
        observation: dict,
        info: dict,
        reasoning: str,
        entropy: float,
        top_hypotheses: list,
        agent_mode: str,
        heatmap_matrix: list = None
    ):
        """Records one execution step into the trace ledger."""
        entry = {
            "step": step_number,
            "timestamp": datetime.now().isoformat(),
            "agent_position": list(agent_pos),
            "battery_level": battery,
            "action": action,
            "action_params": action_kwargs,
            "battery_cost": cost,
            "agent_mode": agent_mode,
            "reasoning": reasoning,
            "belief_entropy_bits": entropy,
            "top_hypotheses": top_hypotheses,
            "belief_heatmap": heatmap_matrix,
            "observation_summary": {
                "scan_nearby_clue": observation.get("scan_result", {}).get("clue_nearby"),
                "eve_in_sight": observation.get("scan_result", {}).get("eve_in_sight", False),
                "rssi_reading": observation.get("signal_reading", {}).get("rssi") if "signal_reading" in observation else None,
                "bearing_reading": observation.get("signal_reading", {}).get("bearing_deg") if "signal_reading" in observation else None,
                "clue_collected": observation.get("clue_data", {}).get("id") if "clue_data" in observation else None,
            },
            "status_message": info.get("message", "")
        }
        self.traces.append(entry)

    def save_traces(self, env_metadata: dict = None) -> Tuple[Path, Path]:
        """Saves traces to both JSON and Markdown format."""
        slug = "".join(c if c.isalnum() else "_" for c in self.mission_name).lower()
        json_path = self.output_dir / f"trace_{slug}.json"
        md_path = self.output_dir / f"trace_{slug}.md"

        payload = {
            "mission_name": self.mission_name,
            "total_steps": len(self.traces),
            "recorded_at": datetime.now().isoformat(),
            "trace_history": self.traces
        }
        if env_metadata:
            payload.update(env_metadata)

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Save Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 📜 Mission Action Trace: {self.mission_name}\n\n")
            f.write(f"- **Total Steps Recorded:** {len(self.traces)}\n")
            f.write(f"- **Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Step | Mode | Action | Battery | Position | Entropy (bits) | Reason / Observation |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            
            for t in self.traces:
                action_str = t["action"]
                if t["action_params"]:
                    action_str += f" ({', '.join(f'{k}={v}' for k, v in t['action_params'].items())})"
                f.write(
                    f"| {t['step']} | `{t['agent_mode']}` | **{action_str}** | "
                    f"{t['battery_level']}% (-{t['battery_cost']}) | {t['agent_position']} | "
                    f"{t['belief_entropy_bits']} | {t['reasoning']} |\n"
                )

        return json_path, md_path
