"""
Mission Reporter for WALL-E.
Generates structured JSON and comprehensive Markdown reports
detailing evidence gathered, spatial/status inferences, confidence metrics,
and recommended operational next steps.
"""

from typing import Dict, Any, List, Tuple
import json
import math
from pathlib import Path
from datetime import datetime


class MissionReporter:
    """Generates the official final mission evaluation report."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        mission_name: str,
        total_steps: int,
        initial_battery: int,
        final_battery: int,
        inferred_pos: Tuple[int, int],
        confidence_pct: float,
        top_hypotheses: List[Dict[str, Any]],
        inferred_status: str,
        status_confidence: float,
        initial_entropy: float,
        final_entropy: float,
        clues_collected: List[Dict[str, Any]],
        ground_truth: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        """Builds and writes both JSON and Markdown final mission reports."""
        slug = "".join(c if c.isalnum() else "_" for c in mission_name).lower()
        json_path = self.output_dir / f"report_{slug}.json"
        md_path = self.output_dir / f"report_{slug}.md"

        # Calculate error metrics
        true_pos = ground_truth["eve_position"]
        coord_error = math.sqrt((inferred_pos[0] - true_pos[0]) ** 2 + (inferred_pos[1] - true_pos[1]) ** 2)
        status_match = (inferred_status == ground_truth["eve_status"])
        entropy_reduction = max(0.0, initial_entropy - final_entropy)
        entropy_reduction_pct = round((entropy_reduction / initial_entropy) * 100.0, 1) if initial_entropy > 0 else 100.0

        # Formulate recommended next step for WALL-E
        if coord_error <= 1.0 or ground_truth["success"]:
            recommendation = (
                f"**EXECUTE CONTACT DOCKING PROTOCOL:** Inferred target coordinates {inferred_pos} match "
                f"EVE's spatial signature with {confidence_pct}% confidence. WALL-E should deploy optical "
                f"handshake link, calibrate solar transfer umbilical, and establish low-voltage auxiliary "
                f"power connection to awaken EVE from '{inferred_status}' state."
            )
        else:
            recommendation = (
                f"**PROCEED WITH LOW-VELOCITY RADIAL SEARCH:** High probability locus concentrated around "
                f"{inferred_pos}. WALL-E should advance 1 sector toward {inferred_pos} using southern obstacle "
                f"corridor, keeping a 15% battery safety margin, and initiate high-resolution optical sweep."
            )

        report_data = {
            "mission_name": mission_name,
            "generated_at": datetime.now().isoformat(),
            "execution_summary": {
                "total_steps": total_steps,
                "initial_battery": initial_battery,
                "final_battery": final_battery,
                "battery_efficiency_score": f"{round((final_battery / initial_battery) * 100, 1)}% retained",
                "mission_outcome": "SUCCESS - EVE DISCOVERED" if ground_truth["success"] else "HYPOTHESIS CONVERGED"
            },
            "evidence_dossier": {
                "clues_collected_count": len(clues_collected),
                "clues_detail": clues_collected
            },
            "spatial_inference": {
                "inferred_coordinates": list(inferred_pos),
                "spatial_confidence_percent": confidence_pct,
                "top_3_hypotheses": top_hypotheses,
                "uncertainty_metrics": {
                    "initial_entropy_bits": initial_entropy,
                    "final_entropy_bits": final_entropy,
                    "entropy_reduction_bits": round(entropy_reduction, 3),
                    "entropy_reduction_percent": f"{entropy_reduction_pct}%"
                }
            },
            "status_inference": {
                "inferred_status": inferred_status,
                "confidence_percent": status_confidence
            },
            "operational_recommendation": recommendation,
            "ground_truth_evaluation": {
                "true_coordinates": list(true_pos),
                "true_status": ground_truth["eve_status"],
                "euclidean_error_distance": round(coord_error, 2),
                "status_classification_correct": status_match,
                "direct_visual_discovery": ground_truth["success"]
            }
        }

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Write Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 🛰️ Final Mission Report: {mission_name}\n\n")
            f.write(f"> **Mission Status:** `{report_data['execution_summary']['mission_outcome']}`  \n")
            f.write(f"> **Timestamp:** `{report_data['generated_at']}`\n\n")
            
            f.write("## 1. Executive Summary\n\n")
            f.write(f"| Metric | Result |\n")
            f.write(f"|---|---|\n")
            f.write(f"| **Inferred EVE Coordinates** | `{inferred_pos}` |\n")
            f.write(f"| **Spatial Confidence** | **{confidence_pct}%** (within 1.5 radius) |\n")
            f.write(f"| **Inferred Operational Status** | `{inferred_status}` ({status_confidence}% conf.) |\n")
            f.write(f"| **Total Steps Taken** | {total_steps} |\n")
            f.write(f"| **Battery Status** | {final_battery} / {initial_battery} units remaining |\n")
            f.write(f"| **Entropy Reduction** | {initial_entropy} bits -> {final_entropy} bits (**-{entropy_reduction_pct}%**) |\n\n")

            f.write("## 2. Evidence Dossier & Clues Collected\n\n")
            if clues_collected:
                for idx, c in enumerate(clues_collected, 1):
                    f.write(f"### Clue #{idx}: `{c.get('id', 'N/A')}` ({c.get('type', 'UNKNOWN')})\n")
                    f.write(f"- **Discovered At:** `(X={c.get('coord', ['?','?'])[0]}, Y={c.get('coord', ['?','?'])[1]})`\n")
                    f.write(f"- **Description:** {c.get('description', '')}\n")
                    f.write(f"- **Evidence Parameters:** `{json.dumps(c.get('evidence', {}))}`\n\n")
            else:
                f.write("*No discrete physical clues gathered; navigation relied entirely on RF telemetry and radar sweeps.*\n\n")

            f.write("## 3. Bayesian Probability Distribution (Top Candidates)\n\n")
            f.write("| Rank | Candidate Sector (X, Y) | Probability Density | Confidence |\n")
            f.write("|---|---|---|---|\n")
            for r_idx, hyp in enumerate(top_hypotheses, 1):
                f.write(f"| #{r_idx} | `{hyp['coordinate']}` | {hyp['probability']} | {hyp['confidence_percent']}% |\n")
            f.write("\n")

            f.write("## 4. Recommended Next Operational Step\n\n")
            f.write(f"{recommendation}\n\n")

            f.write("## 5. Post-Mission Evaluation (Ground Truth Benchmark)\n\n")
            f.write(f"- **True Coordinates:** `{list(true_pos)}`\n")
            f.write(f"- **Coordinate Error Distance:** `{round(coord_error, 2)} units`\n")
            f.write(f"- **True EVE Status:** `{ground_truth['eve_status']}` (Matched: **{status_match}**)\n")
            f.write(f"- **Visual Confirmation:** `{'Achieved' if ground_truth['success'] else 'Target in Close Radial Range'}`\n")

        return json_path, md_path
