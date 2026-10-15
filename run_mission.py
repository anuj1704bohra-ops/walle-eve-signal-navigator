"""
Main Entry Point: WALL-E and EVE's Signal Autonomous Search Mission.
Executes the simulation, coordinates the agent, logs traces, generates final reports,
and provides both terminal and web visualization modes.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser

# Ensure Windows PowerShell/cmd stdout handles Unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.environment import WastelandEnvironment
from src.core.agent import AutonomousWallEAgent
from src.utils.logger import TraceLogger
from src.utils.reporter import MissionReporter
from src.visualization.terminal_view import TerminalVisualizer


def run_simulation(config_path: str, live_terminal: bool = False, delay: float = 0.1):
    """Runs a complete autonomous search mission."""
    print(f"\n🚀 Initializing WALL-E Autonomous Mission from: {config_path}")
    
    # 1. Initialize Environment
    env = WastelandEnvironment(config_path)
    initial_obs = env.reset()
    
    # 2. Initialize Agent Brain & Logging
    agent = AutonomousWallEAgent(initial_obs, env.costs)
    mission_name = env.config.get("mission_name", "WALL-E Mission")
    logger = TraceLogger(mission_name)
    reporter = MissionReporter()

    initial_entropy = agent.belief.get_shannon_entropy()
    initial_battery = env.battery

    print(f"🌍 Map Dimensions: {env.width}x{env.height} | Initial Battery: {initial_battery}% | Initial Entropy: {initial_entropy} bits")
    print("=" * 80)

    done = False
    step = 0

    while not done:
        step += 1
        bat_before = env.battery
        agent_pos_before = env.agent_pos

        # Agent deliberates and decides next move
        action, action_kwargs, reasoning = agent.decide_next_action()

        # Environment processes the action
        obs, cost, done, info = env.step(action, **action_kwargs)

        # Agent perceives outcome and updates Bayesian belief
        agent.process_observation(obs, info)

        entropy_after = agent.belief.get_shannon_entropy()
        top_hyp = agent.belief.get_top_hypotheses(3)

        # Record step in trace ledger
        logger.record_step(
            step_number=step,
            agent_pos=agent_pos_before,
            battery=bat_before,
            action=action,
            action_kwargs=action_kwargs,
            cost=cost,
            observation=obs,
            info=info,
            reasoning=reasoning,
            entropy=entropy_after,
            top_hypotheses=top_hyp,
            agent_mode=agent.state_mode,
            heatmap_matrix=agent.belief.get_heatmap_matrix()
        )

        # Live Terminal Visualizer
        if live_terminal:
            # Clear screen (ANSI)
            print("\033[2J\033[H", end="")
            print(f"\n🛰️  {TerminalVisualizer.BOLD}{mission_name}{TerminalVisualizer.RESET}")
            grid_str = TerminalVisualizer.render_grid(
                width=env.width,
                height=env.height,
                agent_pos=env.agent_pos,
                obstacles=env.obstacles,
                solar_stations=env.solar_stations,
                hazards=env.hazards,
                clues_discovered=env.discovered_clues,
                top_hypothesis=agent.belief.get_highest_belief_coordinate(),
                eve_found=env.eve_found
            )
            print(grid_str)
            telem_str = TerminalVisualizer.render_telemetry(
                step=step,
                battery=env.battery,
                max_battery=env.max_battery,
                mode=agent.state_mode,
                entropy=entropy_after,
                action=f"{action} {action_kwargs.get('direction', '')}".strip(),
                reasoning=reasoning
            )
            print(telem_str)
            time.sleep(delay)

    # Final Post-Mission Processing
    print("\n" + "=" * 80)
    print(f"🏁 Mission Completed in {step} steps! Reason: {info.get('termination_reason', 'FINISHED')}")

    # Save traces with map layout metadata
    env_meta = {
        "grid_dimensions": [env.width, env.height],
        "obstacles": [list(o) for o in env.obstacles],
        "solar_recharge_stations": [list(s) for s in env.solar_stations],
        "hazards": [list(h) for h in env.hazards],
        "clues": [{"id": c["id"], "position": list(c["position"])} for c in env.config.get("clues", [])],
        "eve_target_ground_truth": list(env._true_eve_pos)
    }
    json_trace, md_trace = logger.save_traces(env_metadata=env_meta)
    print(f"💾 Full action trace saved to:\n  - JSON: {json_trace}\n  - Markdown: {md_trace}")

    # Generate official report
    inferred_coord = agent.belief.get_highest_belief_coordinate()
    confidence = agent.belief.get_confidence_radius(inferred_coord, radius=1.5)
    inferred_status, status_conf = agent.belief.get_inferred_status()
    ground_truth = env.get_ground_truth_for_evaluation()

    json_report, md_report = reporter.generate_report(
        mission_name=mission_name,
        total_steps=step,
        initial_battery=initial_battery,
        final_battery=env.battery,
        inferred_pos=inferred_coord,
        confidence_pct=confidence,
        top_hypotheses=agent.belief.get_top_hypotheses(3),
        inferred_status=inferred_status,
        status_confidence=status_conf,
        initial_entropy=initial_entropy,
        final_entropy=agent.belief.get_shannon_entropy(),
        clues_collected=agent.analyzed_clues,
        ground_truth=ground_truth
    )

    print(f"📄 Official Mission Report generated at:\n  - Markdown: {md_report}\n  - JSON: {json_report}")
    print("\n🎯 Final Results Summary:")
    print(f"  • Inferred Location: {inferred_coord} (Confidence: {confidence}%)")
    print(f"  • True Location:     {ground_truth['eve_position']}")
    print(f"  • Inferred Status:   {inferred_status} ({status_conf}%)")
    print(f"  • Remaining Battery: {env.battery}%")
    print(f"  • Visual Discovery:  {'SUCCESS' if ground_truth['success'] else 'PROXIMITY ACHIEVED'}")
    print("=" * 80 + "\n")

    return json_trace, json_report


def export_bundled_traces():
    """Generates a static JS bundle of all traces so the UI works offline or via file://"""
    traces_dir = PROJECT_ROOT / "outputs" / "traces"
    out_file = PROJECT_ROOT / "src" / "visualization" / "web" / "mission_traces.js"
    
    slug_files = {
        "wasteland": "trace_mission_sector_7___wasteland_discovery.json",
        "dust_storm": "trace_mission_sol_404___heavy_dust_storm.json",
        "decoy": "trace_mission_sector_9___decoy_interference_challenge.json"
    }
    
    traces_obj = {}
    for key, filename in slug_files.items():
        fp = traces_dir / filename
        if fp.exists():
            try:
                import json
                traces_obj[key] = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass

    if traces_obj:
        import json
        js_code = f"window.BUILTIN_TRACES = {json.dumps(traces_obj, indent=2)};\n"
        out_file.write_text(js_code, encoding="utf-8")


def launch_web_server(port: int = 8080):
    """Launches local web dashboard server with automatic port hunting."""
    os.chdir(PROJECT_ROOT)
    export_bundled_traces()
    
    class CustomHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            # Disable cache for local dev
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()

        def translate_path(self, path):
            # Route frontend web assets
            clean_path = path.split("?")[0]
            if clean_path in ["/", "/index.html"]:
                return str(PROJECT_ROOT / "src" / "visualization" / "web" / "index.html")
            elif clean_path.startswith("/style.css"):
                return str(PROJECT_ROOT / "src" / "visualization" / "web" / "style.css")
            elif clean_path.startswith("/app.js"):
                return str(PROJECT_ROOT / "src" / "visualization" / "web" / "app.js")
            elif clean_path.startswith("/mission_traces.js"):
                return str(PROJECT_ROOT / "src" / "visualization" / "web" / "mission_traces.js")
            return super().translate_path(path)

    # Hunt for an available port starting from requested port
    current_port = port
    server = None
    for p in range(port, port + 25):
        try:
            server = HTTPServer(('127.0.0.1', p), CustomHandler)
            current_port = p
            break
        except OSError:
            continue

    if server is None:
        print("❌ Could not bind server to any available port between 8080 and 8105.")
        return

    url = f"http://127.0.0.1:{current_port}"
    print(f"\n🌐 Starting WALL-E Mission Web Dashboard at: {url}")
    print(f"   Direct file alternative: file:///{str(PROJECT_ROOT / 'src' / 'visualization' / 'web' / 'index.html').replace(os.sep, '/')}")
    print("   Press Ctrl+C to stop the server.\n")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web dashboard server.")


def main():
    parser = argparse.ArgumentParser(description="WALL-E & EVE's Signal: Autonomous AI Agent Simulation")
    parser.add_argument("--config", type=str, default="configs/mission_wasteland.json",
                        help="Path to mission configuration file (JSON or YAML)")
    parser.add_argument("--live", action="store_true",
                        help="Render live step-by-step ASCII terminal animation")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="Delay in seconds between terminal animation steps")
    parser.add_argument("--web", action="store_true",
                        help="Launch interactive web dashboard")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for the web dashboard server")
    parser.add_argument("--all-configs", action="store_true",
                        help="Run all sample mission configurations sequentially")

    args = parser.parse_args()

    if args.web:
        launch_web_server(args.port)
        return

    if args.all_configs:
        config_files = [
            "configs/mission_wasteland.json",
            "configs/mission_dust_storm.yaml",
            "configs/mission_decoy_challenge.json"
        ]
        for cfg in config_files:
            run_simulation(cfg, live_terminal=args.live, delay=args.delay)
    else:
        run_simulation(args.config, live_terminal=args.live, delay=args.delay)


if __name__ == "__main__":
    main()
