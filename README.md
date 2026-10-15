# Signal Navigator: Autonomous POMDP Agent & Bayesian Inference

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Interface](https://img.shields.io/badge/UI-NASA%20JPL%20Telemetry-orange.svg)](#interactive-mission-control-web-ui)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](tests/)

> An engineering-grade autonomous search simulation where an exploration rover (**WALL-E**) locates an unknown target beacon (**EVE**) across hostile wasteland terrain using **Partially Observable Markov Decision Processes (POMDP)**, **Recursive Bayesian Belief Filtering**, and **Resource-Constrained Information-Theoretic Planning**.

---

## 🛰️ Problem Formulation

In real-world planetary exploration and search-and-rescue robotics, an agent rarely has full state visibility. Sensors are noisy, terrain features block signals, battery power is strictly bounded, and target coordinates are unknown.

**Signal Navigator** models this challenge mathematically as a **POMDP** defined by the 7-tuple $\langle S, A, T, R, \Omega, O, \gamma \rangle$:

1. **State Space ($S$)**:
   $$\mathbf{s} = (p_{\text{agent}}, p_{\text{target}}, e_{\text{battery}}, \mathcal{C}_{\text{collected}})$$
   The true target position $p_{\text{target}} \in \mathbb{R}^2$ is hidden from the agent.
2. **Action Space ($A$)**:
   * Spatial Navigation: $\text{MOVE}(\text{NORTH}, \text{SOUTH}, \text{EAST}, \text{WEST})$
   * Active Sensing: $\text{LONG\_RANGE\_PING}$ (directional bearing), $\text{LOCAL\_SCAN}$ (high-confidence radius)
   * Artifact Analysis: $\text{ANALYZE\_CLUE}$ (collapses candidate hypothesis subspace)
   * Resource Management: $\text{RECHARGE}$ (at verified solar charging pads)
3. **Belief Representation ($b$)**:
   A discrete 2D probability distribution over all traversable grid coordinates:
   $$b(x, y) = P(p_{\text{target}} = (x, y) \mid o_{1:t}, a_{1:t}), \quad \sum_{x, y} b(x, y) = 1.0$$
4. **Bayesian Belief Update**:
   Upon executing action $a$ and receiving noisy observation $o \in \Omega$:
   $$b'(x, y) = \frac{O(o \mid (x, y), a) \cdot b(x, y)}{\sum_{x', y'} O(o \mid (x', y'), a) \cdot b(x', y')}$$
5. **Shannon Entropy (Uncertainty Metric)**:
   The spatial uncertainty of the belief state is tracked in bits:
   $$H(B) = -\sum_{(x, y)} b(x, y) \log_2 b(x, y)$$
   As the agent collects clues and pings signals, $H(B)$ monotonically collapses toward 0.00 bits.
6. **Utility Function & Action Selection**:
   Actions balance expected information gain against energy expenditure and safety:
   $$\mathcal{U}(a) = w_{\text{info}} \cdot \mathbb{E}[\Delta H(B) \mid a] - w_{\text{cost}} \cdot \text{Cost}(a) - w_{\text{risk}} \cdot \text{RiskToChargingPad}(p_{\text{agent}}')$$
   If battery level drops below the **20% fail-safe threshold**, the agent overrides exploration and initiates an emergency retreat to the nearest solar station.

---

## 🖥️ Interactive Mission Control Web UI

The user interface is designed as a **scientific mission-control telemetry instrument** inspired by NASA JPL workstations, Linear, Observable, and Distill.pub.

* **Design System**: Strict dark palette (`#0E0F11` bg, `#15171A` surface, `#23262B` borders) with a single Mars orange accent (`#E8743B`). No distracting neon glows or generic dashboard boxes.
* **Typography**: `Inter` for clean UI controls; `JetBrains Mono` with `tabular-nums` for coordinates, percentages, and telemetry counters.
* **Hero Grid (Left 60%)**:
  * Real-time single-hue Bayesian belief heatmap.
  * Muted slate obstacle cells with subtle diagonal hatching.
  * Autonomous rover marker with heading vector and faint sensor ring.
  * Polyline breadcrumb trajectory fading with age.
  * Interactive hover tooltips showing coordinates, terrain type, and $P(\text{target})$.
* **Telemetry Diagnostics (Right 40%)**:
  * **Agent State**: Clean key/value telemetry list with sentence-case status and action rationale notes.
  * **Entropy Sparkline**: Live SVG line chart of Shannon entropy $H(B)$ vs. step with active step marker.
  * **Battery Monitor**: Gauge with highlighted 20% safe-return threshold tick.
  * **Posterior $P(\text{target})$**: Horizontal candidate probability bars.
  * **Event Log**: Compact, expandable single-line event rows.
* **Keyboard Accessibility**: `Space` (Run/Pause), `ArrowLeft` / `ArrowRight` (Step Prev/Next), `R` (Reset).

---

## 📂 Repository Structure

```
├── configs/                      # Mission configurations (terrain, hazards, obstacles)
│   ├── mission_wasteland.json    # Baseline Sector 7 wasteland
│   ├── mission_dust_storm.yaml   # Sol 404 heavy sensor noise scenario
│   └── mission_decoy_challenge.json # Sector 9 multi-hypothesis decoy scenario
├── outputs/                      # Simulation traces & evaluation reports
│   ├── reports/                  # Markdown & JSON performance metrics
│   └── traces/                   # Serialized step-by-step state & belief histories
├── src/
│   ├── core/
│   │   ├── agent.py              # POMDP Agent state machine & policy
│   │   ├── belief.py             # Recursive Bayesian belief distribution & entropy
│   │   ├── environment.py        # Grid world terrain, obstacles, solar pads
│   │   └── planner.py            # A* pathfinding & information gain optimizer
│   ├── utils/
│   │   ├── logger.py             # Structured event logging
│   │   └── reporter.py           # Mission evaluation reporting
│   └── visualization/
│       ├── terminal_view.py      # Rich terminal ASCII grid renderer
│       └── web/                  # Web Mission Control Interface
│           ├── index.html        # Clean semantic HTML5 layout
│           ├── style.css         # Design tokens & responsive styles
│           ├── app.js            # Canvas rendering & telemetry bindings
│           └── mission_traces.js # Standalone bundled datasets (offline-ready)
├── tests/
│   └── test_mission.py           # Unit tests for belief updates and simulation logic
├── requirements.txt              # Minimal external dependencies
├── run_mission.py                # CLI runner & local visualization server
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.9 or higher

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/anuj1704bohra-ops/walle-eve-signal-navigator.git
cd walle-eve-signal-navigator
pip install -r requirements.txt
```

### 3. Run the Simulation

**Run a headless mission in the terminal:**
```bash
python run_mission.py --config configs/mission_wasteland.json
```

**Launch the interactive web visualization:**
```bash
python run_mission.py --web --port 8080
```
Open your browser at:
```
http://127.0.0.1:8080/
```

*Note: You can also double-click `src/visualization/web/index.html` directly in File Explorer to open it via `file:///` — all mission datasets are pre-bundled for offline demonstration.*

### 4. Run Unit Tests
```bash
python -m unittest tests/test_mission.py
```

---


## 📜 License

MIT License — feel free to use and extend for academic, personal, or competitive robotics projects.
