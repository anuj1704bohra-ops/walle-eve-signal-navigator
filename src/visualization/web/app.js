/**
 * Signal Navigator - Scientific Mission Control System
 * Autonomous Search & Bayesian Inference Simulation Visualizer
 * Reference: NASA JPL telemetry tools, Linear, Observable, Distill.pub
 */

// Global Application State
let missionData = null;
let currentStepIndex = 0;
let isPlaying = false;
let playInterval = null;
let playbackSpeed = 1.0;
let showHeatmap = true;
let showTrail = true;
let showObserverMode = false;
let autoScrollLog = true;

// DOM Elements
const canvas = document.getElementById("missionCanvas");
const ctx = canvas.getContext("2d");
const cellTooltip = document.getElementById("cellTooltip");

const missionSelect = document.getElementById("missionSelect");
const missionStatusText = document.getElementById("missionStatusText");
const statusDot = document.getElementById("statusDot");
const gridDimensionsBadge = document.getElementById("gridDimensionsBadge");

const btnPlayPause = document.getElementById("btnPlayPause");
const playBtnLabel = document.getElementById("playBtnLabel");
const playIcon = document.getElementById("playIcon");
const btnPrevStep = document.getElementById("btnPrevStep");
const btnNextStep = document.getElementById("btnNextStep");
const btnReset = document.getElementById("btnReset");

const stepProgress = document.getElementById("stepProgress");
const currentStepDisplay = document.getElementById("currentStepDisplay");
const maxStepDisplay = document.getElementById("maxStepDisplay");
const speedPills = document.querySelectorAll(".speed-pill");

const toggleHeatmap = document.getElementById("toggleHeatmap");
const toggleTrail = document.getElementById("toggleTrail");
const toggleObserverMode = document.getElementById("toggleObserverMode");

// Telemetry Elements
const agentModeBadge = document.getElementById("agentModeBadge");
const lastActionText = document.getElementById("lastActionText");
const currentCoordinatesText = document.getElementById("currentCoordinatesText");
const cluesCountText = document.getElementById("cluesCountText");
const actionRationaleText = document.getElementById("actionRationaleText");

// Entropy Chart Elements
const entropyValueText = document.getElementById("entropyValueText");
const entropyDeltaTag = document.getElementById("entropyDeltaTag");
const entropyAreaPath = document.getElementById("entropyAreaPath");
const entropyLinePath = document.getElementById("entropyLinePath");
const entropyCurrentPoint = document.getElementById("entropyCurrentPoint");

// Battery Elements
const batteryPercentText = document.getElementById("batteryPercentText");
const batteryCostLabel = document.getElementById("batteryCostLabel");
const batteryProgressBar = document.getElementById("batteryProgressBar");
const batteryHealthStatus = document.getElementById("batteryHealthStatus");

// Posterior Hypotheses Elements
const distributionConfidenceLabel = document.getElementById("distributionConfidenceLabel");
const topHypothesesList = document.getElementById("topHypothesesList");

// Event Log Elements
const reasoningLogContainer = document.getElementById("reasoningLogContainer");
const eventCountBadge = document.getElementById("eventCountBadge");
const btnAutoScroll = document.getElementById("btnAutoScroll");
const btnClearLog = document.getElementById("btnClearLog");

// Modal Elements
const btnDownloadReport = document.getElementById("btnDownloadReport");
const btnDownloadTrace = document.getElementById("btnDownloadTrace");
const reportModal = document.getElementById("reportModal");
const btnCloseModal = document.getElementById("btnCloseModal");
const btnModalCloseSecondary = document.getElementById("btnModalCloseSecondary");
const modalReportBody = document.getElementById("modalReportBody");

// Canvas Layout Geometry
const AXIS_MARGIN_TOP = 20;
const AXIS_MARGIN_LEFT = 22;
const AXIS_MARGIN_BOTTOM = 8;
const AXIS_MARGIN_RIGHT = 8;

// Initialize on Load
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadMission(missionSelect.value);
});

function setupEventListeners() {
  missionSelect.addEventListener("change", (e) => loadMission(e.target.value));

  btnPlayPause.addEventListener("click", togglePlay);
  btnPrevStep.addEventListener("click", () => setStep(currentStepIndex - 1));
  btnNextStep.addEventListener("click", () => setStep(currentStepIndex + 1));
  btnReset.addEventListener("click", () => setStep(0));

  stepProgress.addEventListener("input", (e) => {
    setStep(parseInt(e.target.value, 10));
  });

  speedPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      speedPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      playbackSpeed = parseFloat(pill.dataset.speed);
      if (isPlaying) {
        clearInterval(playInterval);
        startPlayback();
      }
    });
  });

  toggleHeatmap.addEventListener("change", (e) => {
    showHeatmap = e.target.checked;
    render();
  });

  toggleTrail.addEventListener("change", (e) => {
    showTrail = e.target.checked;
    render();
  });

  toggleObserverMode.addEventListener("change", (e) => {
    showObserverMode = e.target.checked;
    render();
  });

  btnAutoScroll.addEventListener("click", () => {
    autoScrollLog = !autoScrollLog;
    btnAutoScroll.classList.toggle("active", autoScrollLog);
  });

  btnClearLog.addEventListener("click", () => {
    reasoningLogContainer.innerHTML = "";
    eventCountBadge.textContent = "0";
  });

  btnDownloadReport.addEventListener("click", openReportModal);
  btnCloseModal.addEventListener("click", () => reportModal.classList.add("hidden"));
  btnModalCloseSecondary.addEventListener("click", () => reportModal.classList.add("hidden"));
  btnDownloadTrace.addEventListener("click", downloadTraceJson);

  // Canvas Hover Tooltip
  canvas.addEventListener("mousemove", handleCanvasMouseMove);
  canvas.addEventListener("mouseleave", () => cellTooltip.classList.add("hidden"));

  // Keyboard Shortcuts (Space, Left/Right arrow, R)
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      setStep(currentStepIndex - 1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      setStep(currentStepIndex + 1);
    } else if (e.key === "r" || e.key === "R") {
      setStep(0);
    }
  });
}

/**
 * Loads mission trace data from bundled offline datasets or local HTTP server
 */
async function loadMission(missionKey) {
  stopPlayback();

  // Check bundled offline traces first (works via file:/// and offline)
  if (window.BUILTIN_TRACES && window.BUILTIN_TRACES[missionKey]) {
    missionData = window.BUILTIN_TRACES[missionKey];
  } else {
    const slugMap = {
      wasteland: "trace_mission_sector_7___wasteland_discovery.json",
      dust_storm: "trace_mission_sol_404___heavy_dust_storm.json",
      decoy: "trace_mission_sector_9___decoy_interference_challenge.json"
    };

    const filename = slugMap[missionKey] || "trace_mission_sector_7___wasteland_discovery.json";

    try {
      const res = await fetch(`/outputs/traces/${filename}`);
      if (res.ok) {
        missionData = await res.json();
      } else {
        throw new Error(`Server returned HTTP ${res.status}`);
      }
    } catch (err) {
      console.warn("Notice: Fetching trace over network failed, loading procedural dataset.", err);
      missionData = generateDemonstrationTrace(missionKey);
    }
  }

  // Update Grid dimensions badge
  const [gw, gh] = missionData.grid_dimensions || [16, 16];
  gridDimensionsBadge.textContent = `${gw} × ${gh}`;

  // Reset controls
  currentStepIndex = 0;
  const totalSteps = missionData.trace_history ? missionData.trace_history.length : 0;
  stepProgress.max = totalSteps > 0 ? totalSteps - 1 : 0;
  maxStepDisplay.textContent = totalSteps > 0 ? totalSteps - 1 : 0;
  
  reasoningLogContainer.innerHTML = "";
  eventCountBadge.textContent = "0";

  setStep(0);
}

function setStep(index) {
  if (!missionData || !missionData.trace_history || missionData.trace_history.length === 0) return;
  const history = missionData.trace_history;
  if (index < 0) index = 0;
  if (index >= history.length) {
    index = history.length - 1;
    stopPlayback();
  }

  currentStepIndex = index;
  stepProgress.value = index;
  currentStepDisplay.textContent = index;

  updateTelemetry(history[index]);
  render();
}

function togglePlay() {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (!missionData || !missionData.trace_history) return;
  if (currentStepIndex >= missionData.trace_history.length - 1) {
    currentStepIndex = 0;
  }
  isPlaying = true;
  playBtnLabel.textContent = "Pause";
  playIcon.innerHTML = '<path d="M4 2h3v12H4zm5 0h3v12H9z"/>';
  btnPlayPause.classList.remove("btn-primary");

  const intervalMs = Math.max(70, 480 / playbackSpeed);
  playInterval = setInterval(() => {
    if (currentStepIndex < missionData.trace_history.length - 1) {
      setStep(currentStepIndex + 1);
    } else {
      stopPlayback();
    }
  }, intervalMs);
}

function stopPlayback() {
  isPlaying = false;
  if (playInterval) clearInterval(playInterval);
  playBtnLabel.textContent = "Run";
  playIcon.innerHTML = '<path d="M4 2v12l10-6z"/>';
  btnPlayPause.classList.add("btn-primary");
}

/**
 * Updates telemetry metrics, entropy chart, battery bar, agent state, and event log
 */
function updateTelemetry(stepData) {
  if (!stepData) return;

  // 1. Agent State: Clean Key/Value List
  const rawMode = stepData.agent_mode || "RUNNING";
  const modeMap = {
    "INITIALIZING": "Initialising",
    "TRACKING_EVE": "Tracking target",
    "SCANNING_BEACON": "Scanning beacon",
    "ANALYZING_CLUE": "Analysing clue",
    "APPROACHING_CLUE": "Approaching clue",
    "RECHARGING": "Recharging",
    "EMERGENCY_RECHARGE": "Critical retreat",
    "LOCAL_RADAR_SWEEP": "Local radar sweep",
    "EXPLORATION_SWEEP": "Exploration sweep"
  };
  agentModeBadge.textContent = modeMap[rawMode] || rawMode.toLowerCase();

  let actionDisplay = stepData.action;
  if (stepData.action_params?.direction) {
    actionDisplay += ` [${stepData.action_params.direction}]`;
  }
  lastActionText.textContent = actionDisplay;

  const [wx, wy] = stepData.agent_position;
  currentCoordinatesText.textContent = `(${wx}, ${wy})`;

  // Clues count
  const historySlice = missionData.trace_history.slice(0, currentStepIndex + 1);
  const totalClues = missionData.clues?.length || 3;
  const uniqueClues = new Set(
    historySlice
      .filter((t) => t.observation_summary?.clue_collected)
      .map((t) => t.observation_summary.clue_collected)
  ).size;
  cluesCountText.textContent = `${uniqueClues} / ${totalClues}`;

  // "Why this action" Rationale note
  actionRationaleText.textContent = stepData.reasoning || "Evaluating posterior distribution.";

  // 2. Shannon Entropy Telemetry & Line Chart
  const entropy = stepData.belief_entropy_bits;
  entropyValueText.textContent = entropy.toFixed(2);

  if (currentStepIndex > 0) {
    const prevEntropy = missionData.trace_history[currentStepIndex - 1]?.belief_entropy_bits || entropy;
    const delta = entropy - prevEntropy;
    if (Math.abs(delta) < 0.001) {
      entropyDeltaTag.textContent = `Δ 0.00`;
      entropyDeltaTag.style.color = "var(--text-faint)";
    } else if (delta < 0) {
      entropyDeltaTag.textContent = `Δ ${delta.toFixed(2)}`;
      entropyDeltaTag.style.color = "var(--text-muted)";
    } else {
      entropyDeltaTag.textContent = `Δ +${delta.toFixed(2)}`;
      entropyDeltaTag.style.color = "var(--warning)";
    }
  } else {
    entropyDeltaTag.textContent = `baseline`;
    entropyDeltaTag.style.color = "var(--text-faint)";
  }

  // Draw Entropy Line Chart
  renderEntropyChart(missionData.trace_history, currentStepIndex);

  // 3. Battery Management
  const bat = stepData.battery_level;
  batteryPercentText.textContent = `${bat}%`;
  batteryProgressBar.style.width = `${bat}%`;

  const cost = stepData.battery_cost;
  if (stepData.action === "RECHARGE") {
    batteryCostLabel.textContent = `Gain: +25%`;
  } else if (cost > 0) {
    batteryCostLabel.textContent = `Cost: -${cost}%`;
  } else {
    batteryCostLabel.textContent = `Cost: 0%`;
  }

  // Battery health status
  if (bat <= 20) {
    batteryProgressBar.className = "battery-fill battery-low";
    batteryHealthStatus.textContent = "Critical reserve (<20%)";
    batteryHealthStatus.style.color = "var(--danger)";
  } else if (bat <= 35) {
    batteryProgressBar.className = "battery-fill battery-warn";
    batteryHealthStatus.textContent = "Reserve low";
    batteryHealthStatus.style.color = "var(--warning)";
  } else {
    batteryProgressBar.className = "battery-fill";
    batteryHealthStatus.textContent = "Adequate reserve";
    batteryHealthStatus.style.color = "var(--text-faint)";
  }

  // 4. Posterior P(target): Horizontal Probability Bars
  renderPosteriorHypotheses(stepData.top_hypotheses);

  // 5. Event Log: Compact Single-Line Rows (Click to Expand)
  appendLedgerEntry(stepData);

  // Status Indicator
  const isFinalStep = currentStepIndex === missionData.trace_history.length - 1;
  if (isFinalStep) {
    statusDot.className = "status-dot status-complete";
    missionStatusText.textContent = "Target located";
  } else if (rawMode.includes("EMERGENCY")) {
    statusDot.className = "status-dot status-emergency";
    missionStatusText.textContent = "Critical retreat";
  } else if (rawMode.includes("RECHARGE")) {
    statusDot.className = "status-dot";
    missionStatusText.textContent = "Recharging";
  } else {
    statusDot.className = "status-dot";
    missionStatusText.textContent = "Agent active";
  }
}

/**
 * Draws a lightweight SVG line chart of Shannon entropy vs step
 */
function renderEntropyChart(history, activeIndex) {
  if (!history || history.length === 0) return;

  const width = 280;
  const height = 64;
  const padLeft = 4;
  const padRight = 6;
  const padTop = 6;
  const padBottom = 8;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  const maxSteps = Math.max(1, history.length - 1);
  const maxEntropy = 8.0;

  // Build points
  const points = history.map((entry, idx) => {
    const x = padLeft + (idx / maxSteps) * chartW;
    const yVal = Math.max(0, Math.min(maxEntropy, entry.belief_entropy_bits));
    const y = padTop + (1 - (yVal / maxEntropy)) * chartH;
    return [x, y];
  });

  // Construct SVG paths
  let lineD = `M ${points[0][0].toFixed(1)} ${points[0][1].toFixed(1)}`;
  for (let i = 1; i < points.length; i++) {
    lineD += ` L ${points[i][0].toFixed(1)} ${points[i][1].toFixed(1)}`;
  }

  const baselineY = (height - padBottom).toFixed(1);
  const areaD = `${lineD} L ${points[points.length - 1][0].toFixed(1)} ${baselineY} L ${points[0][0].toFixed(1)} ${baselineY} Z`;

  entropyLinePath.setAttribute("d", lineD);
  entropyAreaPath.setAttribute("d", areaD);

  // Current step marker point
  if (points[activeIndex]) {
    entropyCurrentPoint.setAttribute("cx", points[activeIndex][0].toFixed(1));
    entropyCurrentPoint.setAttribute("cy", points[activeIndex][1].toFixed(1));
  }
}

/**
 * Renders Posterior P(target) candidate rows with single Mars orange accent
 */
function renderPosteriorHypotheses(topHypotheses) {
  topHypothesesList.innerHTML = "";
  if (!topHypotheses || topHypotheses.length === 0) {
    topHypothesesList.innerHTML = '<span class="text-faint" style="font-size: 11px;">Prior uniform across sectors.</span>';
    distributionConfidenceLabel.textContent = "Uniform";
    return;
  }

  const total = topHypotheses.length;
  distributionConfidenceLabel.textContent = `${total} ${total === 1 ? 'hypothesis' : 'hypotheses'}`;

  topHypotheses.slice(0, 4).forEach((h) => {
    const coord = h.coordinate;
    const pct = (h.probability * 100).toFixed(1);

    const row = document.createElement("div");
    row.className = "posterior-row";
    row.innerHTML = `
      <span class="posterior-coord mono-val">(${coord[0]}, ${coord[1]})</span>
      <div class="posterior-bar-track" title="Sector (${coord[0]}, ${coord[1]}): ${pct}%">
        <div class="posterior-bar-fill" style="width: ${Math.min(100, Math.max(2, pct))}%"></div>
      </div>
      <span class="posterior-val mono-val">${pct}%</span>
    `;
    topHypothesesList.appendChild(row);
  });
}

/**
 * Appends compact single-line event rows (click to expand)
 */
function appendLedgerEntry(stepData) {
  const existing = document.getElementById(`ledger-step-${stepData.step}`);
  if (existing) {
    if (autoScrollLog) existing.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }

  const item = document.createElement("div");
  item.id = `ledger-step-${stepData.step}`;
  item.className = "event-row";

  // Compact action tag
  let tag = stepData.action;
  if (tag === "LONG_RANGE_PING") tag = "PING";
  else if (tag === "LOCAL_SCAN") tag = "SCAN";
  else if (tag === "ANALYZE_CLUE") tag = "CLUE";
  else if (tag === "RECHARGE") tag = "CHG";
  else if (stepData.agent_mode.includes("EMERGENCY")) tag = "RETREAT";

  // Clean short summary
  let msg = stepData.reasoning;
  if (msg.length > 58) {
    msg = msg.substring(0, 56) + "...";
  }

  item.innerHTML = `
    <div class="event-row-summary">
      <span class="event-step-col mono-val">${String(stepData.step).padStart(3, '0')}</span>
      <span class="event-tag-col mono-val">${tag}</span>
      <span class="event-msg-col">${msg}</span>
    </div>
    <div class="event-row-detail">
      ${stepData.reasoning}
    </div>
  `;

  // Click to expand / collapse full explanation
  item.addEventListener("click", () => {
    item.classList.toggle("expanded");
  });

  reasoningLogContainer.appendChild(item);
  eventCountBadge.textContent = `${reasoningLogContainer.children.length}`;

  if (autoScrollLog) {
    reasoningLogContainer.scrollTop = reasoningLogContainer.scrollHeight;
  }
}

/**
 * Crisp Scientific Canvas Grid Renderer
 * Single-hue Mars orange heatmap, muted slate obstacles, fading polyline trail
 */
function render() {
  if (!missionData) return;

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const [gridW, gridH] = missionData.grid_dimensions || [16, 16];
  
  const viewW = width - AXIS_MARGIN_LEFT - AXIS_MARGIN_RIGHT;
  const viewH = height - AXIS_MARGIN_TOP - AXIS_MARGIN_BOTTOM;
  const cellW = viewW / gridW;
  const cellH = viewH / gridH;

  const currentStep = missionData.trace_history ? missionData.trace_history[currentStepIndex] : null;
  const historySlice = missionData.trace_history ? missionData.trace_history.slice(0, currentStepIndex + 1) : [];

  // 1. Draw Small Muted Coordinate Labels Along Axes
  ctx.fillStyle = "#555962";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let x = 0; x < gridW; x++) {
    const cx = AXIS_MARGIN_LEFT + (x + 0.5) * cellW;
    ctx.fillText(`${x}`, cx, AXIS_MARGIN_TOP / 2);
  }

  ctx.textAlign = "right";
  for (let y = 0; y < gridH; y++) {
    const cy = AXIS_MARGIN_TOP + (y + 0.5) * cellH;
    ctx.fillText(`${y}`, AXIS_MARGIN_LEFT - 5, cy);
  }

  // 2. Draw Subtle Grid Cells (Faint ~6% opacity lines)
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const rx = AXIS_MARGIN_LEFT + x * cellW;
      const ry = AXIS_MARGIN_TOP + y * cellH;

      ctx.fillStyle = "#101114";
      ctx.fillRect(rx, ry, cellW, cellH);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      ctx.strokeRect(rx, ry, cellW, cellH);
    }
  }

  // 3. Draw Hazards / Dust Zones (Lower-contrast warm tint)
  const hazards = missionData.hazards || [];
  hazards.forEach(([hx, hy]) => {
    const rx = AXIS_MARGIN_LEFT + hx * cellW;
    const ry = AXIS_MARGIN_TOP + hy * cellH;

    ctx.fillStyle = "rgba(224, 138, 60, 0.09)";
    ctx.fillRect(rx + 1, ry + 1, cellW - 2, cellH - 2);

    ctx.strokeStyle = "rgba(224, 138, 60, 0.22)";
    ctx.lineWidth = 1;
    ctx.strokeRect(rx + 1, ry + 1, cellW - 2, cellH - 2);
  });

  // 4. Draw Obstacles (Solid Muted Slate with Subtle Diagonal Hatch, No X marks)
  const obstacles = missionData.obstacles || [];
  obstacles.forEach(([ox, oy]) => {
    const rx = AXIS_MARGIN_LEFT + ox * cellW;
    const ry = AXIS_MARGIN_TOP + oy * cellH;

    ctx.fillStyle = "#252930";
    ctx.fillRect(rx + 1, ry + 1, cellW - 2, cellH - 2);

    ctx.strokeStyle = "#2F343E";
    ctx.lineWidth = 1;
    ctx.strokeRect(rx + 1, ry + 1, cellW - 2, cellH - 2);

    // Subtle diagonal hatch line (no X marks)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.beginPath();
    ctx.moveTo(rx + 2, ry + cellH - 2);
    ctx.lineTo(rx + cellW - 2, ry + 2);
    ctx.stroke();
  });

  // 5. Draw Charging Stations (Small Clean Bolt Icon, No Bright Green Highlight)
  const solarStations = missionData.solar_recharge_stations || [];
  solarStations.forEach(([sx, sy]) => {
    const rx = AXIS_MARGIN_LEFT + sx * cellW;
    const ry = AXIS_MARGIN_TOP + sy * cellH;

    ctx.fillStyle = "#16181C";
    ctx.fillRect(rx + 2, ry + 2, cellW - 4, cellH - 4);
    ctx.strokeStyle = "#2E333C";
    ctx.lineWidth = 1;
    ctx.strokeRect(rx + 2, ry + 2, cellW - 4, cellH - 4);

    // Minimal bolt icon in muted white
    ctx.fillStyle = "#C8CBD0";
    ctx.beginPath();
    const cx = rx + cellW / 2;
    const cy = ry + cellH / 2;
    ctx.moveTo(cx + 0.5, cy - 6);
    ctx.lineTo(cx - 4, cy);
    ctx.lineTo(cx - 0.5, cy);
    ctx.lineTo(cx - 1, cy + 6);
    ctx.lineTo(cx + 4, cy - 1);
    ctx.lineTo(cx + 0.5, cy - 1);
    ctx.closePath();
    ctx.fill();
  });

  // 6. Draw Clue Beacons (Subtle Diamond Marker)
  const clues = missionData.clues || [];
  clues.forEach((c) => {
    const [cx, cy] = c.position;
    const rx = AXIS_MARGIN_LEFT + cx * cellW;
    const ry = AXIS_MARGIN_TOP + cy * cellH;
    const isCollected = historySlice.some(
      (t) => t.agent_position[0] === cx && t.agent_position[1] === cy
    );

    const midX = rx + cellW / 2;
    const midY = ry + cellH / 2;

    ctx.fillStyle = isCollected ? "rgba(232, 116, 59, 0.12)" : "rgba(232, 116, 59, 0.3)";
    ctx.strokeStyle = "#E8743B";
    ctx.lineWidth = 1.25;

    ctx.beginPath();
    ctx.moveTo(midX, midY - 5);
    ctx.lineTo(midX + 5, midY);
    ctx.lineTo(midX, midY + 5);
    ctx.lineTo(midX - 5, midY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  });

  // 7. Draw Belief Heatmap: Single-Hue Ramp from Transparent to Mars Orange (#E8743B)
  if (showHeatmap && currentStep) {
    if (currentStep.belief_heatmap) {
      const matrix = currentStep.belief_heatmap;
      const flat = matrix.flat();
      const maxP = Math.max(...flat, 0.0001);

      for (let y = 0; y < gridH; y++) {
        for (let x = 0; x < gridW; x++) {
          const p = matrix[y] ? matrix[y][x] : 0;
          if (p > 0.005) {
            const rx = AXIS_MARGIN_LEFT + x * cellW;
            const ry = AXIS_MARGIN_TOP + y * cellH;
            const normP = Math.min(1.0, p / maxP);
            const alpha = Math.min(0.85, 0.08 + normP * 0.72);

            ctx.fillStyle = `rgba(232, 116, 59, ${alpha})`;
            ctx.fillRect(rx + 1, ry + 1, cellW - 2, cellH - 2);
          }
        }
      }
    } else if (currentStep.top_hypotheses) {
      currentStep.top_hypotheses.forEach((h) => {
        const [tx, ty] = h.coordinate;
        const rx = AXIS_MARGIN_LEFT + tx * cellW;
        const ry = AXIS_MARGIN_TOP + ty * cellH;
        const alpha = Math.min(0.75, Math.max(0.12, h.probability * 1.2));

        ctx.fillStyle = `rgba(232, 116, 59, ${alpha})`;
        ctx.fillRect(rx + 1, ry + 1, cellW - 2, cellH - 2);
      });
    }

    // Target Hypothesis Highlight: Thin Orange Outline Only (No Magenta)
    const topHyp = currentStep.top_hypotheses ? currentStep.top_hypotheses[0] : null;
    if (topHyp) {
      const [tx, ty] = topHyp.coordinate;
      const rx = AXIS_MARGIN_LEFT + tx * cellW;
      const ry = AXIS_MARGIN_TOP + ty * cellH;

      ctx.strokeStyle = "#E8743B";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rx + 2, ry + 2, cellW - 4, cellH - 4);
    }
  }

  // 8. Ground Truth Observer Mode (Subtle Muted Outline)
  if (showObserverMode && missionData.eve_target_ground_truth) {
    const [ex, ey] = missionData.eve_target_ground_truth;
    const rx = AXIS_MARGIN_LEFT + ex * cellW;
    const ry = AXIS_MARGIN_TOP + ey * cellH;

    ctx.strokeStyle = "#8B8F96";
    ctx.lineWidth = 1.25;
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(rx + 2, ry + 2, cellW - 4, cellH - 4);
    ctx.setLineDash([]);
  }

  // 9. Path Trail: Polyline that Fades with Age (No Dots on Every Cell)
  if (showTrail && historySlice.length > 1) {
    const totalSegs = historySlice.length - 1;
    for (let i = 0; i < totalSegs; i++) {
      const p1 = historySlice[i].agent_position;
      const p2 = historySlice[i + 1].agent_position;

      const x1 = AXIS_MARGIN_LEFT + (p1[0] + 0.5) * cellW;
      const y1 = AXIS_MARGIN_TOP + (p1[1] + 0.5) * cellH;
      const x2 = AXIS_MARGIN_LEFT + (p2[0] + 0.5) * cellW;
      const y2 = AXIS_MARGIN_TOP + (p2[1] + 0.5) * cellH;

      const ageRatio = (i + 1) / (totalSegs + 1);
      const alpha = 0.08 + ageRatio * 0.45;

      ctx.strokeStyle = `rgba(232, 116, 59, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  // 10. Agent Rover: Small Rover Marker with Heading Indicator & Faint Sensor Ring
  if (currentStep) {
    const [wx, wy] = currentStep.agent_position;
    const rx = AXIS_MARGIN_LEFT + wx * cellW;
    const ry = AXIS_MARGIN_TOP + wy * cellH;
    const cx = rx + cellW / 2;
    const cy = ry + cellH / 2;

    // Faint sensor range ring
    ctx.strokeStyle = "rgba(232, 116, 59, 0.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, cellW * 0.95, 0, Math.PI * 2);
    ctx.stroke();

    // Rover chassis
    ctx.fillStyle = "#E8743B";
    ctx.strokeStyle = "#0E0F11";
    ctx.lineWidth = 1;
    ctx.fillRect(cx - 5.5, cy - 4.5, 11, 9);
    ctx.strokeRect(cx - 5.5, cy - 4.5, 11, 9);

    // Tread lines
    ctx.fillStyle = "#15171A";
    ctx.fillRect(cx - 7.5, cy - 5, 2, 10);
    ctx.fillRect(cx + 5.5, cy - 5, 2, 10);

    // Heading indicator arrow/tick
    const dir = currentStep.action_params?.direction;
    if (dir) {
      const dirMap = {
        NORTH: [0, -1],
        SOUTH: [0, 1],
        EAST: [1, 0],
        WEST: [-1, 0]
      };
      const dvec = dirMap[dir];
      if (dvec) {
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + dvec[0] * 6, cy + dvec[1] * 6);
        ctx.stroke();
      }
    }
  }
}

/**
 * Interactive Cell Tooltip Handler
 */
function handleCanvasMouseMove(e) {
  if (!missionData) return;

  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const [gridW, gridH] = missionData.grid_dimensions || [16, 16];
  const viewW = canvas.width - AXIS_MARGIN_LEFT - AXIS_MARGIN_RIGHT;
  const viewH = canvas.height - AXIS_MARGIN_TOP - AXIS_MARGIN_BOTTOM;
  const cellW = viewW / gridW;
  const cellH = viewH / gridH;

  const gx = Math.floor((mouseX - AXIS_MARGIN_LEFT) / cellW);
  const gy = Math.floor((mouseY - AXIS_MARGIN_TOP) / cellH);

  if (gx >= 0 && gx < gridW && gy >= 0 && gy < gridH) {
    let terrain = "Traversable";
    if (missionData.obstacles?.some(([x, y]) => x === gx && y === gy)) terrain = "Obstacle";
    else if (missionData.hazards?.some(([x, y]) => x === gx && y === gy)) terrain = "Dust hazard";
    else if (missionData.solar_recharge_stations?.some(([x, y]) => x === gx && y === gy)) terrain = "Charging pad";
    else if (missionData.clues?.some((c) => c.position[0] === gx && c.position[1] === gy)) terrain = "Clue artifact";

    const currentStep = missionData.trace_history ? missionData.trace_history[currentStepIndex] : null;
    let probText = "0.00%";
    if (currentStep?.belief_heatmap && currentStep.belief_heatmap[gy]) {
      const p = currentStep.belief_heatmap[gy][gx];
      probText = `${(p * 100).toFixed(2)}%`;
    }

    cellTooltip.innerHTML = `
      <div><strong>(${gx}, ${gy})</strong> — ${terrain}</div>
      <div style="color: var(--accent);">P(target): ${probText}</div>
    `;
    cellTooltip.style.left = `${Math.min(canvas.width - 150, mouseX + 12)}px`;
    cellTooltip.style.top = `${Math.min(canvas.height - 50, mouseY + 12)}px`;
    cellTooltip.classList.remove("hidden");
  } else {
    cellTooltip.classList.add("hidden");
  }
}

/**
 * Modal Report Viewer
 */
function openReportModal() {
  if (!missionData) return;

  const totalSteps = missionData.trace_history ? missionData.trace_history.length : 0;
  const finalStep = missionData.trace_history ? missionData.trace_history[totalSteps - 1] : null;
  const finalBattery = finalStep ? finalStep.battery_level : 100;
  const topCandidate = finalStep?.top_hypotheses ? finalStep.top_hypotheses[0] : null;
  const targetCoord = topCandidate ? `(${topCandidate.coordinate[0]}, ${topCandidate.coordinate[1]})` : "(Unknown)";
  const conf = topCandidate ? topCandidate.confidence_percent || 100.0 : 100.0;

  modalReportBody.innerHTML = `
    <h3 style="font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 6px;">
      Mission: ${missionData.mission_name || "Autonomous search"}
    </h3>
    <p style="color: var(--text-muted); margin-bottom: 12px;">
      Autonomous POMDP search completed. Quantitative telemetry summary:
    </p>
    
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Status</td>
          <td><strong style="color: var(--text);">Target located</strong></td>
        </tr>
        <tr>
          <td>Inferred target coordinates</td>
          <td><span class="mono-val">${targetCoord}</span> (${conf}% confidence)</td>
        </tr>
        <tr>
          <td>True target coordinates</td>
          <td><span class="mono-val">(${missionData.eve_target_ground_truth ? missionData.eve_target_ground_truth.join(', ') : '13, 13'})</span></td>
        </tr>
        <tr>
          <td>Final battery</td>
          <td><span class="mono-val">${finalBattery}%</span> remaining</td>
        </tr>
        <tr>
          <td>Total steps</td>
          <td><span class="mono-val">${totalSteps}</span> steps</td>
        </tr>
        <tr>
          <td>Evidence fragments</td>
          <td><span class="mono-val">${missionData.clues?.length || 3}</span> analysed</td>
        </tr>
      </tbody>
    </table>

    <h4 style="font-size: 12px; font-weight: 600; color: var(--text); margin-top: 14px; margin-bottom: 4px;">
      Recommended next operational step
    </h4>
    <p style="background: var(--bg); padding: 8px 10px; border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted);">
      Establish optical handshake and auxiliary power connection to awaken target system at sector ${targetCoord}.
    </p>
  `;
  reportModal.classList.remove("hidden");
}

function downloadTraceJson() {
  if (!missionData) return;
  const blob = new Blob([JSON.stringify(missionData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `signal_navigator_trace.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Procedural Fallback Generator (if trace is not available)
 */
function generateDemonstrationTrace(missionKey) {
  const gridW = 16, gridH = 16;
  const obstacles = [[3,3], [3,4], [3,5], [4,5], [5,5], [8,8], [8,9], [9,8], [9,9], [11,2], [11,3], [12,2], [6,12], [7,12]];
  const solar_recharge_stations = [[1,1], [1,14], [14,1], [8,7]];
  const hazards = [[4,8], [5,8], [6,8]];
  const clues = [
    { position: [4, 2], id: "clue_01" },
    { position: [10, 4], id: "clue_02" },
    { position: [12, 10], id: "clue_03" }
  ];

  const positions = [
    [1,1], [1,1], [2,1], [3,1], [4,1], [4,2], [4,2], [5,2], [6,2], [7,2],
    [8,3], [9,3], [10,4], [10,4], [9,5], [8,6], [8,7], [8,7], [9,8], [10,9],
    [11,10], [12,10], [12,10], [12,11], [12,12], [13,13]
  ];

  const trace_history = positions.map((pos, idx) => {
    let mode = "TRACKING_EVE";
    let action = "MOVE";
    let reasoning = `Evaluating spatial trajectory towards sector (${pos[0]}, ${pos[1]}).`;
    let bat = 100 - (idx * 3);

    if (idx === 0) {
      action = "LONG_RANGE_PING";
      mode = "INITIALIZING";
      reasoning = "Initial calibration ping: received RSSI 0.28 from south-east quadrant.";
    } else if (idx === 6 || idx === 13 || idx === 22) {
      action = "ANALYZE_CLUE";
      mode = "ANALYZING_CLUE";
      reasoning = `Environmental artifact analyzed. Evidence extracted, narrowing search corridor.`;
    } else if (idx === 17) {
      action = "RECHARGE";
      mode = "RECHARGING";
      bat += 25;
      reasoning = "Docked at solar recharge array [8,7]. Replenished battery to safe operational threshold.";
    }

    const entropy = Math.max(0.0, 7.84 - (idx * 0.32));
    return {
      step: idx,
      agent_position: pos,
      battery_level: Math.min(100, Math.max(15, bat)),
      battery_cost: action === "RECHARGE" ? 0 : 2,
      action: action,
      action_params: { direction: "SOUTH" },
      agent_mode: mode,
      reasoning: reasoning,
      belief_entropy_bits: entropy,
      top_hypotheses: [
        { coordinate: [13, 13], probability: 0.15 + (idx * 0.032), confidence_percent: Math.min(100, 20 + idx * 3.3) },
        { coordinate: [12, 11], probability: 0.08, confidence_percent: 12.0 },
        { coordinate: [10, 12], probability: 0.04, confidence_percent: 6.0 }
      ]
    };
  });

  return {
    mission_name: "Mission Sector 7 - Wasteland Discovery",
    grid_dimensions: [gridW, gridH],
    obstacles,
    solar_recharge_stations,
    hazards,
    clues,
    eve_target_ground_truth: [13, 13],
    trace_history
  };
}
