/**
 * BATTERY INTELLIGENCE — Dashboard Client Application
 * Phase 7: Validated ML Models (M1–M5), Adaptive Counterfactual Trust,
 * Peer-Relative Trajectory Overlays, PMRI, Research Benchmarks, and Hardware Validation.
 */

// Application State
const state = {
  currentBatteryId: "b1c0",
  currentCutoff: 100, // 40, 70, 100, or 'all'
  currentModel: "M5", // 'M1', 'M2', 'M3', 'M4', 'M5'
  currentView: "overview",
  currentMetric: "QDischarge", // 'QDischarge', 'IR', 'Tmax', 'Tavg', 'chargetime'
  currentFeatureGroup: "Initial Condition",
  comparisonCells: ["b1c0", "b1c1"],
  comparisonMetric: "QDischarge",
  batteriesList: [],
  currentBatteryData: null,
  currentM5Data: null,
  overviewData: null,
  researchSummary: null,
  researchMetrics: null,
  researchDifferences: null,
  multiAlgoMetrics: null,
  peerControlsData: null,
  selectivePredictionData: null,
  m5Ablations: null,
  m5Summary: null,
  trajectoryChart: null,
  comparisonChart: null,
  peerTrajectoryChart: null,
  sensitivityChart: null,
};

// Color Palette for Multi-cell Overlays
const PALETTE = [
  "#38bdf8", // Sky blue
  "#34d399", // Emerald
  "#fbbf24", // Amber
  "#f43f5e", // Rose
  "#a78bfa", // Purple
  "#fb923c", // Orange
  "#2dd4bf", // Teal
  "#818cf8", // Indigo
];

// Metric Metadata (Display Names and Units)
const METRIC_META = {
  QDischarge: { label: "Discharge Capacity", unit: "Ah", color: "#38bdf8" },
  IR: { label: "Internal Resistance", unit: "Ω", color: "#fbbf24" },
  Tmax: { label: "Maximum Temperature", unit: "°C", color: "#f43f5e" },
  Tavg: { label: "Average Temperature", unit: "°C", color: "#2dd4bf" },
  chargetime: { label: "Charging Duration", unit: "min", color: "#818cf8" },
};

// DOM Content Loaded Handler
document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
});

/**
 * Initializes the dashboard by fetching database overview, research artifacts, and default battery.
 */
async function initDashboard() {
  try {
    // 1. Fetch system overview
    const resOverview = await fetch("/api/overview");
    if (resOverview.ok) {
      state.overviewData = await resOverview.json();
      document.getElementById("stat-total-cells").textContent = state.overviewData.total_cells;
      document.getElementById("stat-eligible-cells").textContent = state.overviewData.eligible_cells;
      document.getElementById("stat-excluded-cells").textContent = state.overviewData.excluded_cells;
      const mCountEl = document.getElementById("stat-models-count");
      if (mCountEl) mCountEl.textContent = state.overviewData.total_models || 5;
      const pCountEl = document.getElementById("stat-predictions-count");
      if (pCountEl) pCountEl.textContent = state.overviewData.total_predictions || 372;

      // Update overview risk breakdown cards
      updateOverviewRiskBreakdown();
    }

    // 2. Fetch all batteries list
    const resBatteries = await fetch("/api/batteries");
    if (!resBatteries.ok) throw new Error("Failed to load batteries list.");
    state.batteriesList = await resBatteries.json();

    // 3. Populate battery selector dropdowns
    populateBatteryDropdown();
    populateCompareDropdown();

    // 4. Attach event listeners
    setupEventListeners();
    setupNavigation();
    setupHardwareValidation();

    // 5. Load research section data asynchronously
    loadAndRenderResearchSection();
    loadAndRenderM5Ablations();

    // 6. Load initial default battery
    await loadBattery(state.currentBatteryId);
    await updateComparisonChart();

  } catch (err) {
    console.error("Dashboard initialization error:", err);
  }
}

/**
 * Sets up sidebar navigation and view panel switching.
 */
function setupNavigation() {
  const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
  navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.dataset.view;
      if (targetView) {
        switchView(targetView);
      }
    });
  });

  // Mobile sidebar toggle
  const toggleBtn = document.getElementById("sidebar-toggle");
  const sidebar = document.getElementById("app-sidebar");
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }
}

/**
 * Switches the active dashboard view.
 */
window.switchView = function (viewName) {
  state.currentView = viewName;

  // Update nav item active states
  const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
  navItems.forEach((item) => {
    if (item.dataset.view === viewName) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Update view panel visibility
  const panels = document.querySelectorAll(".view-panel");
  panels.forEach((p) => {
    p.classList.remove("active");
  });
  const activePanel = document.getElementById(`view-${viewName}`);
  if (activePanel) {
    activePanel.classList.add("active");
  }

  // Close mobile sidebar if open
  const sidebar = document.getElementById("app-sidebar");
  if (sidebar && sidebar.classList.contains("open")) {
    sidebar.classList.remove("open");
  }

  // Trigger chart resizes for hidden canvases
  setTimeout(() => {
    if (state.trajectoryChart) state.trajectoryChart.resize();
    if (state.comparisonChart) state.comparisonChart.resize();
    if (state.peerTrajectoryChart) state.peerTrajectoryChart.resize();
    if (state.sensitivityChart) state.sensitivityChart.resize();
  }, 50);

  // Scroll to top of workspace
  window.scrollTo({ top: 0, behavior: "smooth" });
};

/**
 * Populates the primary battery selector dropdown.
 */
function populateBatteryDropdown() {
  const select = document.getElementById("battery-select");
  select.innerHTML = "";

  const batches = {
    batch1: { label: "Batch 1 (2017-05-12) — 46 cells", items: [] },
    batch2: { label: "Batch 2 (2017-06-30) — 48 cells", items: [] },
    batch3: { label: "Batch 3 (2018-04-12) — 46 cells", items: [] },
  };

  state.batteriesList.forEach((b) => {
    if (batches[b.batch]) {
      batches[b.batch].items.push(b);
    }
  });

  Object.entries(batches).forEach(([batchKey, bInfo]) => {
    const optgroup = document.createElement("optgroup");
    optgroup.label = bInfo.label;

    bInfo.items.forEach((cell) => {
      const opt = document.createElement("option");
      opt.value = cell.cell_id;
      const eligibilityTag = cell.is_benchmark_eligible ? "" : " [EXCLUDED]";
      opt.textContent = `${cell.cell_id} (${cell.policy})${eligibilityTag}`;
      if (cell.cell_id === state.currentBatteryId) {
        opt.selected = true;
      }
      optgroup.appendChild(opt);
    });

    select.appendChild(optgroup);
  });
}

/**
 * Populates the comparison battery picker dropdown.
 */
function populateCompareDropdown() {
  const select = document.getElementById("compare-cell-select");
  if (!select) return;
  select.innerHTML = '<option value="">+ Add cell to comparison...</option>';

  state.batteriesList.forEach((cell) => {
    if (!state.comparisonCells.includes(cell.cell_id) && cell.is_benchmark_eligible) {
      const opt = document.createElement("option");
      opt.value = cell.cell_id;
      opt.textContent = `${cell.cell_id} (${cell.batch}, ${cell.policy})`;
      select.appendChild(opt);
    }
  });
}

/**
 * Configures all event listeners for interactive filters and buttons.
 */
function setupEventListeners() {
  // Battery Selection Change
  document.getElementById("battery-select").addEventListener("change", (e) => {
    const newCellId = e.target.value;
    if (newCellId) {
      state.currentBatteryId = newCellId;
      loadBattery(newCellId);
    }
  });

  // Observation Cutoff Pills
  const cutoffButtons = document.querySelectorAll("#cutoff-selector .pill-btn");
  cutoffButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      cutoffButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const val = btn.dataset.cutoff;
      state.currentCutoff = val === "all" ? "all" : parseInt(val, 10);
      const maskEl = document.getElementById("mask-cutoff-label");
      if (maskEl) maskEl.textContent = val;
      loadBattery(state.currentBatteryId);
      updateComparisonChart();
    });
  });

  // Model Selector Pills
  const modelButtons = document.querySelectorAll("#model-selector .pill-btn");
  modelButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modelButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentModel = btn.dataset.model;
      if (state.currentBatteryData) {
        renderRiskPredictions(state.currentBatteryData.model_predictions, state.currentBatteryData.metadata, state.currentCutoff);
      }
      updateElevatedRiskTally();
    });
  });

  // Metric Switcher Tabs for Main Trajectory Chart
  const metricTabs = document.querySelectorAll("#metric-tab-bar .tab-btn");
  metricTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      metricTabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentMetric = btn.dataset.metric;
      renderTrajectoryChart();
    });
  });

  // Feature Group Tabs
  const featureTabs = document.querySelectorAll("#feature-group-tabs .tab-btn");
  featureTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      featureTabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentFeatureGroup = btn.dataset.group;
      renderFeaturesTable();
    });
  });

  // Add Cell to Comparison
  const compSelect = document.getElementById("compare-cell-select");
  if (compSelect) {
    compSelect.addEventListener("change", (e) => {
      const cid = e.target.value;
      if (cid && !state.comparisonCells.includes(cid)) {
        state.comparisonCells.push(cid);
        updateComparisonChart();
        populateCompareDropdown();
      }
      e.target.value = "";
    });
  }

  // Quick Cohort: Same Policy
  const btnSamePolicy = document.getElementById("btn-compare-same-policy");
  if (btnSamePolicy) {
    btnSamePolicy.addEventListener("click", () => {
      if (!state.currentBatteryData) return;
      const policy = state.currentBatteryData.metadata.policy;
      const matched = state.batteriesList
        .filter((b) => b.policy === policy && b.is_benchmark_eligible)
        .map((b) => b.cell_id);

      if (matched.length > 0) {
        state.comparisonCells = Array.from(new Set([state.currentBatteryId, ...matched])).slice(0, 5);
        updateComparisonChart();
        populateCompareDropdown();
      }
    });
  }

  // Quick Cohort: Across Batches
  const btnBatches = document.getElementById("btn-compare-batch-samples");
  if (btnBatches) {
    btnBatches.addEventListener("click", () => {
      state.comparisonCells = ["b1c0", "b2c0", "b3c0"];
      updateComparisonChart();
      populateCompareDropdown();
    });
  }

  // Reset Comparison
  const btnClearComp = document.getElementById("btn-clear-compare");
  if (btnClearComp) {
    btnClearComp.addEventListener("click", () => {
      state.comparisonCells = [state.currentBatteryId];
      updateComparisonChart();
      populateCompareDropdown();
    });
  }

  // Research Tabs Switcher
  const rTabs = document.querySelectorAll("#research-tabs-bar .research-tab-btn");
  rTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      rTabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const targetPanelId = btn.dataset.tab;
      document.querySelectorAll(".research-panel").forEach((p) => (p.style.display = "none"));
      const p = document.getElementById(targetPanelId);
      if (p) p.style.display = "block";
    });
  });
}

/**
 * Loads and renders battery details across all dashboard sections.
 */
async function loadBattery(cellId) {
  try {
    const cutoff = state.currentCutoff;
    const effectiveCutoff = cutoff === "all" ? 100 : cutoff;

    // Fetch primary battery details
    const res = await fetch(`/api/battery?id=${cellId}&cutoff=${cutoff}`);
    if (!res.ok) throw new Error(`Could not load battery data for ${cellId}.`);
    const data = await res.json();
    state.currentBatteryData = data;

    // 1. Render Summary Cards
    renderSummaryCards(data.summary_cards, data.metadata);

    // 2. Render Metadata Strip & Quick Tally
    renderMetadataStrip(data.metadata);
    updateElevatedRiskTally();

    // 3. Render Features Table
    renderFeaturesTable();

    // 4. Render Main Trajectory Chart
    renderTrajectoryChart();

    // 5. Render ML Risk Predictions & Comparison
    renderRiskPredictions(data.model_predictions, data.metadata, cutoff);

    // 6. Render Sequestered Ground-Truth Targets
    renderGroundTruthTargets(data.targets, data.metadata);

    // 7. Render Factual Observations
    renderFactualObservations(data.factual_observations);

    // 8. Render Peer-Relative Aligned Trajectory Chart & PMRI
    renderPeerRelativeSection(cellId, effectiveCutoff);
    renderPMRISection(data.peer_info);

    // 9. Render M5 Robustness Engine Section
    renderM5Section(cellId, effectiveCutoff);

  } catch (err) {
    console.error("Error loading battery:", err);
  }
}

/**
 * Updates the Overview view model risk breakdown counts.
 */
function updateOverviewRiskBreakdown() {
  if (!state.overviewData || !state.overviewData.elevated_risk_counts) return;
  const cutoffCounts = state.overviewData.elevated_risk_counts[100] || {};
  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val !== undefined ? `${val} / 124` : "--";
  };
  setEl("rmb-m1", cutoffCounts["M1"]);
  setEl("rmb-m2", cutoffCounts["M2"]);
  setEl("rmb-m3", cutoffCounts["M3"]);
  setEl("rmb-m4", cutoffCounts["M4"]);
  setEl("rmb-m5", cutoffCounts["M5"]);
}

/**
 * Renders the top metadata strip tags.
 */
function renderMetadataStrip(meta) {
  document.getElementById("meta-batch").textContent = meta.batch;
  document.getElementById("meta-partition").textContent = meta.benchmark_partition;
  document.getElementById("meta-policy").textContent = meta.policy;
}

/**
 * Updates the Elevated Risk Count pill in the controller strip.
 */
function updateElevatedRiskTally() {
  const tallyEl = document.getElementById("meta-risk-tally");
  if (!tallyEl || !state.overviewData || !state.overviewData.elevated_risk_counts) return;

  const effectiveCutoff = state.currentCutoff === "all" ? 100 : state.currentCutoff;
  const cutoffMap = state.overviewData.elevated_risk_counts[effectiveCutoff];

  if (cutoffMap && cutoffMap[state.currentModel] !== undefined) {
    const count = cutoffMap[state.currentModel];
    tallyEl.textContent = `${count} / 124 Cells (${((count / 124) * 100).toFixed(1)}%)`;
  } else {
    tallyEl.textContent = "--";
  }
}

/**
 * Populates physical summary metric cards.
 */
function renderSummaryCards(cards, meta) {
  document.getElementById("card-cell-id").textContent = meta.cell_id;
  const statusEl = document.getElementById("card-cell-status");

  if (meta.is_benchmark_eligible) {
    statusEl.innerHTML = `<span class="badge badge-teal">${meta.benchmark_partition} (Eligible)</span>`;
  } else {
    statusEl.innerHTML = `<span class="badge badge-amber">Excluded Channel</span>`;
  }

  document.getElementById("card-observed-cycles").textContent = cards.observed_cycles_in_view;
  document.getElementById("card-total-cycles").textContent = `of ${cards.total_cycles_recorded} recorded`;

  document.getElementById("card-latest-q").textContent = cards.q_latest_ah !== null ? cards.q_latest_ah.toFixed(3) : "--";
  document.getElementById("card-initial-q").textContent = `Baseline (Cyc 2): ${cards.q_start_ah !== null ? cards.q_start_ah.toFixed(3) + " Ah" : "--"}`;

  document.getElementById("card-retention").textContent = cards.capacity_retention_pct !== null ? `${cards.capacity_retention_pct.toFixed(1)}%` : "--";
  document.getElementById("card-fade-delta").textContent = `Fade ΔQ: ${cards.capacity_fade_delta_ah !== null ? cards.capacity_fade_delta_ah.toFixed(3) + " Ah" : "--"}`;

  document.getElementById("card-latest-ir").textContent = cards.ir_latest_ohm !== null ? cards.ir_latest_ohm.toFixed(4) : "--";
  document.getElementById("card-ir-growth").textContent = `Growth: ${cards.ir_growth_pct !== null ? (cards.ir_growth_pct >= 0 ? "+" : "") + cards.ir_growth_pct.toFixed(1) + "%" : "--"}`;

  document.getElementById("card-tmax").textContent = cards.tmax_latest_c !== null ? cards.tmax_latest_c.toFixed(1) : "--";
  document.getElementById("card-chargetime").textContent = `Charge time: ${cards.chargetime_latest_min !== null ? cards.chargetime_latest_min.toFixed(1) + " min" : "--"}`;
}

/**
 * Renders the main cycle-by-cycle trajectory chart using Chart.js.
 */
function renderTrajectoryChart() {
  const ctx = document.getElementById("trajectory-chart").getContext("2d");
  if (!state.currentBatteryData) return;

  const cycles = state.currentBatteryData.cycles;
  const metric = state.currentMetric;
  const metaInfo = METRIC_META[metric];

  const points = cycles.map((c) => ({
    x: c.cycle,
    y: c[metric],
  })).filter((p) => p.y !== null && p.y !== undefined);

  if (state.trajectoryChart) {
    state.trajectoryChart.destroy();
  }

  const datasets = [
    {
      label: `${metaInfo.label} (${state.currentBatteryId})`,
      data: points,
      borderColor: metaInfo.color,
      backgroundColor: "rgba(56, 189, 248, 0.05)",
      borderWidth: 2,
      pointRadius: points.length > 200 ? 0 : 2,
      pointHoverRadius: 4,
      tension: 0.1,
      fill: false,
    },
  ];

  // Capacity threshold reference lines
  if (metric === "QDischarge") {
    const maxX = Math.max(...points.map((p) => p.x), 100);
    datasets.push({
      label: "Nominal Capacity (1.10 Ah)",
      data: [{ x: 1, y: 1.10 }, { x: maxX, y: 1.10 }],
      borderColor: "#10b981",
      borderWidth: 1.5,
      borderDash: [4, 4],
      pointRadius: 0,
      fill: false,
    });
    datasets.push({
      label: "80% EOL Threshold (0.88 Ah)",
      data: [{ x: 1, y: 0.88 }, { x: maxX, y: 0.88 }],
      borderColor: "#f43f5e",
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
    });
  }

  // Vertical cutoff reference indicator
  if (state.currentCutoff !== "all") {
    const minY = Math.min(...points.map((p) => p.y)) * 0.95;
    const maxY = Math.max(...points.map((p) => p.y)) * 1.05;
    datasets.push({
      label: `Observation Cutoff (${state.currentCutoff})`,
      data: [
        { x: state.currentCutoff, y: minY },
        { x: state.currentCutoff, y: maxY },
      ],
      borderColor: "rgba(255, 255, 255, 0.4)",
      borderWidth: 1.5,
      borderDash: [4, 4],
      pointRadius: 0,
      fill: false,
    });
  }

  state.trajectoryChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#162035",
          borderColor: "#23314d",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: (items) => `Cycle ${items[0].parsed.x}`,
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)} ${metaInfo.unit}`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Cycle Number", color: "#94a3b8", font: { size: 12, weight: 600 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
        y: {
          title: { display: true, text: `${metaInfo.label} (${metaInfo.unit})`, color: "#94a3b8", font: { size: 12, weight: 600 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

/**
 * Renders the ML EOL Risk Prediction card and five-model comparison table.
 */
function renderRiskPredictions(preds, meta, cutoff) {
  const probNumEl = document.getElementById("risk-prob-num");
  const modelLblEl = document.getElementById("risk-model-label");
  const cutoffLblEl = document.getElementById("risk-cutoff-label");
  const statusBadgeEl = document.getElementById("risk-status-badge");
  const detailModelEl = document.getElementById("risk-detail-model");
  const detailThreshEl = document.getElementById("risk-detail-thresh");
  const detailFeaturesEl = document.getElementById("risk-detail-features");
  const detailPartitionEl = document.getElementById("risk-detail-partition");
  const tbody = document.getElementById("models-comparison-tbody");

  const effectiveCutoff = cutoff === "all" ? 100 : cutoff;
  modelLblEl.textContent = state.currentModel;
  cutoffLblEl.textContent = effectiveCutoff;

  if (!preds || !preds.model_details) {
    probNumEl.textContent = "N/A";
    statusBadgeEl.className = "risk-badge";
    statusBadgeEl.textContent = "Excluded / Unavailable";
    detailModelEl.textContent = `${state.currentModel}`;
    detailThreshEl.textContent = "--";
    detailFeaturesEl.textContent = "--";
    detailPartitionEl.textContent = meta.benchmark_partition || "Excluded";
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No validated predictions available for excluded battery.</td></tr>`;
    return;
  }

  const activeModelKey = state.currentModel;
  const activeDetail = preds.model_details[activeModelKey];

  if (activeDetail && activeDetail.probability !== null) {
    const probPct = (activeDetail.probability * 100).toFixed(2);
    probNumEl.textContent = `${probPct}%`;
    probNumEl.className = activeDetail.is_elevated_risk ? "risk-score-num text-rose" : "risk-score-num text-cyan";

    if (activeDetail.is_elevated_risk) {
      statusBadgeEl.className = "risk-badge risk-badge-elevated";
      statusBadgeEl.innerHTML = `⚠️ ELEVATED FAILURE RISK (≥ ${activeDetail.threshold.toFixed(4)})`;
    } else {
      statusBadgeEl.className = "risk-badge risk-badge-nominal";
      statusBadgeEl.innerHTML = `✓ NOMINAL RISK (&lt; ${activeDetail.threshold.toFixed(4)})`;
    }

    detailModelEl.textContent = `${activeModelKey}: ${activeDetail.name}`;
    detailThreshEl.textContent = `${activeDetail.threshold.toFixed(4)} (5-Fold CV Optimal)`;
    detailFeaturesEl.textContent = `${activeDetail.feature_count} Features`;
    detailPartitionEl.textContent = `${preds.benchmark_partition} (${preds.validation_status})`;
  } else {
    probNumEl.textContent = "--%";
    statusBadgeEl.className = "risk-badge";
    statusBadgeEl.textContent = "Prediction Unavailable";
  }

  // Populate Five-Model Comparative Table
  tbody.innerHTML = "";
  const modelOrder = ["M1", "M2", "M3", "M4", "M5"];
  modelOrder.forEach((mKey) => {
    const det = preds.model_details[mKey];
    if (!det) return;

    const tr = document.createElement("tr");
    if (mKey === activeModelKey) {
      tr.style.backgroundColor = "rgba(56, 189, 248, 0.08)";
      tr.style.fontWeight = "600";
    }

    const probVal = det.probability !== null ? (det.probability * 100).toFixed(2) + "%" : "--";
    const riskBadgeHtml = det.is_elevated_risk
      ? `<span class="badge badge-danger" style="font-size:0.72rem;">Elevated Risk</span>`
      : `<span class="badge badge-success" style="font-size:0.72rem;">Nominal</span>`;

    tr.innerHTML = `
      <td><strong>${mKey}</strong></td>
      <td>${det.name}</td>
      <td class="text-cyan">${det.feature_count}</td>
      <td class="text-muted" style="font-size:0.78rem;">${det.groups.join(" + ")}</td>
      <td><strong>${probVal}</strong></td>
      <td class="text-muted">${det.threshold.toFixed(4)}</td>
      <td>${riskBadgeHtml}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Fetches and renders the peer-relative aligned trajectory overlay chart and relative metrics.
 */
async function renderPeerRelativeSection(cellId, cutoff) {
  const effectiveCutoff = cutoff === "all" ? 100 : cutoff;
  const chipsContainer = document.getElementById("peer-chips-container");
  const metricsTbody = document.getElementById("peer-metrics-tbody");
  chipsContainer.innerHTML = "";
  metricsTbody.innerHTML = "";

  try {
    const res = await fetch(`/api/peers/trajectories?id=${cellId}&cutoff=${effectiveCutoff}`);
    if (!res.ok) {
      chipsContainer.innerHTML = `<span class="text-muted" style="font-size:0.82rem;">No peer match data available for this cell at Cutoff ${effectiveCutoff}.</span>`;
      return;
    }

    const peerData = await res.json();
    state.peerTrajectoryData = peerData;

    // Render peer chips
    (peerData.peers || []).forEach((p, idx) => {
      const chip = document.createElement("div");
      chip.className = "peer-badge-chip";
      chip.innerHTML = `
        <span style="color:${PALETTE[idx + 1]}; font-size:1.1rem; line-height:0;">●</span>
        <span>Peer ${idx + 1}: <strong>${p.cell_id}</strong></span>
        <span class="text-muted">dist = ${p.distance.toFixed(3)}</span>
      `;
      chipsContainer.appendChild(chip);
    });

    // Render Peer Trajectory Overlay Chart
    renderPeerOverlayChart(peerData);

    // Populate Relative Deviation Metrics Table
    const relMetrics = [
      {
        name: "Relative Capacity Deviation (ΔQ_rel)",
        col: "peer_rel_capacity_fade_delta",
        val: state.currentBatteryData.peer_info ? state.currentBatteryData.peer_info.peer_rel_capacity_fade_delta : null,
        desc: "Difference between cell capacity fade and mean peer capacity fade.",
      },
      {
        name: "Relative Resistance Growth (ΔIR_rel)",
        col: "peer_rel_ir_growth_delta",
        val: state.currentBatteryData.peer_info ? state.currentBatteryData.peer_info.peer_rel_ir_growth_delta : null,
        desc: "Differential internal resistance growth rate compared to matched reference peers.",
      },
      {
        name: "Cohort Trajectory Dispersion",
        col: "peer_dispersion_q",
        val: state.currentBatteryData.peer_info ? state.currentBatteryData.peer_info.peer_dispersion_q : null,
        desc: "Standard deviation of discharge capacity across the k=5 peers at cutoff.",
      },
      {
        name: "Mean Reference Peer Distance",
        col: "peer_dist_mean",
        val: state.currentBatteryData.peer_info ? state.currentBatteryData.peer_info.peer_dist_mean : null,
        desc: "Average standardized Euclidean distance in initial state space (Q2, IR2, protocol).",
      },
    ];

    relMetrics.forEach((m) => {
      const tr = document.createElement("tr");
      const valStr = m.val !== null && m.val !== undefined ? (typeof m.val === "number" ? m.val.toFixed(4) : m.val) : "--";
      tr.innerHTML = `
        <td><strong>${m.name}</strong></td>
        <td><span class="code-badge">${m.col}</span></td>
        <td class="feat-val">${valStr}</td>
        <td class="text-muted">${m.desc}</td>
      `;
      metricsTbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Error rendering peer section:", err);
  }
}

/**
 * Renders the aligned peer overlay Chart.js chart.
 */
function renderPeerOverlayChart(peerData) {
  const ctx = document.getElementById("peer-trajectory-chart").getContext("2d");
  if (state.peerTrajectoryChart) {
    state.peerTrajectoryChart.destroy();
  }

  const datasets = [];

  // 1. Target Cell Trajectory
  datasets.push({
    label: `Target Battery (${peerData.cell_id})`,
    data: peerData.target_trajectory.map((p) => ({ x: p.cycle, y: p.q_discharge })),
    borderColor: "#38bdf8",
    backgroundColor: "transparent",
    borderWidth: 3,
    pointRadius: 0,
    tension: 0.1,
    zIndex: 10,
  });

  // 2. Peer Trajectories (k=5)
  (peerData.peers || []).forEach((peer, idx) => {
    datasets.push({
      label: `Peer ${idx + 1} (${peer.cell_id})`,
      data: peer.trajectory.map((p) => ({ x: p.cycle, y: p.q_discharge })),
      borderColor: PALETTE[idx + 1] || "#94a3b8",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      borderDash: [2, 2],
      pointRadius: 0,
      tension: 0.1,
    });
  });

  // 3. Centroid / Mean Peer Trajectory
  if (peerData.mean_peer_trajectory && peerData.mean_peer_trajectory.length > 0) {
    datasets.push({
      label: "Reference Peer Centroid (Mean)",
      data: peerData.mean_peer_trajectory.map((p) => ({ x: p.cycle, y: p.mean_q })),
      borderColor: "#0d9488",
      backgroundColor: "transparent",
      borderWidth: 2.5,
      borderDash: [5, 5],
      pointRadius: 0,
      tension: 0.1,
    });
  }

  state.peerTrajectoryChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          display: true,
          labels: { color: "#94a3b8", font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
          backgroundColor: "#162035",
          borderColor: "#23314d",
          borderWidth: 1,
          callbacks: {
            title: (items) => `Cycle ${items[0].parsed.x}`,
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)} Ah`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Observation Cycle", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
        y: {
          title: { display: true, text: "Discharge Capacity (Ah)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

/**
 * Renders the PMRI scores and component progress bars.
 */
function renderPMRISection(pInfo) {
  const compValEl = document.getElementById("pmri-composite-val");
  const meanDistEl = document.getElementById("pmri-mean-dist");
  const dispEl = document.getElementById("pmri-dispersion");
  const warnBadge = document.getElementById("pmri-warning-badge");

  if (!pInfo || pInfo.pmri === null || pInfo.pmri === undefined) {
    compValEl.textContent = "N/A";
    meanDistEl.textContent = "--";
    dispEl.textContent = "--";
    warnBadge.className = "badge badge-warning";
    warnBadge.textContent = "Censored / Excluded";
    return;
  }

  compValEl.textContent = pInfo.pmri.toFixed(4);
  meanDistEl.textContent = pInfo.peer_dist_mean ? pInfo.peer_dist_mean.toFixed(4) : "--";
  dispEl.textContent = pInfo.peer_dispersion_q ? `${pInfo.peer_dispersion_q.toFixed(4)} Ah` : "--";

  if (pInfo.reliability_warning && pInfo.reliability_warning !== "NONE") {
    warnBadge.className = "badge badge-danger";
    warnBadge.textContent = `Warning: ${pInfo.reliability_warning}`;
  } else {
    warnBadge.className = "badge badge-success";
    warnBadge.textContent = "Reliable Peer Match";
  }

  // Component Progress Bars
  const setBar = (fillId, valId, val) => {
    const fEl = document.getElementById(fillId);
    const vEl = document.getElementById(valId);
    if (fEl && vEl) {
      if (val !== null && val !== undefined) {
        fEl.style.width = `${Math.min(100, Math.max(0, val * 100))}%`;
        vEl.textContent = val.toFixed(4);
      } else {
        fEl.style.width = "0%";
        vEl.textContent = "--";
      }
    }
  };

  setBar("bar-s-sim", "val-s-sim", pInfo.pmri_s_sim);
  setBar("bar-s-cov", "val-s-cov", pInfo.pmri_s_cov);
  setBar("bar-s-agree", "val-s-agree", pInfo.pmri_s_agree);
  setBar("bar-s-consist", "val-s-consist", pInfo.pmri_s_consist);
}

/**
 * Fetches and renders the M5 Robustness Engine live diagnostics, perturbation audit traces, and sensitivity waterfall.
 */
async function renderM5Section(cellId, cutoff) {
  const effectiveCutoff = cutoff === "all" ? 100 : cutoff;
  document.getElementById("m5-active-cell").textContent = cellId;
  document.getElementById("m5-active-cutoff").textContent = effectiveCutoff;

  const probPeerEl = document.getElementById("m5-prob-peer");
  const probHistEl = document.getElementById("m5-prob-history");
  const uEl = document.getElementById("m5-instability-u");
  const tEl = document.getElementById("m5-trust-score");
  const finalEl = document.getElementById("m5-prob-final");
  const badgeEl = document.getElementById("m5-abstention-badge");
  const explanationEl = document.getElementById("m5-explanation-box");
  const tracesTbody = document.getElementById("m5-perturbations-tbody");

  try {
    const res = await fetch(`/api/m5/robustness?id=${cellId}&cutoff=${effectiveCutoff}`);
    if (!res.ok) {
      probPeerEl.textContent = "N/A";
      probHistEl.textContent = "N/A";
      uEl.textContent = "--";
      tEl.textContent = "--";
      finalEl.textContent = "N/A";
      badgeEl.className = "badge badge-outline";
      badgeEl.textContent = "Unavailable";
      explanationEl.textContent = "M5 robustness data unavailable for this channel.";
      tracesTbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No perturbations available.</td></tr>`;
      return;
    }

    const m5 = await res.json();
    state.currentM5Data = m5;

    probPeerEl.textContent = `${(m5.prob_peer * 100).toFixed(2)}%`;
    probHistEl.textContent = `${(m5.prob_history * 100).toFixed(2)}%`;
    uEl.textContent = m5.instability_u !== null ? (m5.instability_u < 0.001 ? m5.instability_u.toExponential(3) : m5.instability_u.toFixed(4)) : "--";
    tEl.textContent = m5.trust_score !== null ? m5.trust_score.toFixed(4) : "--";
    finalEl.textContent = `${(m5.prob_m5 * 100).toFixed(2)}%`;

    // Abstention badge
    badgeEl.textContent = m5.abstention_reason || "NOMINAL_TRUST";
    if (m5.abstention_reason === "NOMINAL_TRUST") {
      badgeEl.className = "badge badge-success";
    } else if (m5.abstention_reason === "LOW_ADAPTIVE_TRUST") {
      badgeEl.className = "badge badge-warning";
    } else {
      badgeEl.className = "badge badge-danger";
    }

    explanationEl.innerHTML = `<strong>Inference Policy:</strong> ${m5.explanation}`;

    // Populate perturbation traces table
    tracesTbody.innerHTML = "";
    (m5.perturbations || []).forEach((p) => {
      const tr = document.createElement("tr");
      const deltaStr = p.prob_delta !== null ? (p.prob_delta >= 0 ? "+" : "") + (p.prob_delta * 100).toFixed(3) + "%" : "0.000%";
      const flipBadge = p.prediction_flip
        ? `<span class="badge badge-danger" style="font-size:0.68rem;">FLIP</span>`
        : `<span class="badge badge-outline" style="font-size:0.68rem;">Stable</span>`;

      tr.innerHTML = `
        <td><code>${p.perturbation_id}</code></td>
        <td>${p.perturbation_type}</td>
        <td>${p.excluded_peer || '<span class="text-muted">—</span>'}</td>
        <td>${p.added_peer || '<span class="text-muted">—</span>'}</td>
        <td><strong>${(p.predicted_prob * 100).toFixed(3)}%</strong></td>
        <td class="${p.prob_delta > 0 ? 'text-rose' : 'text-teal'}">${deltaStr}</td>
        <td>${flipBadge}</td>
      `;
      tracesTbody.appendChild(tr);
    });

    // Render Sensitivity Waterfall Chart
    renderM5SensitivityChart(m5.sensitivity_ranking || []);

  } catch (err) {
    console.error("Error rendering M5 section:", err);
  }
}

/**
 * Renders the M5 peer sensitivity waterfall bar chart.
 */
function renderM5SensitivityChart(ranking) {
  const ctx = document.getElementById("m5-sensitivity-chart").getContext("2d");
  if (state.sensitivityChart) {
    state.sensitivityChart.destroy();
  }

  if (!ranking || ranking.length === 0) {
    return;
  }

  const labels = ranking.map((r) => r.peer_id);
  const deltas = ranking.map((r) => r.prob_delta * 100);
  const colors = deltas.map((d) => (d >= 0 ? "rgba(244, 63, 94, 0.7)" : "rgba(13, 148, 136, 0.7)"));
  const borders = deltas.map((d) => (d >= 0 ? "#f43f5e" : "#0d9488"));

  state.sensitivityChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Δ Failure Probability on Peer Removal (%)",
          data: deltas,
          backgroundColor: colors,
          borderColor: borders,
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#162035",
          borderColor: "#23314d",
          borderWidth: 1,
          callbacks: {
            label: (item) => `Δ Risk: ${item.parsed.y >= 0 ? "+" : ""}${item.parsed.y.toFixed(4)}%`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Excluded Peer Cell ID (LOO)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
        y: {
          title: { display: true, text: "Δ Risk (%)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" },
        },
      },
    },
  });
}

/**
 * Loads and renders Phase 6 research benchmark artifacts and hypothesis verdicts.
 */
async function loadAndRenderResearchSection() {
  try {
    // 1. Fetch Model Metrics Table
    const resMetrics = await fetch("/api/models/metrics");
    if (resMetrics.ok) {
      state.researchMetrics = await resMetrics.json();
      renderResearchMetricsTable(state.researchMetrics);
    }

    // 2. Fetch Hypothesis Summary
    const resHyp = await fetch("/api/research/summary");
    if (resHyp.ok) {
      state.researchSummary = await resHyp.json();
      renderHypothesisCards(state.researchSummary);
    }

    // 3. Fetch Paired Differences
    const resDiff = await fetch("/api/research/differences");
    if (resDiff.ok) {
      state.researchDifferences = await resDiff.json();
      renderDifferencesTable(state.researchDifferences);
    }

    // 4. Fetch Multi-Algorithm Benchmark
    const resAlgo = await fetch("/api/research/algorithms");
    if (resAlgo.ok) {
      state.multiAlgoMetrics = await resAlgo.json();
      renderMultiAlgoTable(state.multiAlgoMetrics);
    }

    // 5. Fetch Multi-Seed Peer Controls
    const resCtrl = await fetch("/api/research/peer-controls");
    if (resCtrl.ok) {
      state.peerControlsData = await resCtrl.json();
      renderPeerControlsTable(state.peerControlsData);
    }

    // 6. Fetch Selective Prediction Trade-off
    const resSel = await fetch("/api/research/selective-prediction");
    if (resSel.ok) {
      state.selectivePredictionData = await resSel.json();
      renderSelectivePredictionTable(state.selectivePredictionData);
    }

    // Setup algorithm filter dropdown
    const algoFilter = document.getElementById("filter-algo-select");
    if (algoFilter && !algoFilter.dataset.bound) {
      algoFilter.dataset.bound = "true";
      algoFilter.addEventListener("change", (e) => {
        if (state.multiAlgoMetrics) {
          renderMultiAlgoTable(state.multiAlgoMetrics, e.target.value);
        }
      });
    }

  } catch (err) {
    console.error("Error loading research section:", err);
  }
}

/**
 * Loads and renders Phase 7 M5 ablation studies.
 */
async function loadAndRenderM5Ablations() {
  try {
    const res = await fetch("/api/m5/ablations");
    if (res.ok) {
      state.m5Ablations = await res.json();
      const tbody = document.getElementById("m5-ablations-tbody");
      if (!tbody) return;
      tbody.innerHTML = "";

      state.m5Ablations.forEach((row) => {
        const tr = document.createElement("tr");
        if (row.ablation_mode === "full") {
          tr.style.backgroundColor = "rgba(6, 182, 212, 0.08)";
          tr.style.fontWeight = "600";
        }
        tr.innerHTML = `
          <td><strong>Cutoff ${row.cutoff}</strong></td>
          <td><span class="code-badge">${row.ablation_mode}</span></td>
          <td>${row.description}</td>
          <td><strong>${row.primary_test_brier ? row.primary_test_brier.toFixed(4) : "--"}</strong></td>
          <td>${row.primary_test_auroc ? row.primary_test_auroc.toFixed(4) : "--"}</td>
          <td>${row.primary_test_auprc ? row.primary_test_auprc.toFixed(4) : "--"}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Error loading M5 ablations:", err);
  }
}

/**
 * Renders the Research Model Metrics Table.
 */
function renderResearchMetricsTable(metrics) {
  const tbody = document.getElementById("research-metrics-tbody");
  tbody.innerHTML = "";

  metrics.forEach((m) => {
    const tr = document.createElement("tr");
    if (m.model === "M5") {
      tr.style.backgroundColor = "rgba(6, 182, 212, 0.08)";
      tr.style.fontWeight = "600";
    }
    const fmt = (val) => (val !== null && val !== undefined ? (typeof val === "number" ? val.toFixed(4) : val) : "--");
    tr.innerHTML = `
      <td><strong>${m.cutoff} cyc</strong></td>
      <td><strong>${m.model}</strong></td>
      <td><span class="badge badge-outline">${m.partition}</span></td>
      <td class="text-muted">${m.n_samples} (${m.n_positive}/${m.n_negative})</td>
      <td>${fmt(m.auroc)}</td>
      <td>${fmt(m.auprc)}</td>
      <td><strong>${fmt(m.brier_score)}</strong></td>
      <td>${fmt(m.ece)}</td>
      <td>${fmt(m.accuracy)}</td>
      <td>${fmt(m.sensitivity)}</td>
      <td>${fmt(m.specificity)}</td>
      <td>${fmt(m.f1)}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the Hypothesis Verdict Cards.
 */
function renderHypothesisCards(summary) {
  const grid = document.getElementById("hypothesis-cards-grid");
  grid.innerHTML = "";

  const hypKeys = [
    { key: "H1_incremental_peer_relative_value", title: "H1: Incremental Peer-Relative Value (M3 vs M2)" },
    { key: "H2_peer_match_reliability_value", title: "H2: Peer-Match Reliability Value (M4 vs M3)" },
    { key: "H3_monotonic_cutoff_benefit", title: "H3: Monotonic Observation Benefit" },
    { key: "H4_robustness_across_conditions", title: "H4: Generalization Across Batches" },
  ];

  hypKeys.forEach((item) => {
    const data = summary[item.key];
    if (!data) return;

    const card = document.createElement("div");
    card.className = "hypothesis-card";

    let verdictClass = "hyp-verdict-parity";
    if (data.verdict.includes("SUPPORTED") && !data.verdict.includes("NOT")) {
      verdictClass = "hyp-verdict-supported";
    }

    card.innerHTML = `
      <div class="hyp-header">
        <span class="hyp-title">${item.title}</span>
        <span class="hyp-verdict ${verdictClass}">${data.verdict}</span>
      </div>
      <p class="hyp-statement">"${data.statement}"</p>
      <p class="hyp-rationale"><strong>Empirical Rationale:</strong> ${data.rationale}</p>
    `;
    grid.appendChild(card);
  });
}

/**
 * Renders the Paired Bootstrap Differences Table.
 */
function renderDifferencesTable(diffs) {
  const tbody = document.getElementById("research-diff-tbody");
  tbody.innerHTML = "";

  diffs.forEach((d) => {
    const tr = document.createElement("tr");
    const ciFmt = (mean, low, high) => {
      if (mean === null || mean === undefined) return "--";
      return `${mean >= 0 ? "+" : ""}${mean.toFixed(4)} [${low.toFixed(4)}, ${high.toFixed(4)}]`;
    };

    const sigBadge = d.significant
      ? `<span class="badge badge-success" style="font-size:0.72rem;">Significant</span>`
      : `<span class="badge badge-outline" style="font-size:0.72rem;">Neutral / Inconclusive</span>`;

    tr.innerHTML = `
      <td><strong>${d.cutoff} cyc</strong></td>
      <td><span class="badge badge-outline">${d.partition}</span></td>
      <td><strong>${d.comparison}</strong></td>
      <td>${ciFmt(d.delta_auroc_mean, d.delta_auroc_ci_lower, d.delta_auroc_ci_upper)}</td>
      <td>${ciFmt(d.delta_auprc_mean, d.delta_auprc_ci_lower, d.delta_auprc_ci_upper)}</td>
      <td>${ciFmt(d.delta_brier_mean, d.delta_brier_ci_lower, d.delta_brier_ci_upper)}</td>
      <td>${sigBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the Multi-Algorithm Benchmark Table with algorithm filter.
 */
function renderMultiAlgoTable(records, selectedAlgo = "all") {
  const tbody = document.getElementById("multi-algo-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const fmt = (val) => (val !== null && val !== undefined ? (typeof val === "number" ? val.toFixed(4) : val) : "--");

  const filtered = selectedAlgo === "all" ? records : records.filter((r) => r.algorithm === selectedAlgo);

  filtered.forEach((r) => {
    const tr = document.createElement("tr");
    if (r.partition === "primary_test" && r.model === "M4") {
      tr.style.backgroundColor = "rgba(6, 182, 212, 0.06)";
    }
    tr.innerHTML = `
      <td><strong>${r.cutoff} cyc</strong></td>
      <td><span class="code-badge">${r.algorithm}</span></td>
      <td><span class="badge badge-outline">${r.algorithm_family || "--"}</span></td>
      <td><strong>${r.model}</strong></td>
      <td><span class="badge ${r.partition === 'primary_test' ? 'badge-accent' : 'badge-outline'}">${r.partition}</span></td>
      <td>${fmt(r.auroc)}</td>
      <td>${fmt(r.auprc)}</td>
      <td><strong>${fmt(r.brier_score)}</strong></td>
      <td>${fmt(r.sensitivity)}</td>
      <td>${fmt(r.specificity)}</td>
      <td>${fmt(r.f1_score)}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the Multi-Seed Peer Controls Table and PMRI summary.
 */
function renderPeerControlsTable(data) {
  const tbody = document.getElementById("peer-controls-tbody");
  if (!tbody || !data || !data.permutations) return;
  tbody.innerHTML = "";

  const perms = data.permutations;
  const cutoffs = [40, 70, 100];

  cutoffs.forEach((cutoff) => {
    const genuine = perms.find((p) => p.cutoff === cutoff && p.trial_type === "genuine");
    const nulls = perms.filter((p) => p.cutoff === cutoff && p.trial_type === "randomized_null");

    if (!genuine || nulls.length === 0) return;

    const nullAurocs = nulls.map((n) => n.auroc);
    const nullBriers = nulls.map((n) => n.brier_score);

    const meanNullAuc = nullAurocs.reduce((a, b) => a + b, 0) / nullAurocs.length;
    const stdNullAuc = Math.sqrt(nullAurocs.reduce((a, b) => a + Math.pow(b - meanNullAuc, 2), 0) / nullAurocs.length);
    const meanNullBrier = nullBriers.reduce((a, b) => a + b, 0) / nullBriers.length;
    const pVal = nullAurocs.filter((a) => a >= genuine.auroc).length / nullAurocs.length;

    const tr = document.createElement("tr");
    const sigBadge = pVal < 0.05
      ? `<span class="badge badge-success">p = ${pVal.toFixed(4)} (Sig. Non-Random)</span>`
      : `<span class="badge badge-outline">p = ${pVal.toFixed(4)} (Permutation Parity)</span>`;

    tr.innerHTML = `
      <td><strong>Cutoff ${cutoff}</strong></td>
      <td><strong>${genuine.auroc.toFixed(4)}</strong></td>
      <td>${meanNullAuc.toFixed(4)} &plusmn; ${stdNullAuc.toFixed(4)}</td>
      <td>${sigBadge}</td>
      <td><strong>${genuine.brier_score.toFixed(4)}</strong></td>
      <td>${meanNullBrier.toFixed(4)}</td>
      <td>${pVal < 0.05 ? "Peer trajectory signal statistically exceeds randomized pairings" : "Empirical null reaches parity with genuine model"}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the Selective Prediction Risk-Coverage Trade-Off Table.
 */
function renderSelectivePredictionTable(records) {
  const tbody = document.getElementById("selective-pred-tbody");
  if (!tbody || !records) return;
  tbody.innerHTML = "";

  const fmt = (val) => (val !== null && val !== undefined && !isNaN(val) ? (typeof val === "number" ? val.toFixed(4) : val) : "--");

  records.forEach((r) => {
    const tr = document.createElement("tr");
    if (r.target_coverage === 0.95) {
      tr.style.backgroundColor = "rgba(16, 185, 129, 0.08)";
    }
    tr.innerHTML = `
      <td><strong>Cutoff ${r.cutoff}</strong></td>
      <td><span class="code-badge">${r.selection_mechanism}</span></td>
      <td>${(r.target_coverage * 100).toFixed(0)}%</td>
      <td>${fmt(r.training_tau)}</td>
      <td><strong>${(r.realized_test_coverage * 100).toFixed(1)}%</strong> (${r.n_retained_test}/${r.n_total_test})</td>
      <td>${fmt(r.retained_auroc)}</td>
      <td><strong>${fmt(r.retained_brier_score)}</strong></td>
      <td>${r.retained_accuracy ? (r.retained_accuracy * 100).toFixed(1) + "%" : "--"}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Sets up Hardware Cycler CSV Ingestion, Drag-and-Drop, Sample Loaders, and Real-time Auditing.
 */
function setupHardwareValidation() {
  const fileInput = document.getElementById("hw-file-input");
  const dropzone = document.getElementById("hw-upload-dropzone");
  const textarea = document.getElementById("hw-csv-textarea");
  const btnValid = document.getElementById("btn-hw-load-valid");
  const btnCorrupt = document.getElementById("btn-hw-load-corrupt");
  const btnValidate = document.getElementById("btn-hw-execute-validate");

  // Sample CSVs
  const validSampleCsv = [
    "Cycle_Index,Discharge_Capacity_Ah,Internal_Resistance_Ohm,Temperature_Max_C,Charge_Time_Min",
    ...Array.from({ length: 45 }, (_, i) => {
      const cyc = i + 1;
      const q = (1.065 - 0.00015 * cyc).toFixed(4);
      const ir = (0.0152 + 0.00002 * cyc).toFixed(4);
      const temp = (31.2 + 0.05 * Math.sin(cyc)).toFixed(1);
      const tc = (42.5 + 0.02 * cyc).toFixed(1);
      return `${cyc},${q},${ir},${temp},${tc}`;
    }),
  ].join("\n");

  const corruptSampleCsv = [
    "Cycle_Index,Discharge_Capacity_Ah,Internal_Resistance_Ohm,Temperature_Max_C,Charge_Time_Min",
    "1,1.065,0.0152,31.2,42.5",
    "2,1.064,0.0153,31.3,42.5",
    "5,3.250,0.0154,31.4,42.6", // Non-monotonic cycle step (1, 2, 5) & capacity out of bounds (3.25 Ah > 2.5 Ah)
    "3,0.120,-0.0050,95.0,42.8", // Negative IR & temperature runaway (95°C > 80°C)
    "6,1.061,0.0156,31.5,42.7",
  ].join("\n");

  if (btnValid) {
    btnValid.addEventListener("click", () => {
      textarea.value = validSampleCsv;
      executeHardwareValidation(validSampleCsv);
    });
  }

  if (btnCorrupt) {
    btnCorrupt.addEventListener("click", () => {
      textarea.value = corruptSampleCsv;
      executeHardwareValidation(corruptSampleCsv);
    });
  }

  if (btnValidate) {
    btnValidate.addEventListener("click", () => {
      const text = textarea.value.trim();
      if (!text) {
        alert("Please paste or upload CSV telemetry data to audit.");
        return;
      }
      executeHardwareValidation(text);
    });
  }

  // File dropzone events
  if (dropzone && fileInput) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        readFileContent(file);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        readFileContent(e.target.files[0]);
      }
    });
  }

  function readFileContent(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      textarea.value = evt.target.result;
      executeHardwareValidation(evt.target.result);
    };
    reader.readAsText(file);
  }
}

/**
 * Executes hardware CSV validation via backend POST endpoint.
 */
async function executeHardwareValidation(csvContent) {
  const resultWrap = document.getElementById("hw-audit-result-wrap");
  const verdictBadge = document.getElementById("hw-audit-verdict-badge");
  const summaryText = document.getElementById("hw-audit-summary-text");
  const errorsWrap = document.getElementById("hw-audit-errors-wrap");
  const errorsList = document.getElementById("hw-audit-errors-list");
  const previewWrap = document.getElementById("hw-audit-preview-wrap");
  const sampleTbody = document.getElementById("hw-audit-sample-tbody");

  try {
    const res = await fetch("/api/hardware/validate", {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: csvContent,
    });

    const data = await res.json();
    resultWrap.style.display = "block";

    if (data.is_valid || data.valid) {
      verdictBadge.innerHTML = `<span class="badge badge-success">VALIDATED FOR INGESTION</span>`;
      summaryText.textContent = `✓ ${data.message} Schema detected: ${data.schema_detected}. Ingestion buffer ready.`;
      errorsWrap.style.display = "none";
    } else {
      verdictBadge.innerHTML = `<span class="badge badge-danger">REJECTED — INTEGRITY ERRORS</span>`;
      summaryText.textContent = `⚠️ ${data.message}`;

      // Populate error list
      errorsWrap.style.display = "block";
      errorsList.innerHTML = "";
      (data.errors || data.anomalies || []).forEach((err) => {
        const li = document.createElement("li");
        li.textContent = err;
        errorsList.appendChild(li);
      });
    }

    // Populate sample rows
    if (data.sample_rows && data.sample_rows.length > 0) {
      previewWrap.style.display = "block";
      sampleTbody.innerHTML = "";
      data.sample_rows.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${r.cycle}</strong></td>
          <td>${r.capacity_ah !== undefined ? r.capacity_ah.toFixed(4) : (r.voltage !== undefined ? r.voltage.toFixed(3) + "V" : "--")}</td>
          <td>${r.ir_ohm !== undefined ? r.ir_ohm.toFixed(4) : (r.current !== undefined ? r.current.toFixed(2) + "A" : "--")}</td>
          <td>${r.temperature_c !== undefined ? r.temperature_c.toFixed(1) : (r.temperature !== undefined ? r.temperature.toFixed(1) : "--")}</td>
          <td>${r.charge_time_min !== undefined ? r.charge_time_min.toFixed(1) : "--"}</td>
        `;
        sampleTbody.appendChild(tr);
      });
    } else {
      previewWrap.style.display = "none";
    }

  } catch (err) {
    console.error("Hardware validation error:", err);
    alert("Could not complete hardware telemetry validation. Ensure server is active.");
  }
}

/**
 * Updates the multi-cell comparison chart.
 */
async function updateComparisonChart() {
  const ctx = document.getElementById("comparison-chart").getContext("2d");
  const metric = state.comparisonMetric;
  const metaInfo = METRIC_META[metric];
  renderCompareChips();

  try {
    const promises = state.comparisonCells.map(async (cid) => {
      const res = await fetch(`/api/battery?id=${cid}&cutoff=${state.currentCutoff}`);
      if (!res.ok) return null;
      return res.json();
    });

    const batteries = (await Promise.all(promises)).filter(Boolean);

    if (state.comparisonChart) {
      state.comparisonChart.destroy();
    }

    const datasets = batteries.map((b, idx) => {
      const color = PALETTE[idx % PALETTE.length];
      return {
        label: `${b.metadata.cell_id} (${b.metadata.policy})`,
        data: b.cycles.map((c) => ({ x: c.cycle, y: c[metric] })).filter((p) => p.y !== null && p.y !== undefined),
        borderColor: color,
        backgroundColor: "transparent",
        borderWidth: b.metadata.cell_id === state.currentBatteryId ? 3 : 1.5,
        borderDash: b.metadata.cell_id === state.currentBatteryId ? [] : [3, 3],
        pointRadius: 0,
        tension: 0.1,
      };
    });

    state.comparisonChart = new Chart(ctx, {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: {
            display: true,
            labels: { color: "#94a3b8", font: { size: 11 }, boxWidth: 12 },
          },
          tooltip: {
            backgroundColor: "#162035",
            borderColor: "#23314d",
            borderWidth: 1,
            callbacks: {
              title: (items) => `Cycle ${items[0].parsed.x}`,
              label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)} ${metaInfo.unit}`,
            },
          },
        },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "Cycle Number", color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" },
          },
          y: {
            title: { display: true, text: `${metaInfo.label} (${metaInfo.unit})`, color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" },
          },
        },
      },
    });

  } catch (err) {
    console.error("Error updating comparison chart:", err);
  }
}

/**
 * Renders chips for currently selected comparison cells.
 */
function renderCompareChips() {
  const container = document.getElementById("compare-pills-list");
  if (!container) return;
  container.innerHTML = "";

  state.comparisonCells.forEach((cid) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `
      <span>${cid}</span>
      <span class="chip-remove" data-id="${cid}" title="Remove from comparison">&times;</span>
    `;
    container.appendChild(chip);
  });

  container.querySelectorAll(".chip-remove").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const idToRemove = e.target.dataset.id;
      state.comparisonCells = state.comparisonCells.filter((id) => id !== idToRemove);
      if (state.comparisonCells.length === 0) {
        state.comparisonCells = [state.currentBatteryId];
      }
      updateComparisonChart();
      populateCompareDropdown();
    });
  });
}

/**
 * Renders the engineered features table.
 */
function renderFeaturesTable() {
  if (!state.currentBatteryData) return;
  const tbody = document.getElementById("features-table-body");
  tbody.innerHTML = "";

  const grpName = state.currentFeatureGroup;
  const featureList = state.currentBatteryData.features_grouped[grpName] || [];

  if (featureList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No features available for this group at the selected cutoff.</td></tr>`;
    return;
  }

  featureList.forEach((f) => {
    const tr = document.createElement("tr");

    let formattedVal = "--";
    if (f.value !== null && f.value !== undefined) {
      if (typeof f.value === "number") {
        if (Math.abs(f.value) < 0.001 && f.value !== 0) {
          formattedVal = f.value.toExponential(4);
        } else {
          formattedVal = f.value.toFixed(4);
        }
      } else {
        formattedVal = String(f.value);
      }
    }

    tr.innerHTML = `
      <td><strong>${f.name}</strong></td>
      <td><span class="code-badge">${f.column}</span></td>
      <td class="feat-val">${formattedVal}</td>
      <td class="text-muted">${f.unit}</td>
      <td class="text-muted">${f.description}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the ground-truth benchmark targets audit box.
 */
function renderGroundTruthTargets(tInfo, meta) {
  const eolEl = document.getElementById("val-target-eol");
  const rulEl = document.getElementById("val-target-rul");
  const f500El = document.getElementById("val-target-500");
  const f700El = document.getElementById("val-target-700");

  if (!meta.is_benchmark_eligible) {
    eolEl.textContent = "Censored (Excluded)";
    eolEl.className = "t-val text-amber";
    rulEl.textContent = "Censored";
    f500El.textContent = "Censored";
    f700El.textContent = "Censored";
    return;
  }

  eolEl.textContent = tInfo.target_eol_cycle ? `${tInfo.target_eol_cycle} cycles` : "--";
  eolEl.className = "t-val text-teal";

  rulEl.textContent = tInfo.target_rul ? `${tInfo.target_rul} cycles` : "--";

  if (tInfo.target_fail_within_500 === 1) {
    f500El.textContent = "Positive (Fails within 500 cyc)";
    f500El.className = "t-val text-rose";
  } else if (tInfo.target_fail_within_500 === 0) {
    f500El.textContent = "Negative (Survives >500 cyc)";
    f500El.className = "t-val text-emerald";
  } else {
    f500El.textContent = "--";
  }

  if (tInfo.target_fail_by_700 === 1) {
    f700El.textContent = "Positive (≤ 700 cyc)";
    f700El.className = "t-val text-rose";
  } else if (tInfo.target_fail_by_700 === 0) {
    f700El.textContent = "Negative (> 700 cyc)";
    f700El.className = "t-val text-emerald";
  } else {
    f700El.textContent = "--";
  }
}

/**
 * Renders data-grounded explainability observations.
 */
function renderFactualObservations(obsList) {
  const container = document.getElementById("observations-container");
  container.innerHTML = "";

  if (!obsList || obsList.length === 0) {
    container.innerHTML = `<p class="text-muted">No specific trajectory anomalies detected for this cell window.</p>`;
    return;
  }

  obsList.forEach((obs) => {
    const item = document.createElement("div");
    item.className = "obs-item";
    item.innerHTML = `
      <span class="obs-bullet">✦</span>
      <p>${obs}</p>
    `;
    container.appendChild(item);
  });
}
