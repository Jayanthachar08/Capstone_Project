/**
 * Battery Degradation Screening Dashboard Client Application
 * Phase 7: Validated ML Models, Peer-Relative Trajectory Overlay, PMRI, and Research Integration.
 */

// Application State
const state = {
  currentBatteryId: "b1c0",
  currentCutoff: 100, // 40, 70, 100, or 'all'
  currentModel: "M4", // 'M1', 'M2', 'M3', 'M4'
  currentMetric: "QDischarge", // 'QDischarge', 'IR', 'Tmax', 'Tavg', 'chargetime'
  currentFeatureGroup: "Initial Condition",
  comparisonCells: ["b1c0", "b1c1"],
  comparisonMetric: "QDischarge",
  batteriesList: [],
  currentBatteryData: null,
  overviewData: null,
  researchSummary: null,
  researchMetrics: null,
  researchDifferences: null,
  trajectoryChart: null,
  comparisonChart: null,
  peerTrajectoryChart: null,
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
      if (mCountEl) mCountEl.textContent = state.overviewData.total_models || 4;
      const pCountEl = document.getElementById("stat-predictions-count");
      if (pCountEl) pCountEl.textContent = state.overviewData.total_predictions || 372;
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

    // 5. Load research section data asynchronously
    loadAndRenderResearchSection();

    // 6. Load initial default battery
    await loadBattery(state.currentBatteryId);
    await updateComparisonChart();

  } catch (err) {
    console.error("Dashboard initialization error:", err);
    alert("Could not load dashboard data. Ensure dashboard/server.py is running.");
  }
}

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
  document.getElementById("compare-cell-select").addEventListener("change", (e) => {
    const cid = e.target.value;
    if (cid && !state.comparisonCells.includes(cid)) {
      state.comparisonCells.push(cid);
      updateComparisonChart();
      populateCompareDropdown();
    }
    e.target.value = "";
  });

  // Quick Cohort: Same Policy
  document.getElementById("btn-compare-same-policy").addEventListener("click", () => {
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

  // Quick Cohort: Across Batches
  document.getElementById("btn-compare-batch-samples").addEventListener("click", () => {
    state.comparisonCells = ["b1c0", "b2c0", "b3c0"];
    updateComparisonChart();
    populateCompareDropdown();
  });

  // Reset Comparison
  document.getElementById("btn-clear-compare").addEventListener("click", () => {
    state.comparisonCells = [state.currentBatteryId];
    updateComparisonChart();
    populateCompareDropdown();
  });

  // Research Tabs Switcher
  const researchTabs = document.querySelectorAll("#research-tabs-bar .research-tab-btn");
  researchTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      researchTabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      document.querySelectorAll(".research-panel").forEach((panel) => {
        panel.style.display = panel.id === targetId ? "block" : "none";
      });
    });
  });
}

/**
 * Fetches and renders details for the selected battery.
 */
async function loadBattery(cellId) {
  try {
    const cutoffParam = state.currentCutoff;
    const res = await fetch(`/api/battery/${cellId}?cutoff=${cutoffParam}`);
    if (!res.ok) throw new Error(`Failed to load data for ${cellId}`);

    const data = await res.json();
    state.currentBatteryData = data;

    // Update all UI panels
    renderMetadataStrip(data.metadata);
    renderSummaryCards(data.summary_cards, data.metadata);
    renderTrajectoryChart();
    renderFeaturesTable();
    renderGroundTruthTargets(data.target_info, data.metadata);
    renderFactualObservations(data.factual_observations);

    // Phase 7 Enhancements
    renderRiskPredictions(data.model_predictions, data.metadata, cutoffParam);
    renderPeerRelativeSection(cellId, cutoffParam);
    renderPMRIGauge(data.peer_info);
    updateElevatedRiskTally();

  } catch (err) {
    console.error("Error loading battery details:", err);
  }
}

/**
 * Updates the metadata preview strip in the controls card.
 */
function renderMetadataStrip(meta) {
  document.getElementById("meta-batch").textContent = meta.batch || "--";
  document.getElementById("meta-partition").textContent = meta.benchmark_partition || "--";
  document.getElementById("meta-policy").textContent = meta.policy || "--";
  updateElevatedRiskTally();
}

/**
 * Updates the elevated risk tally in the metadata strip based on active cutoff and model.
 */
function updateElevatedRiskTally() {
  const tallyEl = document.getElementById("meta-risk-tally");
  if (!tallyEl || !state.overviewData) return;

  const activeCutoff = state.currentCutoff === "all" ? 100 : state.currentCutoff;
  const activeModel = state.currentModel;
  const counts = state.overviewData.elevated_risk_counts || {};

  if (counts[activeCutoff] && counts[activeCutoff][activeModel] !== undefined) {
    const c = counts[activeCutoff][activeModel];
    tallyEl.textContent = `${c} / 124 cells elevated (Cutoff ${activeCutoff}, ${activeModel})`;
  } else {
    tallyEl.textContent = "--";
  }
}

/**
 * Populates the top summary metric cards.
 */
function renderSummaryCards(cards, meta) {
  document.getElementById("card-cell-id").textContent = cards.cell_id;
  const statusEl = document.getElementById("card-cell-status");
  if (meta.is_benchmark_eligible) {
    statusEl.textContent = `Benchmark Eligible (${meta.benchmark_partition})`;
    statusEl.className = "metric-sub text-teal";
  } else {
    statusEl.textContent = `Excluded: ${meta.exclusion_reason || "Non-conforming"}`;
    statusEl.className = "metric-sub text-amber";
  }

  document.getElementById("card-observed-cycles").textContent = cards.observed_cycles_in_view;
  document.getElementById("card-total-cycles").textContent = `of ${cards.total_life_cycles || cards.total_cycles_recorded} total life`;

  document.getElementById("card-latest-q").textContent =
    cards.q_latest_ah !== null ? cards.q_latest_ah.toFixed(4) : "--";
  document.getElementById("card-initial-q").textContent =
    cards.q_start_ah !== null ? `Baseline (Cyc 2): ${cards.q_start_ah.toFixed(4)} Ah` : "Baseline: --";

  const retEl = document.getElementById("card-retention");
  if (cards.retention_pct !== null) {
    retEl.textContent = `${cards.retention_pct.toFixed(2)}%`;
    retEl.className = cards.retention_pct >= 95 ? "metric-value text-emerald" : "metric-value text-amber";
  } else {
    retEl.textContent = "--%";
  }

  const fadeEl = document.getElementById("card-fade-delta");
  if (cards.q_change_ah !== null) {
    fadeEl.textContent = `Fade ΔQ: ${cards.q_change_ah > 0 ? "+" : ""}${cards.q_change_ah.toFixed(4)} Ah`;
  } else {
    fadeEl.textContent = "Fade ΔQ: -- Ah";
  }

  document.getElementById("card-latest-ir").textContent =
    cards.ir_latest_ohm !== null ? cards.ir_latest_ohm.toFixed(5) : "--";
  document.getElementById("card-ir-growth").textContent =
    cards.ir_change_pct !== null ? `Growth: ${cards.ir_change_pct > 0 ? "+" : ""}${cards.ir_change_pct.toFixed(2)}%` : "--";

  document.getElementById("card-tmax").textContent =
    cards.tmax_latest_c !== null ? cards.tmax_latest_c.toFixed(1) : "--";
  document.getElementById("card-chargetime").textContent =
    cards.ct_latest_min !== null ? `Charge: ${cards.ct_latest_min.toFixed(1)} min` : "--";
}

/**
 * Renders the primary single-battery trajectory chart using Chart.js.
 */
function renderTrajectoryChart() {
  if (!state.currentBatteryData) return;
  const ctx = document.getElementById("trajectory-chart").getContext("2d");
  const cycles = state.currentBatteryData.cycles || [];
  const metric = state.currentMetric;
  const metaInfo = METRIC_META[metric] || METRIC_META.QDischarge;

  const points = cycles
    .filter((c) => (metric === "QDischarge" ? c.cycle >= 2 : c[metric] > 0))
    .map((c) => ({ x: c.cycle, y: c[metric] }));

  if (state.trajectoryChart) {
    state.trajectoryChart.destroy();
  }

  const isCapacity = metric === "QDischarge";
  document.getElementById("legend-nominal").style.display = isCapacity ? "inline-flex" : "none";
  document.getElementById("legend-eol").style.display = isCapacity ? "inline-flex" : "none";

  const datasets = [
    {
      label: `${state.currentBatteryId} — ${metaInfo.label} (${metaInfo.unit})`,
      data: points,
      borderColor: metaInfo.color,
      backgroundColor: "rgba(56, 189, 248, 0.08)",
      borderWidth: 2.2,
      pointRadius: points.length <= 120 ? 3 : 0,
      pointHoverRadius: 5,
      tension: 0.1,
      fill: isCapacity,
    },
  ];

  if (isCapacity && points.length > 0) {
    const minX = points[0].x;
    const maxX = points[points.length - 1].x;

    datasets.push({
      label: "Nominal Capacity (1.10 Ah)",
      data: [{ x: minX, y: 1.1 }, { x: maxX, y: 1.1 }],
      borderColor: "#10b981",
      borderWidth: 1.5,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    });

    datasets.push({
      label: "80% EOL Threshold (0.88 Ah)",
      data: [{ x: minX, y: 0.88 }, { x: maxX, y: 0.88 }],
      borderColor: "#ef4444",
      borderWidth: 1.5,
      borderDash: [5, 5],
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
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#f8fafc",
          borderColor: "#334155",
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
 * Renders the ML EOL Risk Prediction card and four-model comparison table.
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

  // Populate Four-Model Comparative Table
  tbody.innerHTML = "";
  const modelOrder = ["M1", "M2", "M3", "M4"];
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
      ? `<span class="badge badge-rose" style="font-size:0.75rem;">Elevated Risk</span>`
      : `<span class="badge badge-emerald" style="font-size:0.75rem;">Nominal</span>`;

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
        col: "relative_capacity_deviation",
        val: peerData.relative_capacity_deviation,
        desc: "Discharge capacity difference between target cell and peer centroid at cutoff (Ah).",
      },
      {
        name: "Relative Slope Deviation (ΔSlope_rel)",
        col: "relative_slope_deviation",
        val: peerData.relative_slope_deviation,
        desc: "Trajectory capacity fade slope difference relative to matched peers.",
      },
      {
        name: "Relative Fade Deviation (ΔFade_rel)",
        col: "relative_fade_deviation",
        val: peerData.relative_fade_deviation,
        desc: "Cumulative capacity degradation divergence relative to peer group mean.",
      },
      {
        name: "Relative Internal Resistance Deviation",
        col: "relative_ir_deviation",
        val: peerData.relative_ir_deviation,
        desc: "SEI growth impedance divergence relative to peer reference group (Ω).",
      },
    ];

    relMetrics.forEach((m) => {
      const tr = document.createElement("tr");
      const valStr = m.val !== null && m.val !== undefined ? m.val.toFixed(5) : "--";
      tr.innerHTML = `
        <td><strong>${m.name}</strong></td>
        <td><span class="code-badge">${m.col}</span></td>
        <td class="text-cyan"><strong>${valStr}</strong></td>
        <td class="text-muted" style="font-size:0.78rem;">${m.desc}</td>
      `;
      metricsTbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Error rendering peer-relative trajectories:", err);
  }
}

/**
 * Draws the multi-line peer overlay chart using Chart.js.
 */
function renderPeerOverlayChart(data) {
  const ctx = document.getElementById("peer-trajectory-chart").getContext("2d");

  if (state.peerTrajectoryChart) {
    state.peerTrajectoryChart.destroy();
  }

  const datasets = [];

  // 1. Target Cell Dataset
  datasets.push({
    label: `Target Battery (${data.cell_id})`,
    data: data.target_trajectory || [],
    borderColor: "#38bdf8",
    backgroundColor: "rgba(56, 189, 248, 0.12)",
    borderWidth: 2.8,
    pointRadius: 0,
    pointHoverRadius: 4,
    tension: 0.1,
  });

  // 2. Individual Peer Lines (Muted)
  (data.peers || []).forEach((p, idx) => {
    datasets.push({
      label: `Peer ${idx + 1} (${p.cell_id}, d=${p.distance.toFixed(3)})`,
      data: p.points || [],
      borderColor: PALETTE[idx + 1] || "#64748b",
      borderWidth: 1.4,
      borderDash: [3, 3],
      pointRadius: 0,
      fill: false,
    });
  });

  // 3. Aggregate Peer Mean Trajectory
  if (data.mean_peer_trajectory && data.mean_peer_trajectory.length > 0) {
    datasets.push({
      label: "Matched Peer Centroid (Mean Reference)",
      data: data.mean_peer_trajectory,
      borderColor: "#fbbf24",
      borderWidth: 2.2,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
    });
  }

  state.peerTrajectoryChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: "#94a3b8", font: { size: 10 } },
        },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#f8fafc",
          callbacks: {
            title: (items) => `Cycle ${items[0].parsed.x}`,
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)} Ah`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Cycle Number (Observed Window)", color: "#94a3b8" },
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
 * Renders the PMRI gauge card, composite score, and component breakdown bars.
 */
function renderPMRIGauge(peerInfo) {
  const compValEl = document.getElementById("pmri-composite-val");
  const meanDistEl = document.getElementById("pmri-mean-dist");
  const dispEl = document.getElementById("pmri-dispersion");
  const warnBadgeEl = document.getElementById("pmri-warning-badge");

  if (!peerInfo || peerInfo.pmri === null) {
    compValEl.textContent = "--";
    meanDistEl.textContent = "--";
    dispEl.textContent = "--";
    warnBadgeEl.className = "badge badge-outline";
    warnBadgeEl.textContent = "No Peer Data";
    return;
  }

  compValEl.textContent = peerInfo.pmri.toFixed(4);
  meanDistEl.textContent = peerInfo.mean_peer_distance ? peerInfo.mean_peer_distance.toFixed(4) : "--";
  dispEl.textContent = peerInfo.peer_trajectory_dispersion ? `${peerInfo.peer_trajectory_dispersion.toFixed(5)} Ah` : "--";

  if (peerInfo.reliability_warning === "NONE" || !peerInfo.reliability_warning) {
    warnBadgeEl.className = "badge badge-success";
    warnBadgeEl.textContent = "✓ High Confidence Match";
  } else {
    warnBadgeEl.className = "badge badge-warning";
    warnBadgeEl.textContent = `⚠️ Warning: ${peerInfo.reliability_warning}`;
  }

  // Component bars
  const setBar = (fillId, valId, score) => {
    const fillEl = document.getElementById(fillId);
    const valEl = document.getElementById(valId);
    if (!fillEl || !valEl) return;
    if (score !== null && score !== undefined) {
      const pct = Math.max(0, Math.min(100, score * 100));
      fillEl.style.width = `${pct}%`;
      valEl.textContent = score.toFixed(4);
    } else {
      fillEl.style.width = "0%";
      valEl.textContent = "--";
    }
  };

  setBar("bar-s-sim", "val-s-sim", peerInfo.peer_similarity_score);
  setBar("bar-s-cov", "val-s-cov", peerInfo.peer_availability_score);
  setBar("bar-s-agree", "val-s-agree", peerInfo.peer_agreement_score);
  setBar("bar-s-consist", "val-s-consist", peerInfo.peer_feature_completeness);
}

/**
 * Loads and renders the Research Model-Comparison Results & Hypothesis Findings section.
 */
async function loadAndRenderResearchSection() {
  try {
    // 1. Fetch performance metrics
    const resMetrics = await fetch("/api/models/metrics");
    if (resMetrics.ok) {
      state.researchMetrics = await resMetrics.json();
      renderResearchMetricsTable(state.researchMetrics);
    }

    // 2. Fetch hypothesis summary
    const resSummary = await fetch("/api/research/summary");
    if (resSummary.ok) {
      state.researchSummary = await resSummary.json();
      renderHypothesisVerdicts(state.researchSummary);
    }

    // 3. Fetch paired differences
    const resDiff = await fetch("/api/research/differences");
    if (resDiff.ok) {
      state.researchDifferences = await resDiff.json();
      renderResearchDifferencesTable(state.researchDifferences);
    }
  } catch (err) {
    console.error("Error loading research section data:", err);
  }
}

/**
 * Renders the Phase 5/6 model performance benchmark table.
 */
function renderResearchMetricsTable(metrics) {
  const tbody = document.getElementById("research-metrics-tbody");
  if (!tbody || !metrics) return;
  tbody.innerHTML = "";

  metrics.forEach((row) => {
    const tr = document.createElement("tr");
    const auc = row.auroc !== null && row.auroc !== undefined ? row.auroc.toFixed(4) : "NaN";
    const prc = row.auprc !== null && row.auprc !== undefined ? row.auprc.toFixed(4) : "NaN";
    const brier = row.brier_score !== null && row.brier_score !== undefined ? row.brier_score.toFixed(4) : "--";
    const ece = row.ece !== null && row.ece !== undefined ? row.ece.toFixed(4) : "--";
    const acc = row.accuracy !== null && row.accuracy !== undefined ? (row.accuracy * 100).toFixed(1) + "%" : "--";
    const sens = row.sensitivity !== null && row.sensitivity !== undefined ? (row.sensitivity * 100).toFixed(1) + "%" : "--";
    const spec = row.specificity !== null && row.specificity !== undefined ? (row.specificity * 100).toFixed(1) + "%" : "--";
    const f1 = row.f1_score !== null && row.f1_score !== undefined ? row.f1_score.toFixed(3) : "--";

    tr.innerHTML = `
      <td><strong>Cycle ${row.cutoff}</strong></td>
      <td><span class="code-badge">${row.model}</span></td>
      <td>${row.partition}</td>
      <td class="text-muted">${row.n_samples} (${row.positive_count}/${row.negative_count})</td>
      <td class="text-cyan"><strong>${auc}</strong></td>
      <td>${prc}</td>
      <td>${brier}</td>
      <td>${ece}</td>
      <td>${acc}</td>
      <td>${sens}</td>
      <td>${spec}</td>
      <td>${f1}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the formal hypothesis evaluation verdicts (H1–H4).
 */
function renderHypothesisVerdicts(summary) {
  const grid = document.getElementById("hypothesis-cards-grid");
  if (!grid || !summary) return;
  grid.innerHTML = "";

  const hypDefs = [
    { key: "H1_incremental_peer_relative_value", title: "H1: Incremental Peer-Relative Value (M2 vs M3)" },
    { key: "H2_peer_match_reliability_value", title: "H2: Value of Peer-Match Reliability (M3 vs M4)" },
    { key: "H3_early_observation_usefulness", title: "H3: Early-Observation Usefulness (Cutoffs 40/70/100)" },
    { key: "H4_reliability_under_weak_matching", title: "H4: Robustness & Negative Control (Scrambled Peers)" },
  ];

  hypDefs.forEach(({ key, title }) => {
    const data = summary[key];
    if (!data) return;

    const card = document.createElement("div");
    card.className = "hypothesis-card";

    let verdictBadge = "";
    let rationale = "";

    if (key === "H1_incremental_peer_relative_value") {
      const inDist = data.in_distribution_primary_test || {};
      const ood = data.out_of_distribution_secondary_test || {};
      verdictBadge = `<span class="hyp-verdict hyp-verdict-parity">In-Dist: Parity | OOD: Supported</span>`;
      rationale = `<strong>Primary Test:</strong> ${inDist.rationale || ""}<br><br><strong>Secondary Test (Batch 3):</strong> ${ood.rationale || ""}`;
    } else if (key === "H2_peer_match_reliability_value") {
      verdictBadge = `<span class="hyp-verdict hyp-verdict-supported">Feature: Parity | Gate: Supported</span>`;
      rationale = `<strong>Additive Feature:</strong> ${(data.as_additive_feature || {}).rationale || ""}<br><br><strong>Meta-Reliability Filter:</strong> ${(data.as_confidence_gate || {}).rationale || ""}`;
    } else {
      verdictBadge = `<span class="hyp-verdict hyp-verdict-supported">Supported</span>`;
      rationale = data.rationale || "";
    }

    card.innerHTML = `
      <div class="hyp-header">
        <span class="hyp-title">${title}</span>
        ${verdictBadge}
      </div>
      <p class="hyp-statement">“${data.hypothesis_statement || ""}”</p>
      <div class="hyp-rationale">${rationale}</div>
    `;
    grid.appendChild(card);
  });
}

/**
 * Renders the paired model differences table with 95% bootstrap CIs.
 */
function renderResearchDifferencesTable(diffs) {
  const tbody = document.getElementById("research-diff-tbody");
  if (!tbody || !diffs) return;
  tbody.innerHTML = "";

  diffs.forEach((row) => {
    const tr = document.createElement("tr");

    const formatCI = (pt, l, u) => {
      if (pt === null || pt === undefined || isNaN(pt)) return "NaN";
      return `<strong>${pt > 0 ? "+" : ""}${pt.toFixed(4)}</strong> [${l.toFixed(4)}, ${u.toFixed(4)}]`;
    };

    const aucStr = formatCI(row.delta_auroc, row.delta_auroc_ci_lower, row.delta_auroc_ci_upper);
    const prcStr = formatCI(row.delta_auprc, row.delta_auprc_ci_lower, row.delta_auprc_ci_upper);
    const brierStr = formatCI(row.delta_brier, row.delta_brier_ci_lower, row.delta_brier_ci_upper);

    const sigBadge = row.delta_auroc_significant || row.delta_brier_significant
      ? `<span class="badge badge-rose" style="font-size:0.75rem;">Significant</span>`
      : `<span class="badge badge-outline" style="font-size:0.75rem;">Parity (CI spans 0)</span>`;

    tr.innerHTML = `
      <td><strong>Cycle ${row.cutoff}</strong></td>
      <td>${row.partition}</td>
      <td><span class="code-badge">${row.comparison}</span> (${row.description})</td>
      <td>${aucStr}</td>
      <td>${prcStr}</td>
      <td>${brierStr}</td>
      <td>${sigBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Updates the multi-battery exploratory comparison chart.
 */
async function updateComparisonChart() {
  if (state.comparisonCells.length === 0) return;
  const ctx = document.getElementById("comparison-chart").getContext("2d");

  renderCompareChips();

  const idsParam = state.comparisonCells.join(",");
  const cutoffParam = state.currentCutoff;
  const metric = state.comparisonMetric;

  try {
    const res = await fetch(`/api/compare?ids=${idsParam}&metric=${metric}&cutoff=${cutoffParam}`);
    if (!res.ok) throw new Error("Failed to load comparison data.");
    const data = await res.json();

    if (state.comparisonChart) {
      state.comparisonChart.destroy();
    }

    const datasets = data.series.map((s, idx) => ({
      label: `${s.cell_id} (${s.batch}, EOL: ${s.benchmark_cycle_life || "?"} cyc)`,
      data: s.points,
      borderColor: PALETTE[idx % PALETTE.length],
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.1,
    }));

    state.comparisonChart = new Chart(ctx, {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { position: "top", labels: { color: "#94a3b8", font: { size: 11 } } },
          tooltip: {
            backgroundColor: "#0f172a",
            titleColor: "#38bdf8",
            bodyColor: "#f8fafc",
            callbacks: {
              title: (items) => `Cycle ${items[0].parsed.x}`,
              label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)} Ah`,
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
            title: { display: true, text: "Discharge Capacity (Ah)", color: "#94a3b8" },
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
