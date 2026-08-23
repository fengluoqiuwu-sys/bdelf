import { fetchJSON, sendJSON, paintInstanceBadge } from "./common.js";

const PREFS_KEY = "bdelf_export_prefs";

function readLocalPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) return JSON.parse(raw);
  } catch (_) {}
  return { invert: false, width: 960, height: 360 };
}

function writeLocalPrefs(prefs) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch (_) {}
}

async function loadPrefs() {
  try {
    const data = await fetchJSON("/api/charts-prefs");
    if (data && typeof data === "object") return data;
  } catch (_) {}
  return readLocalPrefs();
}

function savePrefs(prefs) {
  writeLocalPrefs(prefs);
  clearTimeout(window.__prefsTimer);
  window.__prefsTimer = setTimeout(() => {
    sendJSON("/api/charts-prefs", "PUT", prefs).catch(() => {});
  }, 400);
}

function themeOf(invert) {
  if (invert) {
    return {
      bg: "#ffffff",
      tick: "#1a1d27",
      grid: "rgba(0, 0, 0, 0.22)",
      line: "#1a1d27",
    };
  }
  return {
    bg: "#0f1117",
    tick: "#f4f6fb",
    grid: "rgba(255, 255, 255, 0.45)",
    line: "#ffffff",
  };
}

const bgPlugin = {
  id: "exportBg",
  beforeDraw(chart) {
    const color = chart.options.plugins?.exportBg?.color || "#0f1117";
    const { ctx } = chart;
    ctx.save();
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, chart.width, chart.height);
    ctx.restore();
  },
};

const refLinePlugin = {
  id: "refLines",
  afterDatasetsDraw(chart) {
    const items = chart.options.plugins?.refLines?.items || [];
    if (!items.length) return;
    const { ctx } = chart;
    const area = chart.chartArea;
    const fallback = chart.options.color || "#888";
    const xScale = chart.scales.x;
    for (const item of items) {
      const color = item.color || fallback;
      if (item.x != null && Number.isFinite(Number(item.x))) {
        if (!xScale) continue;
        const x = xScale.getPixelForValue(Number(item.x));
        if (x < area.left || x > area.right) continue;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.moveTo(x, area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
        const text = item.note || "";
        if (text) {
          ctx.setLineDash([]);
          ctx.font = "600 11px Segoe UI, system-ui, sans-serif";
          ctx.fillStyle = color;
          const rightSide = x > (area.left + area.right) / 2;
          ctx.textAlign = rightSide ? "right" : "left";
          ctx.textBaseline = "top";
          ctx.fillText(text, x + (rightSide ? -5 : 5), area.top + 4);
        }
        ctx.restore();
        continue;
      }
      const yScale = chart.scales[item.yAxisID];
      if (!yScale) continue;
      const y = yScale.getPixelForValue(item.y);
      if (y < area.top || y > area.bottom) continue;
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.moveTo(area.left, y);
      ctx.lineTo(area.right, y);
      ctx.stroke();
      const text = item.note || "";
      if (text) {
        ctx.setLineDash([]);
        ctx.font = "600 11px Segoe UI, system-ui, sans-serif";
        ctx.fillStyle = color;
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText(text, area.right - 2, y - 3);
      }
      ctx.restore();
    }
  },
};

if (window.Chart) {
  window.Chart.register(bgPlugin, refLinePlugin);
}

function parseBound(raw) {
  if (raw === "" || raw == null) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : undefined;
}

function yScalesForTraces(traces, theme) {
  const scales = {};
  (traces || []).forEach((t, i) => {
    const color = t.color || "#5b9fff";
    const scale = {
      position: i === 0 ? "left" : "right",
      grid: { color: theme.grid, lineWidth: 1, drawOnChartArea: i === 0 },
      border: { color: theme.line },
      title: { display: true, text: t.metric || "value", color, font: { size: 13, weight: "600" } },
      ticks: { color, font: { size: 12, weight: "600" } },
    };
    const min = parseBound(t.y_min);
    const max = parseBound(t.y_max);
    if (min !== undefined) scale.min = min;
    if (max !== undefined) scale.max = max;
    scales[`y${i}`] = scale;
  });
  return scales;
}

function buildOptions(spec, invert) {
  const theme = themeOf(invert);
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    devicePixelRatio: 2,
    color: theme.tick,
    interaction: { mode: "nearest", intersect: false },
    scales: {
      x: {
        type: "linear",
        ticks: { color: theme.tick, font: { size: 12, weight: "600" } },
        grid: { color: theme.grid, lineWidth: 1 },
        border: { color: theme.line },
        title: { display: true, text: "tokens (B)", color: theme.tick, font: { size: 13, weight: "600" } },
      },
      ...yScalesForTraces(spec.traces, theme),
    },
    plugins: {
      legend: { labels: { color: theme.tick, font: { size: 12, weight: "600" } } },
      refLines: { items: spec.refItems || [] },
      exportBg: { color: theme.bg },
    },
  };
}

function safeName(title) {
  const raw = String(title || "chart").replace(/[\\/:*?"<>|]+/g, "-").trim();
  return (raw || "chart").slice(0, 80);
}

let chart = null;
let spec = null;

function clampInt(raw, fallback, lo, hi) {
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(hi, Math.max(lo, n));
}

function currentInvert() {
  return document.getElementById("export-invert").checked;
}

function currentWidth() {
  return clampInt(document.getElementById("export-width")?.value, 960, 240, 4096);
}

function currentHeight() {
  return clampInt(document.getElementById("export-height")?.value, 360, 120, 2400);
}

function currentPrefs() {
  return { invert: currentInvert(), width: currentWidth(), height: currentHeight() };
}

function fitPreview() {
  const preview = document.getElementById("export-preview");
  const slot = document.getElementById("export-slot");
  const stage = document.getElementById("export-stage");
  if (!preview || !slot || !stage) return;
  const w = currentWidth();
  const h = currentHeight();
  const availW = Math.max(1, stage.clientWidth);
  const availH = Math.max(1, stage.clientHeight);
  const scale = Math.min(1, availW / w, availH / h);
  const slotW = Math.max(1, Math.round(w * scale));
  const slotH = Math.max(1, Math.round(h * scale));
  slot.style.width = `${slotW}px`;
  slot.style.height = `${slotH}px`;
  preview.style.width = `${w}px`;
  preview.style.height = `${h}px`;
  preview.style.transform = `scale(${scale})`;
}

function applyPreviewChrome(persist = true) {
  const prefs = currentPrefs();
  document.body.classList.toggle("export-invert", prefs.invert);
  document.documentElement.style.background = themeOf(prefs.invert).bg;
  document.body.style.background = prefs.invert ? "#e8eaef" : "#0f1117";
  fitPreview();
  if (persist) savePrefs(prefs);
}

function renderChart() {
  if (!spec || !window.Chart) return;
  applyPreviewChrome();
  const canvas = document.getElementById("export-canvas");
  const invert = currentInvert();
  if (chart) chart.destroy();
  chart = new window.Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets: spec.datasets || [] },
    options: buildOptions(spec, invert),
  });
  fitPreview();
}

function downloadPng() {
  if (!chart) return;
  const a = document.createElement("a");
  a.href = chart.toBase64Image("image/png", 1);
  a.download = `${safeName(spec?.title || "chart")}.png`;
  a.click();
}

function acceptSpec(next) {
  spec = next;
  document.getElementById("export-title").textContent = spec.title || "导出图表";
  document.getElementById("export-hint").classList.add("hidden");
  document.title = `${spec.title || "图表"} · 导出`;
  renderChart();
}

async function boot() {
  paintInstanceBadge();
  const prefs = await loadPrefs();
  document.getElementById("export-invert").checked = !!prefs.invert;
  document.getElementById("export-width").value = String(prefs.width || 960);
  document.getElementById("export-height").value = String(prefs.height || 360);
  applyPreviewChrome(false);

  document.getElementById("export-invert").addEventListener("change", renderChart);
  const onSize = () => {
    applyPreviewChrome();
    if (chart) chart.resize();
    fitPreview();
  };
  document.getElementById("export-width").addEventListener("input", onSize);
  document.getElementById("export-height").addEventListener("input", onSize);
  window.addEventListener("resize", () => { fitPreview(); });
  document.getElementById("export-download").addEventListener("click", downloadPng);

  window.addEventListener("message", (ev) => {
    if (ev.origin !== location.origin) return;
    if (ev.data?.type !== "bdelf-export" || !ev.data.spec) return;
    acceptSpec(ev.data.spec);
  });

  if (window.opener) {
    window.opener.postMessage({ type: "bdelf-export-ready" }, location.origin);
  }
}

boot();
