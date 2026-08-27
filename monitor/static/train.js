import { fetchJSON, sendJSON, el, pct, fmtNum, qs, setActiveNav, kindBadge, distinguishingKeys, formatRunDelta, clickableImage } from "./common.js";

const CHART_REFRESH_MS = 60000;
const LIST_REFRESH_LIVE_MS = 1000;
const LIST_REFRESH_IDLE_MS = 15000;
const PANELS_KEY = "bdelf_monitor_panels";

function progressBars(progress) {
  const wrap = el("div", { className: "run-progress" });
  const overall = el("div", { className: "muted", text: `总进度 ${pct(progress.fraction)} · tokens ${fmtNum(progress.tokens)} / ${fmtNum(progress.target_tokens)}` });
  wrap.appendChild(overall);
  const bar = el("div", { className: "progress-bar" }, [
    el("span", { style: `width:${Math.min(100, (progress.fraction || 0) * 100)}%` }),
  ]);
  wrap.appendChild(bar);
  if (progress.stages && progress.stages.length) {
    for (const st of progress.stages) {
      wrap.appendChild(el("div", {
        className: "muted",
        text: `${st.name}: ${pct(st.fraction)} (${fmtNum(st.done_tokens)} / ${fmtNum(st.budget)})`,
      }));
      wrap.appendChild(el("div", { className: "progress-bar" }, [
        el("span", { style: `width:${Math.min(100, (st.fraction || 0) * 100)}%` }),
      ]));
    }
  }
  return wrap;
}

function openRun(run) {
  location.href = `/train.html?run=${encodeURIComponent(run)}`;
}

function openModel(kind, model) {
  location.href = `/train.html?kind=${encodeURIComponent(kind)}&model=${encodeURIComponent(model)}`;
}

function runMetaText(r, { clickable = true } = {}) {
  const core = `step ${r.last?.step ?? "—"} · loss ${fmtNum(r.last?.train_loss, 4)} · ${fmtNum(r.last?.tokens_per_sec, 0)} tok/s`;
  return clickable ? `${core} · 双击进入图表` : core;
}

function applyRunCardData(card, r) {
  const badge = card.querySelector(":scope > .card-title-row > .badge");
  if (badge) {
    badge.className = `badge ${r.live ? "badge-live" : "badge-idle"}`;
    badge.textContent = r.live ? "训练中" : "已训完";
  }
  const prog = card.querySelector(".run-progress");
  if (prog) prog.replaceWith(progressBars(r.progress || {}));
  const meta = card.querySelector(".run-meta");
  if (meta) {
    const clickable = card.dataset.clickable !== "0";
    meta.textContent = runMetaText(r, { clickable });
  }
}

function runCard(r, { clickable = true, deltaKeys = null } = {}) {
  const delta = formatRunDelta(r, deltaKeys);
  const card = el("div", { className: clickable ? "card clickable" : "card" });
  card.dataset.run = r.run || "";
  card.dataset.clickable = clickable ? "1" : "0";
  if (clickable) {
    card.addEventListener("dblclick", () => openRun(r.run));
  }
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("h3", { text: delta.short, style: "margin:0", title: delta.full || delta.short }),
    el("span", {
      className: `badge ${r.live ? "badge-live" : "badge-idle"}`,
      text: r.live ? "训练中" : "已训完",
    }),
  ]));
  card.appendChild(el("div", {
    className: "muted",
    text: r.hash,
    title: r.run,
  }));
  card.appendChild(progressBars(r.progress || {}));
  card.appendChild(el("div", {
    className: "muted run-meta",
    text: runMetaText(r, { clickable }),
  }));
  return card;
}

function modelCard(m) {
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => openModel(m.kind, m.model));
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("div", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
      kindBadge(m.kind),
      el("h3", { text: m.model, style: "margin:0" }),
    ]),
    el("span", {
      className: `badge ${m.live ? "badge-live" : "badge-idle"}`,
      text: m.live ? `活跃 ${m.live_count}/${m.count}` : "空闲",
    }),
  ]));
  card.appendChild(el("div", {
    className: "muted",
    text: `${m.count} 个 full hash · 双击查看哈希`,
  }));
  return card;
}

function currentKindFilter() {
  const btn = document.querySelector(".kind-filter.active");
  return btn ? (btn.dataset.kind || "") : "";
}

function applyModelFilter() {
  const data = window.__modelData;
  if (!data) return;
  const q = (document.getElementById("model-search")?.value || "").trim().toLowerCase();
  const kind = currentKindFilter();
  let models = data.models || [];
  if (kind) models = models.filter((m) => m.kind === kind);
  if (q) models = models.filter((m) => String(m.model).toLowerCase().includes(q));
  paintModelList(models, data.models?.length || 0);
}

function paintModelList(models, total) {
  const sec = document.getElementById("model-section");
  sec.innerHTML = "";
  const kind = currentKindFilter();
  const q = (document.getElementById("model-search")?.value || "").trim();
  const bits = [`显示 ${models.length} / ${total}`];
  if (kind) bits.push(kind === "latent" ? "Latent" : "LM");
  if (q) bits.push(`搜索 “${q}”`);
  sec.appendChild(el("h2", { text: `模型 (${bits.join(" · ")})` }));
  if (!models.length) {
    sec.appendChild(el("p", { className: "muted", text: "没有匹配的 full 模型" }));
    return;
  }
  const live = models.filter((m) => m.live);
  const idle = models.filter((m) => !m.live);
  if (live.length) {
    sec.appendChild(el("h3", { text: `正在训练 (${live.length})` }));
    const grid = el("div", { className: "list-stack" });
    live.forEach((m) => grid.appendChild(modelCard(m)));
    sec.appendChild(grid);
  }
  sec.appendChild(el("h3", { text: `已训完 / 空闲 (${idle.length})` }));
  if (idle.length) {
    const grid = el("div", { className: "list-stack" });
    idle.forEach((m) => grid.appendChild(modelCard(m)));
    sec.appendChild(grid);
  } else {
    sec.appendChild(el("p", { className: "muted", text: "暂无空闲模型" }));
  }
}

function wireModelFilters() {
  const search = document.getElementById("model-search");
  if (search && !search.dataset.wired) {
    search.dataset.wired = "1";
    search.addEventListener("input", applyModelFilter);
  }
  document.querySelectorAll(".kind-filter").forEach((btn) => {
    if (btn.dataset.wired) return;
    btn.dataset.wired = "1";
    btn.addEventListener("click", () => {
      document.querySelectorAll(".kind-filter").forEach((b) => b.classList.toggle("active", b === btn));
      applyModelFilter();
    });
  });
}

function showView(name) {
  document.getElementById("model-view").classList.toggle("hidden", name !== "model");
  document.getElementById("hash-view").classList.toggle("hidden", name !== "hash");
  document.getElementById("detail-view").classList.toggle("hidden", name !== "detail");
}

function renderModels(data) {
  showView("model");
  window.__runs = data.runs || [];
  window.__modelData = data;
  wireModelFilters();
  applyModelFilter();
}

function renderHashes(data, kind, model, { keepScroll = false } = {}) {
  showView("hash");
  const sec = document.getElementById("hash-section");
  const y = keepScroll ? window.scrollY : null;
  sec.innerHTML = "";
  const runs = (data.runs || []).filter((r) => r.model === model && (!kind || r.kind === kind));
  const deltaKeys = distinguishingKeys(runs);
  const live = runs.filter((r) => r.live);
  const done = runs.filter((r) => !r.live);
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem" }, [
    kindBadge(kind || (runs[0] && runs[0].kind) || "lm"),
    el("span", { text: `${model} · ${runs.length} 个 hash` }),
  ]));
  if (live.length) {
    sec.appendChild(el("h3", { text: `正在训练 (${live.length})` }));
    const grid = el("div", { className: "list-stack" });
    live.forEach((r) => grid.appendChild(runCard(r, { deltaKeys })));
    sec.appendChild(grid);
  }
  sec.appendChild(el("h3", { text: `已训完 (${done.length})` }));
  if (done.length) {
    const grid = el("div", { className: "list-stack" });
    done.forEach((r) => grid.appendChild(runCard(r, { deltaKeys })));
    sec.appendChild(grid);
  } else {
    sec.appendChild(el("p", { className: "muted", text: "暂无已结束的 hash" }));
  }
  window.__runs = data.runs || [];
  window.__hashSig = hashListSignature(runs);
  if (y != null) window.scrollTo(0, y);
}

function hashListSignature(runs) {
  return runs.map((r) => `${r.run}:${r.live ? 1 : 0}`).join("|");
}

function runsListUrl({ model, kind } = {}) {
  const q = new URLSearchParams();
  if (model) q.set("model", model);
  if (kind) q.set("kind", kind);
  const s = q.toString();
  return s ? `/api/runs?${s}` : "/api/runs";
}

function modelRunsOf(data, kind, model) {
  return (data.runs || []).filter((r) => r.model === model && (!kind || r.kind === kind));
}

function patchHashCards(data, kind, model) {
  const runs = modelRunsOf(data, kind, model);
  const sig = hashListSignature(runs);
  if (sig !== window.__hashSig) {
    renderHashes(data, kind, model, { keepScroll: true });
    return;
  }
  const sec = document.getElementById("hash-section");
  const byRun = {};
  sec.querySelectorAll(".card[data-run]").forEach((card) => {
    byRun[card.dataset.run] = card;
  });
  for (const r of runs) {
    const card = byRun[r.run];
    if (card) applyRunCardData(card, r);
  }
  window.__runs = data.runs || [];
}

function stopHashRefresh() {
  window.__hashGen = (window.__hashGen || 0) + 1;
  clearTimeout(window.__hashTimer);
  window.__hashTimer = null;
}

function startHashRefresh(kind, model) {
  stopHashRefresh();
  const gen = window.__hashGen;
  const loop = async () => {
    if (window.__hashGen !== gen) return;
    try {
      const data = await fetchJSON(runsListUrl({ model, kind }));
      if (window.__hashGen !== gen) return;
      patchHashCards(data, kind, model);
    } catch (_) {}
    if (window.__hashGen !== gen) return;
    const mine = modelRunsOf({ runs: window.__runs || [] }, kind, model);
    const ms = mine.some((r) => r.live) ? LIST_REFRESH_LIVE_MS : LIST_REFRESH_IDLE_MS;
    window.__hashTimer = setTimeout(loop, ms);
  };
  loop();
}

async function loadRuns() {
  const data = await fetchJSON("/api/runs");
  renderModels(data);
  return data;
}

const TRACE_COLORS = ["#5b9fff", "#3dd68c", "#f5a623", "#ff6b8a", "#b388ff", "#5eead4", "#f472b6", "#facc15"];
const SKIP_METRIC_COLS = new Set(["step", "tokens", "curriculum_stage"]);
const SERIES_SOURCES = ["train", "eval", "eval_official", "train_official"];
const BILLION = 1e9;

function normalizeSource(source) {
  if (source === "eval" || source === "eval_official" || source === "train_official") return source;
  return "train";
}

function defaultTrace(i, metric, source, extra = {}) {
  return {
    source: normalizeSource(source),
    metric: metric || "train_loss",
    y_min: extra.y_min ?? "",
    y_max: extra.y_max ?? "",
    color: TRACE_COLORS[i % TRACE_COLORS.length],
    refs: normalizeRefs(extra.refs),
  };
}

function defaultPanel(run) {
  return {
    runs: run ? [run] : [],
    mode: "range",
    tokens_from_b: "",
    tokens_to_b: "",
    last_b: "",
    max_points: "4096",
    traces: [],
  };
}

/** 按模型类型写死的默认图，不落盘；删除只记该 hash 不再显示。 */
const OWT_REF_GEN_PPL = 17.0083; // GPT-2 Large · OWT eval L=1024，见 eval/report.py
const OWT_REF_UNIQ_MEAN = {
  elf: 434.7, // T5-small · elf eval 全量 chunk unique-id 均值
  default: 457.2, // GPT-2 · default eval 全量 chunk unique-id 均值
};

const DEFAULT_CHARTS = {
  lm: [
    {
      id: "lm-core",
      name: "lm",
      traces: [
        { source: "train", metric: "train_loss", y_min: "0", y_max: "2" },
        { source: "eval", metric: "eval_loss", y_min: "0" },
        { source: "eval", metric: "gen_ppl", y_min: "0", y_max: "100" },
        { source: "eval", metric: "gen_uniq_mean", y_min: "0", y_max: "500" },
        { source: "eval_official", metric: "gen_len_mean", y_min: "0", y_max: "2048" },
        { source: "train", metric: "lr", y_min: "0", y_max: "0.2" },
      ],
    },
  ],
  latent: [
    {
      id: "latent-core",
      name: "latent",
      traces: [
        { source: "train", metric: "train_loss", y_min: "0", y_max: "2" },
        { source: "eval", metric: "eval_loss", y_min: "0" },
        { source: "train", metric: "recon_ce", y_min: "0" },
        { source: "train", metric: "kl", y_min: "0" },
        { source: "train", metric: "token_acc", y_min: "0", y_max: "1" },
        { source: "train", metric: "lr", y_min: "0", y_max: "0.2" },
      ],
    },
  ],
};

function newPanelId() {
  return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function fieldToB(raw) {
  if (raw === "" || raw == null) return "";
  const n = Number(raw);
  if (!Number.isFinite(n)) return "";
  if (Math.abs(n) >= 1e6) return String(n / BILLION);
  return String(n);
}

function pickField(src, key, fallbackRaw) {
  if (Object.prototype.hasOwnProperty.call(src, key)) return String(src[key] ?? "");
  return fieldToB(fallbackRaw);
}

function normalizePanel(panel) {
  const cur = currentRunInfo().run;
  const src = panel || {};
  let mode = src.mode;
  if (mode !== "range" && mode !== "last") {
    mode = (src.last || src.last_b) ? "last" : "range";
  }
  let traces;
  if (Array.isArray(src.traces)) {
    traces = src.traces.map((t, i) => ({
      ...defaultTrace(i, t.metric, t.source),
      ...t,
      source: normalizeSource(t.source),
      refs: normalizeRefs(t.refs),
    }));
  } else {
    const source = normalizeSource(src.source);
    const metrics = String(src.metrics || "train_loss").split(",").map((s) => s.trim()).filter(Boolean);
    traces = (metrics.length ? metrics : ["train_loss"]).map((m, i) => defaultTrace(i, m, source));
  }
  return {
    id: String(src.id || "").trim(),
    name: String(src.name || "").trim(),
    runs: cur ? [cur] : [],
    mode,
    tokens_from_b: pickField(src, "tokens_from_b", src.tokens_from),
    tokens_to_b: pickField(src, "tokens_to_b", src.tokens_to),
    last_b: pickField(src, "last_b", src.last),
    max_points: src.max_points || "4096",
    traces,
    follow_latest: typeof src.follow_latest === "boolean" ? src.follow_latest : null,
  };
}

function panelsStorageKey() {
  const cur = currentRunInfo();
  return `${PANELS_KEY}:${cur.kind}/${cur.model}`;
}

function chartsQuery() {
  const cur = currentRunInfo();
  return { kind: cur.kind || "", model: cur.model || "" };
}

function loadPanels() {
  if (Array.isArray(window.__panels)) return window.__panels.map(normalizePanel);
  try {
    const raw = localStorage.getItem(panelsStorageKey());
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed.map(normalizePanel);
    }
  } catch (_) {}
  return [];
}

function savePanels(panels) {
  const normalized = panels.map((p) => {
    const n = normalizePanel(p);
    if (!n.id) n.id = newPanelId();
    return n;
  });
  window.__panels = normalized;
  try {
    localStorage.setItem(panelsStorageKey(), JSON.stringify(normalized));
  } catch (_) {}
  schedulePersistPanels();
}

function schedulePersistPanels() {
  clearTimeout(window.__persistPanelsTimer);
  window.__persistPanelsTimer = setTimeout(() => { persistPanelsNow(); }, 400);
}

async function persistPanelsNow() {
  const q = chartsQuery();
  const panels = Array.isArray(window.__panels) ? window.__panels : [];
  const dismissed = window.__dismissed && typeof window.__dismissed === "object" ? window.__dismissed : {};
  const order = window.__chartOrder && typeof window.__chartOrder === "object" ? window.__chartOrder : {};
  try {
    await sendJSON("/api/charts", "PUT", {
      kind: q.kind,
      model: q.model,
      panels,
      dismissed,
      order,
    });
  } catch (e) {
    console.warn("保存图表配置失败", e);
  }
}

async function fetchPanelsFromServer() {
  const q = chartsQuery();
  window.__dismissed = {};
  window.__chartOrder = {};
  try {
    const data = await fetchJSON(`/api/charts?kind=${encodeURIComponent(q.kind)}&model=${encodeURIComponent(q.model)}`);
    if (data.found) {
      window.__panels = Array.isArray(data.panels) ? data.panels.map(normalizePanel) : [];
      window.__dismissed = data.dismissed && typeof data.dismissed === "object" ? data.dismissed : {};
      window.__chartOrder = data.order && typeof data.order === "object" ? data.order : {};
      if (window.__panels.some((p) => !p.id)) savePanels(window.__panels);
      else {
        try {
          localStorage.setItem(panelsStorageKey(), JSON.stringify(window.__panels));
        } catch (_) {}
      }
      return window.__panels;
    }
  } catch (e) {
    console.warn("读取图表配置失败", e);
  }
  const local = loadPanels();
  window.__panels = local;
  if (local.length) schedulePersistPanels();
  return local;
}

function metricAvailable(source, metric) {
  const cols = window.__chartCols?.cols;
  if (!cols) return true;
  const key = normalizeSource(source);
  const pool = cols[key];
  if (!Array.isArray(pool) || !pool.length) return key === "train";
  return pool.includes(metric);
}

function runPreprocessName() {
  const ident = window.__runDetail?.identity || {};
  const listed = (window.__runs || []).find((x) => x.run === currentRunInfo().run);
  return String(ident.preprocess || listed?.identity?.preprocess || "").toLowerCase();
}

function lmDatasetRefs(metric) {
  if (metric === "gen_ppl") {
    return [{ value: String(OWT_REF_GEN_PPL), note: "数据集 gen-ppl" }];
  }
  if (metric === "gen_uniq_mean") {
    const prep = runPreprocessName();
    const uniq = (prep.includes("elf") || prep.includes("t5"))
      ? OWT_REF_UNIQ_MEAN.elf
      : OWT_REF_UNIQ_MEAN.default;
    return [{ value: String(uniq), note: "数据集 uniq" }];
  }
  return [];
}

function resolveMetricSource(preferred, metric) {
  const cols = window.__chartCols?.cols;
  const pref = normalizeSource(preferred);
  if (!cols) return pref;
  const order = [pref];
  for (const s of SERIES_SOURCES) {
    if (!order.includes(s)) order.push(s);
  }
  for (const s of order) {
    const pool = cols[s];
    if (Array.isArray(pool) && pool.includes(metric)) return s;
  }
  return null;
}

function materializeDefaultChart(tmpl) {
  const run = currentRunInfo().run;
  const traces = [];
  (tmpl.traces || []).forEach((t) => {
    const source = resolveMetricSource(t.source, t.metric);
    if (!source) return;
    const refs = (t.refs && t.refs.length) ? t.refs : lmDatasetRefs(t.metric);
    traces.push(defaultTrace(traces.length, t.metric, source, {
      refs,
      y_min: t.y_min,
      y_max: t.y_max,
    }));
  });
  if (!traces.length) return null;
  return stampFollowLatest({
    ...defaultPanel(run),
    name: tmpl.name || tmpl.id,
    traces,
  });
}

function dismissedIdsForHash() {
  const hash = currentRunInfo().hash;
  const all = window.__dismissed || {};
  return new Set(all[hash] || []);
}

function dismissDefaultForHash(id) {
  const hash = currentRunInfo().hash;
  if (!hash || !id) return;
  const all = { ...(window.__dismissed || {}) };
  const set = new Set(all[hash] || []);
  set.add(id);
  all[hash] = [...set];
  window.__dismissed = all;
  schedulePersistPanels();
}

function viewKey(item) {
  if (!item) return "";
  return item.origin === "default" ? `d:${item.defaultId}` : `c:${item.panel.id}`;
}

function loadOrderKeys() {
  const hash = currentRunInfo().hash;
  const all = window.__chartOrder || {};
  return Array.isArray(all[hash]) ? [...all[hash]] : null;
}

function saveOrderKeys(keys) {
  const hash = currentRunInfo().hash;
  if (!hash) return;
  const all = { ...(window.__chartOrder || {}) };
  all[hash] = keys.filter(Boolean);
  window.__chartOrder = all;
  schedulePersistPanels();
}

function mutateOrderKeys(fn) {
  let keys = loadOrderKeys();
  if (!keys) keys = (window.__viewItems || []).map(viewKey);
  saveOrderKeys(fn(keys));
}

function applyOrder(items) {
  const order = loadOrderKeys();
  if (!order || !order.length) return items;
  const map = new Map(items.map((it) => [viewKey(it), it]));
  const out = [];
  for (const key of order) {
    const it = map.get(key);
    if (!it) continue;
    out.push(it);
    map.delete(key);
  }
  items.forEach((it) => {
    if (map.has(viewKey(it))) out.push(it);
  });
  return out;
}

function visibleViewItems() {
  const kind = currentRunInfo().kind === "latent" ? "latent" : "lm";
  const dismissed = dismissedIdsForHash();
  const items = [];
  for (const tmpl of DEFAULT_CHARTS[kind] || []) {
    if (dismissed.has(tmpl.id)) continue;
    const panel = materializeDefaultChart(tmpl);
    if (!panel) continue;
    items.push({ origin: "default", defaultId: tmpl.id, panel });
  }
  loadPanels().forEach((panel, customIdx) => {
    items.push({ origin: "custom", customIdx, panel });
  });
  return applyOrder(items);
}

function viewItem(idx) {
  return (window.__viewItems || [])[idx];
}

function panelElId(key) {
  return `panel-${key}`;
}

function itemIndexByKey(key) {
  return (window.__viewItems || []).findIndex((it) => viewKey(it) === key);
}

function bumpPanelGen(key) {
  if (!window.__panelGen) window.__panelGen = {};
  window.__panelGen[key] = (window.__panelGen[key] || 0) + 1;
  return window.__panelGen[key];
}

function bindPanelIdx(key, fn) {
  return () => {
    const i = itemIndexByKey(key);
    if (i >= 0) fn(i);
  };
}

function destroyPanelByKey(key) {
  bumpPanelGen(key);
  const chart = window.__charts?.[key];
  if (chart) {
    try { chart.destroy(); } catch (_) {}
    delete window.__charts[key];
  }
  if (window.__seriesCache) delete window.__seriesCache[key];
  document.getElementById(panelElId(key))?.remove();
}

function syncMoveButtons() {
  const items = window.__viewItems || [];
  const n = items.length;
  items.forEach((it, i) => {
    const card = document.getElementById(panelElId(viewKey(it)));
    if (!card) return;
    const up = card.querySelector("[data-act=up]");
    const down = card.querySelector("[data-act=down]");
    if (up) up.disabled = i === 0;
    if (down) down.disabled = i === n - 1;
  });
}

function updatePanelChrome(idx) {
  const item = viewItem(idx);
  if (!item) return;
  const card = document.getElementById(panelElId(viewKey(item)));
  if (!card) return;
  const nameEl = card.querySelector(".chart-name");
  if (nameEl) nameEl.textContent = panelName(item.panel);
  const title = card.querySelector(".chart-toolbar-title");
  if (title) {
    const badge = title.querySelector(".badge-chart-default");
    if (item.origin === "default") {
      if (!badge) title.appendChild(el("span", { className: "badge badge-chart-default", text: "默认" }));
    } else if (badge) {
      badge.remove();
    }
  }
  paintFollowStatus(idx);
  syncMoveButtons();
}

async function replacePanelAt(idx, newItem) {
  const old = viewItem(idx);
  const wrap = document.getElementById("panels");
  const oldCard = old ? document.getElementById(panelElId(viewKey(old))) : null;
  const before = oldCard?.nextSibling || null;
  if (old) destroyPanelByKey(viewKey(old));
  window.__viewItems[idx] = newItem;
  await renderPanel(newItem, { before: before && before.parentNode === wrap ? before : null });
  syncMoveButtons();
}

function metricsTitle(panel) {
  const names = (normalizePanel(panel).traces || []).map((t) => t.metric).filter(Boolean);
  return names.length ? names.join(" / ") : "空图表";
}

function panelName(panel) {
  const n = String(normalizePanel(panel).name || "").trim();
  return n || metricsTitle(panel);
}

function copyChartName(name) {
  const base = String(name || "").trim() || "图表";
  return `${base}-copy`;
}

function panelConfigKey(panel) {
  const p = normalizePanel(panel);
  return JSON.stringify({
    name: String(p.name || "").trim(),
    mode: p.mode,
    tokens_from_b: p.tokens_from_b,
    tokens_to_b: p.tokens_to_b,
    last_b: p.last_b,
    max_points: String(p.max_points || "4096"),
    traces: (p.traces || []).map((t) => ({
      source: t.source,
      metric: t.metric,
      y_min: t.y_min ?? "",
      y_max: t.y_max ?? "",
      color: t.color || "",
      refs: t.refs || [],
    })),
  });
}

function bToTokens(raw) {
  if (raw === "" || raw == null) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * BILLION);
}

function tokensToB(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return null;
  return n / BILLION;
}

function finiteMetricY(raw) {
  if (raw === "" || raw == null) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function pointsToXY(points, metric) {
  const out = [];
  for (const pt of points || []) {
    const y = finiteMetricY(pt[metric]);
    if (y == null) continue;
    const x = tokensToB(pt.x);
    if (x == null) continue;
    out.push({ x, y });
  }
  return out;
}

function buildSeriesUrl(panel, trace, run, after) {
  const p = new URLSearchParams();
  p.set("run", run);
  p.set("source", normalizeSource(trace.source));
  p.set("metrics", trace.metric);
  if (after != null && Number.isFinite(Number(after))) {
    p.set("after", String(after));
    if (panel.mode === "range") {
      const to = bToTokens(panel.tokens_to_b);
      if (to != null) p.set("tokens_to", String(to));
    }
  } else if (panel.mode === "last") {
    const last = bToTokens(panel.last_b);
    if (last != null) p.set("last", String(last));
  } else {
    const from = bToTokens(panel.tokens_from_b);
    const to = bToTokens(panel.tokens_to_b);
    if (from != null) p.set("tokens_from", String(from));
    if (to != null) p.set("tokens_to", String(to));
  }
  if (trace.source !== "eval" && panel.max_points) p.set("max_points", panel.max_points);
  return `/api/series?${p}`;
}

function traceCacheKey(trace) {
  return `${trace.source || "train"}::${trace.metric}`;
}

function downsamplePoints(points, maxPoints) {
  const n = points.length;
  if (!maxPoints || n <= maxPoints) return { points, downsampled: false };
  if (maxPoints <= 2) return { points: points.slice(0, maxPoints), downsampled: true };
  const stride = (n - 1) / (maxPoints - 1);
  const indices = new Set([0, n - 1]);
  for (let i = 1; i < maxPoints - 1; i++) indices.add(Math.round(i * stride));
  return { points: [...indices].sort((a, b) => a - b).map((i) => points[i]), downsampled: true };
}

function trimPointsToWindow(points, panel) {
  if (panel.mode !== "last" || !points.length) return points;
  const last = bToTokens(panel.last_b);
  if (last == null) return points;
  const maxX = points[points.length - 1].x;
  const lo = maxX - last;
  return points.filter((pt) => pt.x >= lo);
}

function lastTokenOf(points) {
  if (!points.length) return null;
  const x = Number(points[points.length - 1].x);
  return Number.isFinite(x) ? x : null;
}

function currentRunTokens(run) {
  const r = (window.__runs || []).find((x) => x.run === run);
  const n = Number(r?.last?.tokens ?? r?.progress?.tokens ?? 0);
  return Number.isFinite(n) ? n : 0;
}

function panelWindowEndTokens(panel) {
  const p = normalizePanel(panel);
  if (p.mode === "last") return null;
  return bToTokens(p.tokens_to_b);
}

function includesLatestWindow(panel, currentTokens) {
  const p = normalizePanel(panel);
  if (p.mode === "last") return true;
  const end = bToTokens(p.tokens_to_b);
  if (end == null) return true;
  return currentTokens < end;
}

function stampFollowLatest(panel) {
  const p = normalizePanel(panel);
  const tokens = currentRunTokens(p.runs[0]);
  p.follow_latest = includesLatestWindow(p, tokens);
  return p;
}

function ensureFollowLatest(panel) {
  const p = normalizePanel(panel);
  if (typeof p.follow_latest === "boolean") return p;
  return stampFollowLatest(p);
}

function shouldAutoRefresh(panel) {
  const p = normalizePanel(panel);
  if (!p.follow_latest) return false;
  const run = p.runs[0];
  const r = (window.__runs || []).find((x) => x.run === run);
  if (!r?.live) return false;
  const end = panelWindowEndTokens(p);
  if (end != null && currentRunTokens(run) >= end) return false;
  return true;
}

function followStatus(panel) {
  const p = normalizePanel(panel);
  const end = panelWindowEndTokens(p);
  const tokens = currentRunTokens(p.runs[0]);
  if (shouldAutoRefresh(p)) return "自动刷新";
  if (end != null && tokens >= end) return "已超过末端";
  if (p.follow_latest) return "训练已结束";
  return "不含最新点";
}

function paintFollowStatus(idx) {
  const item = viewItem(idx);
  if (!item) return;
  const card = document.getElementById(panelElId(viewKey(item)));
  const elStatus = card?.querySelector("[data-follow]");
  if (!elStatus) return;
  elStatus.textContent = followStatus(item.panel || {});
}

function stopFollowIfCaughtUp(panel) {
  const p = normalizePanel(panel);
  if (!p.follow_latest) return p;
  const end = panelWindowEndTokens(p);
  if (end != null && currentRunTokens(p.runs[0]) >= end) {
    p.follow_latest = false;
  }
  return p;
}

function field(label, control, className) {
  return el("div", className ? { className } : {}, [el("label", { text: label }), control]);
}

const AXIS_TICK = "#f4f6fb";
const AXIS_GRID = "rgba(255, 255, 255, 0.45)";
const AXIS_LINE = "#ffffff";

function baseAxis() {
  return {
    ticks: { color: AXIS_TICK, font: { size: 12, weight: "600" } },
    grid: { color: AXIS_GRID, lineWidth: 1 },
    border: { color: AXIS_LINE },
    title: { display: true, color: AXIS_TICK, font: { size: 13, weight: "600" } },
  };
}

function parseBound(raw) {
  if (raw === "" || raw == null) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : undefined;
}

function normalizeRefs(refs) {
  if (!Array.isArray(refs)) return [];
  return refs.map((r) => ({
    value: r?.value == null ? "" : String(r.value),
    note: r?.note == null ? "" : String(r.note),
  }));
}

function collectRefLineItems(traces) {
  const items = [];
  (traces || []).forEach((t, i) => {
    const color = t.color || TRACE_COLORS[i % TRACE_COLORS.length];
    (t.refs || []).forEach((r) => {
      const y = Number(r.value);
      if (!Number.isFinite(y)) return;
      items.push({
        y,
        yAxisID: `y${i}`,
        color,
        note: (r.note || "").trim() || String(r.value),
      });
    });
  });
  return items;
}

function runStages(run) {
  if (window.__runDetail?.run === run && Array.isArray(window.__runDetail.progress?.stages)) {
    return window.__runDetail.progress.stages;
  }
  const r = (window.__runs || []).find((x) => x.run === run);
  return Array.isArray(r?.progress?.stages) ? r.progress.stages : [];
}

function collectStageLineItems(panel) {
  const p = normalizePanel(panel);
  const stages = runStages(p.runs[0]);
  if (stages.length < 2) return [];
  const items = [];
  for (let i = 1; i < stages.length; i++) {
    const st = stages[i];
    const prev = stages[i - 1];
    const tokens = Number(st.start_tokens);
    if (!Number.isFinite(tokens) || tokens <= 0) continue;
    const x = tokensToB(tokens);
    if (x == null) continue;
    const prevName = prev?.name || `s${i}`;
    const name = st.name || `s${i + 1}`;
    items.push({
      x,
      note: `分割 ${prevName} | ${name}`,
    });
  }
  return items;
}

function collectChartRefItems(panel) {
  const p = normalizePanel(panel);
  return collectRefLineItems(p.traces).concat(collectStageLineItems(p));
}

function drawRefLineItem(chart, item) {
  const { ctx } = chart;
  const area = chart.chartArea;
  const color = item.color || chart.options.color || AXIS_TICK;
  const xScale = chart.scales.x;
  if (item.x != null && Number.isFinite(Number(item.x))) {
    if (!xScale) return;
    const x = xScale.getPixelForValue(Number(item.x));
    if (x < area.left || x > area.right) return;
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
    return;
  }
  const yScale = chart.scales[item.yAxisID];
  if (!yScale) return;
  const y = yScale.getPixelForValue(item.y);
  if (y < area.top || y > area.bottom) return;
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

const refLinePlugin = {
  id: "refLines",
  afterDatasetsDraw(chart) {
    const items = chart.options.plugins?.refLines?.items || [];
    if (!items.length) return;
    for (const item of items) drawRefLineItem(chart, item);
  },
};
if (window.Chart) window.Chart.register(refLinePlugin);

function yScalesForTraces(traces) {
  const scales = {};
  (traces || []).forEach((t, i) => {
    const axis = baseAxis();
    const color = t.color || TRACE_COLORS[i % TRACE_COLORS.length];
    const scale = {
      ...axis,
      position: i === 0 ? "left" : "right",
      grid: { ...axis.grid, drawOnChartArea: i === 0 },
      title: { ...axis.title, text: t.metric || "value", color },
      ticks: { ...axis.ticks, color },
    };
    const min = parseBound(t.y_min);
    const max = parseBound(t.y_max);
    if (min !== undefined) scale.min = min;
    if (max !== undefined) scale.max = max;
    scales[`y${i}`] = scale;
  });
  return scales;
}

function chartOptions(panel) {
  const p = normalizePanel(panel);
  const axis = baseAxis();
  return {
    responsive: true,
    maintainAspectRatio: false,
    color: AXIS_TICK,
    interaction: { mode: "nearest", intersect: false },
    scales: {
      x: {
        type: "linear",
        ...axis,
        title: { ...axis.title, text: "tokens (B)" },
      },
      ...yScalesForTraces(p.traces),
    },
    plugins: {
      legend: { labels: { color: AXIS_TICK, font: { size: 12, weight: "600" } } },
      refLines: { items: collectChartRefItems(p) },
    },
  };
}

function panelTitle(panel) {
  return panelName(panel);
}

function currentRunInfo() {
  const run = qs("run") || "";
  const parts = run.split("/").filter(Boolean);
  if (parts.length === 4) {
    return { run, variant: parts[0], kind: parts[1], model: parts[2], hash: parts[3] };
  }
  if (parts.length === 3) {
    return { run, variant: parts[0], kind: "", model: parts[1], hash: parts[2] };
  }
  return { run, variant: "", kind: qs("kind") || "", model: qs("model") || "", hash: "" };
}

function metricCols(cols) {
  return (cols || []).filter((c) => c && !SKIP_METRIC_COLS.has(c));
}

async function loadChartColumns() {
  const run = qs("run");
  if (!run) return { train: ["train_loss"], eval: ["eval_loss"], eval_official: [], train_official: [] };
  if (window.__chartCols && window.__chartCols.run === run) return window.__chartCols.cols;
  const detail = await fetchJSON(`/api/runs/${encodeURIComponent(run)}`);
  const src = detail.sources || {};
  const cols = {
    train: metricCols(src.train),
    eval: metricCols(src.eval),
    eval_official: metricCols(src.eval_official),
    train_official: metricCols(src.train_official),
  };
  if (!cols.train.length) cols.train = ["train_loss"];
  window.__chartCols = { run, cols };
  return cols;
}

function syncModeFields() {
  const mode = document.querySelector("#modal-mode .page-tab.active")?.dataset.mode || "range";
  document.getElementById("modal-range-fields")?.classList.toggle("hidden", mode !== "range");
  document.getElementById("modal-last-fields")?.classList.toggle("hidden", mode !== "last");
}

function renderMetricChips(wrap, source, selected) {
  wrap.innerHTML = "";
  const cols = (window.__modalCols || {})[source] || [];
  if (!cols.length) {
    wrap.appendChild(el("span", { className: "muted", text: "该数据源没有可用列" }));
    wrap.dataset.metric = "";
    return "";
  }
  if (!cols.includes(selected)) selected = cols[0];
  cols.forEach((name) => {
    wrap.appendChild(el("button", {
      type: "button",
      className: `metric-chip${name === selected ? " active" : ""}`,
      text: name,
    }));
  });
  wrap.dataset.metric = selected;
  return selected;
}

function formatYRange(trace) {
  const lo = (trace.y_min ?? "") === "" ? "min" : trace.y_min;
  const hi = (trace.y_max ?? "") === "" ? "max" : trace.y_max;
  if (lo === "min" && hi === "max") return "Y 自动";
  return `Y ${lo}–${hi}`;
}

function readOpenRefs() {
  const box = document.getElementById("trace-refs");
  if (!box) return [];
  return [...box.querySelectorAll(".ref-row")].map((row) => ({
    value: row.querySelector(".ref-value")?.value ?? "",
    note: row.querySelector(".ref-note")?.value ?? "",
  }));
}

function renderRefRow(ref) {
  const row = el("div", { className: "ref-row" });
  row.appendChild(el("input", {
    className: "ref-value",
    value: ref.value ?? "",
    placeholder: "Y 值",
  }));
  row.appendChild(el("input", {
    className: "ref-note",
    value: ref.note ?? "",
    placeholder: "说明（可选）",
  }));
  row.appendChild(el("button", {
    type: "button",
    className: "secondary",
    text: "删除",
    onclick: (ev) => ev.currentTarget.closest(".ref-row")?.remove(),
  }));
  return row;
}

function readOpenTraceForm() {
  const source = document.getElementById("trace-source")?.value || "train";
  const chips = document.getElementById("trace-metrics");
  const metric = chips?.dataset.metric || chips?.querySelector(".metric-chip.active")?.textContent || "";
  return {
    source: normalizeSource(source),
    metric,
    y_min: document.getElementById("trace-ymin")?.value ?? "",
    y_max: document.getElementById("trace-ymax")?.value ?? "",
    color: document.getElementById("trace-color")?.value || TRACE_COLORS[0],
    refs: readOpenRefs(),
  };
}

function syncOpenTraceEditor() {
  const idx = window.__editingTraceIdx;
  if (idx == null || !document.getElementById("trace-source")) return;
  const traces = window.__modalTraces || [];
  if (!traces[idx]) return;
  traces[idx] = { ...traces[idx], ...readOpenTraceForm() };
}

function renderTraceRow(trace, idx) {
  const row = el("div", { className: "trace-row" });
  row.appendChild(el("span", {
    className: "trace-swatch",
    style: `background:${trace.color || TRACE_COLORS[idx % TRACE_COLORS.length]}`,
  }));
  row.appendChild(el("span", {
    className: "trace-row-label",
    text: `${trace.source || "train"} · ${trace.metric || "未选指标"}`,
  }));
  row.appendChild(el("span", { className: "muted", text: formatYRange(trace) }));
  const nref = (trace.refs || []).filter((r) => r.value !== "").length;
  if (nref) row.appendChild(el("span", { className: "muted", text: `参考 ${nref}` }));
  row.appendChild(el("button", { type: "button", className: "secondary", text: "编辑", onclick: () => editTrace(idx) }));
  row.appendChild(el("button", { type: "button", className: "secondary", text: "删除", onclick: () => removeTraceEditor(idx) }));
  row.addEventListener("click", (ev) => {
    if (ev.target.closest("button")) return;
    editTrace(idx);
  });
  return row;
}

function renderTraceEditor(trace, idx) {
  const card = el("div", { className: "trace-card", id: "trace-edit" });
  card.appendChild(el("div", { className: "chart-toolbar" }, [
    el("span", { text: `数据 ${idx + 1}`, style: "margin-right:auto;align-self:center;font-weight:600" }),
    el("button", { type: "button", className: "secondary", text: "完成", onclick: () => finishTraceEditor() }),
    el("button", { type: "button", className: "secondary", text: "删除", onclick: () => removeTraceEditor(idx) }),
  ]));
  const src = el("select", { id: "trace-source" }, SERIES_SOURCES.map((name) => el("option", {
    value: name,
    text: name,
    ...(normalizeSource(trace.source) === name ? { selected: "" } : {}),
  })));
  const chips = el("div", { className: "metric-chips", id: "trace-metrics" });
  src.addEventListener("change", () => renderMetricChips(chips, src.value, ""));
  renderMetricChips(chips, src.value, trace.metric);
  chips.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".metric-chip");
    if (!btn) return;
    chips.querySelectorAll(".metric-chip").forEach((b) => b.classList.toggle("active", b === btn));
    chips.dataset.metric = btn.textContent;
  });
  card.appendChild(el("div", { className: "form-row" }, [
    field("数据源", src),
    field("颜色", el("input", {
      id: "trace-color",
      type: "color",
      value: trace.color || TRACE_COLORS[idx % TRACE_COLORS.length],
    })),
    field("Y 下限（空=最小）", el("input", { id: "trace-ymin", value: trace.y_min ?? "", placeholder: "自动" }), "y-bound-field"),
    field("Y 上限（空=最大）", el("input", { id: "trace-ymax", value: trace.y_max ?? "", placeholder: "自动" }), "y-bound-field"),
  ]));
  card.appendChild(el("div", {}, [
    el("label", { text: "指标" }),
    chips,
  ]));
  const refsBox = el("div", { id: "trace-refs" });
  (trace.refs || []).forEach((r) => refsBox.appendChild(renderRefRow(r)));
  card.appendChild(el("div", { style: "margin-top:0.55rem" }, [
    el("label", { text: "参考线" }),
    refsBox,
    el("button", {
      type: "button",
      className: "secondary",
      text: "添加参考线",
      onclick: () => refsBox.appendChild(renderRefRow({ value: "", note: "" })),
    }),
  ]));
  return card;
}

function editTrace(idx) {
  syncOpenTraceEditor();
  window.__editingTraceIdx = idx;
  paintTraceEditors();
}

function finishTraceEditor() {
  syncOpenTraceEditor();
  const t = window.__modalTraces?.[window.__editingTraceIdx];
  if (t && !t.metric) {
    alert("请选择指标");
    return;
  }
  window.__editingTraceIdx = null;
  paintTraceEditors();
}

function startAddTrace() {
  syncOpenTraceEditor();
  const traces = window.__modalTraces || [];
  const cols = window.__modalCols || {};
  const metric = (cols.train || ["train_loss"])[0];
  traces.push(defaultTrace(traces.length, metric, "train"));
  window.__modalTraces = traces;
  window.__editingTraceIdx = traces.length - 1;
  paintTraceEditors();
}

function removeTraceEditor(idx) {
  syncOpenTraceEditor();
  const traces = window.__modalTraces || [];
  traces.splice(idx, 1);
  window.__modalTraces = traces;
  if (window.__editingTraceIdx == null) {
    /* keep collapsed */
  } else if (window.__editingTraceIdx === idx) {
    window.__editingTraceIdx = null;
  } else if (window.__editingTraceIdx > idx) {
    window.__editingTraceIdx -= 1;
  }
  paintTraceEditors();
}

function paintTraceEditors() {
  const box = document.getElementById("modal-traces");
  if (!box) return;
  const traces = window.__modalTraces || [];
  const editing = window.__editingTraceIdx;
  box.innerHTML = "";
  traces.forEach((t, i) => {
    box.appendChild(i === editing ? renderTraceEditor(t, i) : renderTraceRow(t, i));
  });
}

function readModalTraces() {
  syncOpenTraceEditor();
  return (window.__modalTraces || []).map((t, i) => ({
    ...defaultTrace(i, t.metric, t.source),
    ...t,
    source: normalizeSource(t.source),
  }));
}

function readModalPanel() {
  const cur = currentRunInfo().run;
  const mode = document.querySelector("#modal-mode .page-tab.active")?.dataset.mode || "range";
  return normalizePanel({
    id: window.__editPanelId || "",
    name: document.getElementById("modal-name")?.value ?? "",
    runs: cur ? [cur] : [],
    mode,
    tokens_from_b: document.getElementById("modal-from-b")?.value ?? "",
    tokens_to_b: document.getElementById("modal-to-b")?.value ?? "",
    last_b: document.getElementById("modal-last-b")?.value ?? "",
    max_points: document.getElementById("modal-max_points")?.value || "4096",
    traces: readModalTraces(),
  });
}

async function openChartModal(editIdx) {
  window.__editIdx = editIdx;
  const run = qs("run");
  const item = editIdx == null ? null : viewItem(editIdx);
  const panel = normalizePanel(item ? item.panel : defaultPanel(run));
  window.__editPanelId = item?.origin === "custom" ? (item.panel.id || "") : "";
  try {
    window.__modalCols = await loadChartColumns();
  } catch (_) {
    window.__modalCols = { train: ["train_loss"], eval: [], eval_official: [], train_official: [] };
  }
  window.__modalTraces = panel.traces.map((t, i) => ({ ...defaultTrace(i, t.metric, t.source), ...t }));
  window.__editingTraceIdx = null;
  document.getElementById("chart-modal-title").textContent = editIdx == null
    ? "添加图表"
    : (item?.origin === "default" ? "图表设置（默认）" : "图表设置");
  const fields = document.getElementById("chart-modal-fields");
  fields.innerHTML = "";

  fields.appendChild(el("div", { className: "form-row" }, [
    field("名称", el("input", {
      id: "modal-name",
      value: item ? panelName(panel) : "",
      placeholder: "图表名称",
    })),
  ]));

  const modeBar = el("div", { className: "page-tabs", id: "modal-mode" }, [
    el("button", { type: "button", className: `page-tab${panel.mode !== "last" ? " active" : ""}`, "data-mode": "range", text: "token 范围" }),
    el("button", { type: "button", className: `page-tab${panel.mode === "last" ? " active" : ""}`, "data-mode": "last", text: "最近 token" }),
  ]);
  modeBar.querySelectorAll(".page-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      modeBar.querySelectorAll(".page-tab").forEach((b) => b.classList.toggle("active", b === btn));
      syncModeFields();
    });
  });
  fields.appendChild(el("label", { text: "横轴模式（单位 B）" }));
  fields.appendChild(modeBar);

  fields.appendChild(el("div", { className: "form-row", id: "modal-range-fields" }, [
    field("从 (B，空=最前)", el("input", { id: "modal-from-b", value: panel.tokens_from_b, placeholder: "空" })),
    field("到 (B，空=最后)", el("input", { id: "modal-to-b", value: panel.tokens_to_b, placeholder: "空" })),
  ]));
  fields.appendChild(el("div", { className: "form-row", id: "modal-last-fields" }, [
    field("最近 (B)", el("input", { id: "modal-last-b", value: panel.last_b, placeholder: "例如 0.5" })),
  ]));
  fields.appendChild(el("div", { className: "form-row" }, [
    field("max_points", el("input", { id: "modal-max_points", value: panel.max_points || "4096" })),
  ]));
  fields.appendChild(el("h3", { text: "数据", style: "margin:0.6rem 0 0.2rem" }));
  fields.appendChild(el("div", { id: "modal-traces" }));
  fields.appendChild(el("button", {
    type: "button",
    className: "secondary",
    text: "添加数据",
    onclick: startAddTrace,
  }));
  paintTraceEditors();
  syncModeFields();
  document.getElementById("chart-modal").classList.remove("hidden");
}

function closeChartModal() {
  document.getElementById("chart-modal").classList.add("hidden");
  window.__editIdx = null;
  window.__editPanelId = "";
}

function saveChartModal() {
  let panel = readModalPanel();
  if (!panel.runs.length) {
    alert("当前没有选中的 hash");
    return;
  }
  if (!panel.traces.length || panel.traces.some((t) => !t.metric)) {
    alert("请至少添加一条数据，并用选项卡选择指标");
    return;
  }
  if (!panel.name) panel.name = metricsTitle(panel);
  panel = stampFollowLatest(panel);
  const idx = window.__editIdx;
  const item = idx == null ? null : viewItem(idx);
  if (item?.origin === "default") {
    if (panelConfigKey(panel) === panelConfigKey(item.panel)) {
      closeChartModal();
      return;
    }
    panel.id = newPanelId();
    dismissDefaultForHash(item.defaultId);
    const panels = loadPanels();
    panels.push(panel);
    savePanels(panels);
    mutateOrderKeys((keys) => {
      const i = keys.indexOf(`d:${item.defaultId}`);
      if (i >= 0) {
        const next = keys.slice();
        next[i] = `c:${panel.id}`;
        return next;
      }
      return keys.concat(`c:${panel.id}`);
    });
  } else if (item?.origin === "custom") {
    panel.id = item.panel.id || newPanelId();
    const panels = loadPanels();
    const at = panels.findIndex((p) => p.id === item.panel.id);
    if (at >= 0) panels[at] = panel;
    else panels.push(panel);
    savePanels(panels);
  } else {
    panel.id = newPanelId();
    const panels = loadPanels();
    panels.push(panel);
    savePanels(panels);
    if (loadOrderKeys()) {
      mutateOrderKeys((keys) => keys.concat(`c:${panel.id}`));
    }
  }
  closeChartModal();
  applySavedPanel(idx, item, panel);
}

async function applySavedPanel(idx, item, panel) {
  if (item?.origin === "default") {
    await replacePanelAt(idx, { origin: "custom", panel });
    return;
  }
  if (item?.origin === "custom") {
    window.__viewItems[idx] = { ...item, panel };
    updatePanelChrome(idx);
    await refreshPanel(idx, { full: true });
    return;
  }
  const newItem = { origin: "custom", panel };
  window.__viewItems = (window.__viewItems || []).concat(newItem);
  await renderPanel(newItem);
  syncMoveButtons();
}

function snapshotDatasets(chart) {
  return (chart.data.datasets || []).map((d) => ({
    label: d.label,
    data: (d.data || []).map((pt) => ({ x: pt.x, y: pt.y })),
    borderColor: d.borderColor,
    backgroundColor: "transparent",
    tension: d.tension ?? 0.1,
    pointRadius: d.pointRadius ?? 0,
    borderWidth: 2,
    yAxisID: d.yAxisID,
  }));
}

function openExportWindow(idx) {
  const key = viewKey(viewItem(idx));
  const chart = key ? window.__charts?.[key] : null;
  const panel = normalizePanel(viewItem(idx)?.panel || {});
  if (!chart || !panel.traces?.length) {
    alert("当前图表没有可导出的数据");
    return;
  }
  const spec = {
    title: panelTitle(panel),
    traces: panel.traces,
    datasets: snapshotDatasets(chart),
    refItems: collectChartRefItems(panel),
  };
  if (!spec.datasets.some((d) => d.data?.length)) {
    alert("当前图表没有可导出的数据");
    return;
  }
  ensureExportHandshake();
  window.__pendingExport = spec;
  const w = window.open("/export.html", "_blank");
  if (!w) {
    alert("浏览器拦截了新窗口，请允许弹窗后再导出");
    return;
  }
}

function ensureExportHandshake() {
  if (window.__exportHandshake) return;
  window.__exportHandshake = true;
  window.addEventListener("message", (ev) => {
    if (ev.origin !== location.origin) return;
    if (ev.data?.type !== "bdelf-export-ready") return;
    if (!window.__pendingExport || !ev.source) return;
    ev.source.postMessage({ type: "bdelf-export", spec: window.__pendingExport }, location.origin);
  });
}

async function renderPanel(item, { before = null, after = null } = {}) {
  const panel = item.panel;
  const key = viewKey(item);
  const wrap = document.getElementById("panels");
  const card = el("div", { className: "card", id: panelElId(key) });
  const titleBits = [
    el("span", { className: "chart-name", text: panelName(panel) }),
  ];
  if (item.origin === "default") {
    titleBits.push(el("span", { className: "badge badge-chart-default", text: "默认" }));
  }
  const upBtn = el("button", { className: "secondary", "data-act": "up", text: "上移", onclick: bindPanelIdx(key, (i) => movePanel(i, -1)) });
  const downBtn = el("button", { className: "secondary", "data-act": "down", text: "下移", onclick: bindPanelIdx(key, (i) => movePanel(i, 1)) });
  card.appendChild(el("div", { className: "chart-toolbar" }, [
    el("div", { className: "chart-toolbar-title" }, titleBits),
    el("span", { className: "muted", "data-follow": "1", text: followStatus(panel), style: "align-self:center" }),
    upBtn,
    downBtn,
    el("button", { className: "secondary", text: "复制", onclick: bindPanelIdx(key, copyPanel) }),
    el("button", { className: "secondary", text: "刷新", onclick: bindPanelIdx(key, (i) => refreshPanel(i, { full: true })) }),
    el("button", { className: "secondary", text: "导出", onclick: bindPanelIdx(key, openExportWindow) }),
    el("button", { className: "secondary", text: "设置", onclick: bindPanelIdx(key, openChartModal) }),
    el("button", { className: "secondary", text: "删除", onclick: bindPanelIdx(key, removePanel) }),
  ]));
  const canvas = el("canvas");
  card.appendChild(el("div", { className: "chart-wrap" }, [canvas]));
  card.appendChild(el("div", { className: "muted", "data-meta": "1" }));
  if (before) wrap.insertBefore(card, before);
  else if (after) {
    if (after.nextSibling) wrap.insertBefore(card, after.nextSibling);
    else wrap.appendChild(card);
  } else {
    wrap.appendChild(card);
  }

  const ctx = canvas.getContext("2d");
  const chart = new Chart(ctx, {
    type: "line",
    data: { datasets: [] },
    options: chartOptions(panel),
  });
  if (!window.__charts) window.__charts = {};
  window.__charts[key] = chart;
  const idx = itemIndexByKey(key);
  await refreshPanel(idx, { full: true });
  syncMoveButtons();
}

async function refreshPanel(idx, { full = false } = {}) {
  const epoch = window.__panelEpoch;
  const item = viewItem(idx);
  const key = viewKey(item);
  if (!key) return;
  if (full) bumpPanelGen(key);
  const gen = window.__panelGen?.[key] || 0;
  const panel = normalizePanel(item?.panel || {});
  const chart = window.__charts?.[key];
  const meta = document.getElementById(panelElId(key))?.querySelector("[data-meta]");
  if (!panel || !chart) return;
  const run = panel.runs[0];
  const datasets = [];
  const notes = [];
  if (!window.__seriesCache) window.__seriesCache = {};
  if (full || !window.__seriesCache[key]) window.__seriesCache[key] = {};
  if (!run) {
    window.__seriesCache[key] = {};
    chart.data.datasets = [];
    chart.update();
    if (meta) meta.textContent = "当前没有选中的 hash";
    return;
  }
  const cap = parseInt(panel.max_points, 10) || 4096;
  for (let i = 0; i < panel.traces.length; i++) {
    const trace = panel.traces[i];
    const tkey = traceCacheKey(trace);
    const cached = full ? null : window.__seriesCache[key][tkey];
    const after = cached && cached.lastX != null ? cached.lastX : null;
    try {
      const data = await fetchJSON(buildSeriesUrl(panel, trace, run, after));
      if (epoch !== window.__panelEpoch || (window.__panelGen?.[key] || 0) !== gen || window.__charts[key] !== chart) return;
      const metric = trace.metric;
      let points = (cached && after != null)
        ? cached.points.concat((data.points || []).filter((pt) => pt.x > after))
        : (data.points || []);
      points = trimPointsToWindow(points, panel);
      const sampled = downsamplePoints(points, cap);
      points = sampled.points;
      const lastX = lastTokenOf(points) ?? after;
      const nRaw = (cached && after != null)
        ? (cached.n_raw || 0) + (data.n_raw || 0)
        : (data.n_raw || points.length);
      const downsampled = !!(sampled.downsampled || (cached && cached.downsampled) || data.downsampled);
      window.__seriesCache[key][tkey] = { lastX, points, n_raw: nRaw, downsampled };
      datasets.push({
        label: `${trace.source} · ${metric}`,
        data: pointsToXY(points, metric),
        borderColor: trace.color || TRACE_COLORS[i % TRACE_COLORS.length],
        backgroundColor: "transparent",
        tension: 0.1,
        pointRadius: downsampled ? 0 : 2,
        yAxisID: `y${i}`,
      });
      const extra = after != null && data.n_returned ? ` +${data.n_returned}` : "";
      notes.push(`${trace.source}/${metric}: ${points.length}/${nRaw}${downsampled ? " (下采样)" : ""}${extra}`);
    } catch (e) {
      if (epoch !== window.__panelEpoch || (window.__panelGen?.[key] || 0) !== gen || window.__charts[key] !== chart) return;
      const fallback = window.__seriesCache[key][tkey];
      if (fallback?.points?.length) {
        datasets.push({
          label: `${trace.source} · ${trace.metric}`,
          data: pointsToXY(fallback.points, trace.metric),
          borderColor: trace.color || TRACE_COLORS[i % TRACE_COLORS.length],
          backgroundColor: "transparent",
          tension: 0.1,
          pointRadius: fallback.downsampled ? 0 : 2,
          yAxisID: `y${i}`,
        });
      }
      notes.push(`${trace.source}/${trace.metric}: 错误 ${e.message}`);
    }
  }
  if (epoch !== window.__panelEpoch || (window.__panelGen?.[key] || 0) !== gen || window.__charts[key] !== chart) return;
  chart.data.datasets = datasets;
  const nextOpts = chartOptions(panel);
  chart.options.scales = nextOpts.scales;
  chart.options.plugins = nextOpts.plugins;
  chart.update("none");
  if (meta) meta.textContent = notes.join(" · ");
  paintFollowStatus(idx);
}

function movePanel(idx, delta) {
  const items = window.__viewItems || [];
  const j = idx + delta;
  if (j < 0 || j >= items.length) return;
  const wrap = document.getElementById("panels");
  const cardA = document.getElementById(panelElId(viewKey(items[idx])));
  const cardB = document.getElementById(panelElId(viewKey(items[j])));
  if (!wrap || !cardA || !cardB) return;
  if (delta < 0) wrap.insertBefore(cardA, cardB);
  else wrap.insertBefore(cardB, cardA);
  const tmp = items[idx];
  items[idx] = items[j];
  items[j] = tmp;
  saveOrderKeys(items.map(viewKey));
  syncMoveButtons();
}

function copyPanel(idx) {
  const item = viewItem(idx);
  if (!item) return;
  const src = normalizePanel(item.panel);
  const copy = stampFollowLatest({
    ...src,
    id: newPanelId(),
    name: copyChartName(panelName(src)),
  });
  const panels = loadPanels();
  if (item.origin === "custom") {
    const at = panels.findIndex((p) => p.id === item.panel.id);
    panels.splice(at >= 0 ? at + 1 : panels.length, 0, copy);
  } else {
    panels.push(copy);
  }
  savePanels(panels);
  mutateOrderKeys((keys) => {
    const i = keys.indexOf(viewKey(item));
    if (i >= 0) {
      const next = keys.slice();
      next.splice(i + 1, 0, `c:${copy.id}`);
      return next;
    }
    return keys.concat(`c:${copy.id}`);
  });
  const newItem = { origin: "custom", panel: copy };
  window.__viewItems.splice(idx + 1, 0, newItem);
  const after = document.getElementById(panelElId(viewKey(item)));
  renderPanel(newItem, { after }).then(() => syncMoveButtons());
}

function removePanel(idx) {
  const item = viewItem(idx);
  if (!item) return;
  if (item.origin === "default") {
    dismissDefaultForHash(item.defaultId);
    mutateOrderKeys((keys) => keys.filter((k) => k !== `d:${item.defaultId}`));
  } else if (item.origin === "custom") {
    const panels = loadPanels().filter((p) => p.id !== item.panel.id);
    savePanels(panels);
    mutateOrderKeys((keys) => keys.filter((k) => k !== `c:${item.panel.id}`));
  }
  const key = viewKey(item);
  window.__viewItems.splice(idx, 1);
  destroyPanelByKey(key);
  syncMoveButtons();
}

async function initPanels() {
  const wrap = document.getElementById("panels");
  window.__panelEpoch = (window.__panelEpoch || 0) + 1;
  const epoch = window.__panelEpoch;
  if (window.__charts) {
    Object.values(window.__charts).forEach((c) => {
      try { c.destroy(); } catch (_) {}
    });
  }
  wrap.innerHTML = "";
  window.__charts = {};
  window.__seriesCache = {};
  window.__panelGen = {};
  try {
    await loadChartColumns();
  } catch (_) {}
  if (epoch !== window.__panelEpoch) return;
  const custom = loadPanels();
  if (custom.some((p) => typeof p.follow_latest !== "boolean" || !p.id)) {
    savePanels(custom.map(ensureFollowLatest));
  }
  window.__viewItems = visibleViewItems();
  for (let i = 0; i < window.__viewItems.length; i++) {
    if (epoch !== window.__panelEpoch) return;
    await renderPanel(window.__viewItems[i]);
  }
  clearInterval(window.__panelTimer);
  window.__panelTimer = setInterval(() => { tickChartRefresh(); }, CHART_REFRESH_MS);
}

async function tickChartRefresh() {
  if (window.__chartTickBusy) return;
  window.__chartTickBusy = true;
  const epoch = window.__panelEpoch;
  try {
    if (epoch !== window.__panelEpoch) return;
    const items = window.__viewItems || [];
    const custom = loadPanels();
    let dirty = false;
    for (let i = 0; i < items.length; i++) {
      if (epoch !== window.__panelEpoch) return;
      const item = items[i];
      const next = stopFollowIfCaughtUp(item.panel);
      if (next.follow_latest !== item.panel.follow_latest) {
        item.panel = next;
        if (item.origin === "custom") {
          const at = custom.findIndex((p) => p.id === item.panel.id);
          if (at >= 0) custom[at] = next;
          else custom[item.customIdx] = next;
          dirty = true;
        }
      }
      paintFollowStatus(i);
      if (shouldAutoRefresh(item.panel)) refreshPanel(i);
    }
    if (dirty) savePanels(custom);
  } finally {
    window.__chartTickBusy = false;
  }
}

function kvTable(obj) {
  const table = el("table", { className: "kv-table" });
  const entries = Object.entries(obj || {}).filter(([, v]) => v !== "" && v != null && typeof v !== "object");
  if (!entries.length) {
    return el("p", { className: "muted", text: "无标量参数" });
  }
  for (const [k, v] of entries) {
    table.appendChild(el("tr", {}, [
      el("th", { text: k }),
      el("td", { text: String(v) }),
    ]));
  }
  return table;
}

function setDetailTab(tab) {
  document.querySelectorAll("#detail-tabs .page-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  document.getElementById("charts-section").classList.toggle("hidden", tab !== "charts");
  document.getElementById("eval-page").classList.toggle("hidden", tab !== "eval");
  if (tab === "charts") {
    if (!document.getElementById("panels").children.length) initPanels();
  }
  if (tab === "eval") {
    const cfg = window.__evalSetup;
    if (cfg) setupEvalPage(cfg.run, cfg.steps, cfg.preferredStep);
  }
}

function wireDetailTabs() {
  document.querySelectorAll("#detail-tabs .page-tab").forEach((btn) => {
    if (btn.dataset.wired) return;
    btn.dataset.wired = "1";
    btn.addEventListener("click", () => setDetailTab(btn.dataset.tab));
  });
}

async function loadEvalStep(run, step) {
  const paramsBox = document.getElementById("eval-params");
  const idsBox = document.getElementById("eval-ids");
  const itemBox = document.getElementById("eval-item");
  paramsBox.innerHTML = "<p class='muted'>加载该步 eval…</p>";
  idsBox.innerHTML = "";
  itemBox.classList.add("hidden");
  itemBox.innerHTML = "";
  const data = await fetchJSON(`/api/runs/${encodeURIComponent(run)}/eval-samples/${step}`);
  window.__evalStep = { run, step, data };
  paramsBox.innerHTML = "";
  paramsBox.appendChild(el("h3", { text: `Eval 参数 · step ${step}` }));
  const params = data.params || {};
  paramsBox.appendChild(el("h3", { text: "eval_log", style: "margin-top:0.75rem" }));
  paramsBox.appendChild(kvTable(params.eval_log));
  if (params.eval_official && Object.keys(params.eval_official).length) {
    paramsBox.appendChild(el("h3", { text: "eval_components", style: "margin-top:0.75rem" }));
    paramsBox.appendChild(kvTable(params.eval_official));
  }
  if (params.meta && Object.keys(params.meta).length) {
    paramsBox.appendChild(el("h3", { text: "meta", style: "margin-top:0.75rem" }));
    paramsBox.appendChild(kvTable(params.meta));
  }

  const items = data.items || [];
  idsBox.appendChild(el("h3", { text: `条目 (${items.length})` }));
  if (!items.length) {
    idsBox.appendChild(el("p", { className: "muted", text: "该步没有逐条样本 / probe" }));
    return;
  }
  const sel = el("select", { id: "eval-item-select" });
  sel.appendChild(el("option", { value: "", text: "选择条目…" }));
  items.forEach((item) => {
    const id = item.id != null ? String(item.id) : "";
    const extra = item.length != null && item.length !== "" ? ` · length ${item.length}` : "";
    sel.appendChild(el("option", { value: id, text: `${id}${extra}` }));
  });
  sel.addEventListener("change", () => {
    const id = sel.value;
    if (!id) {
      itemBox.classList.add("hidden");
      itemBox.innerHTML = "";
      return;
    }
    showEvalItem(run, step, id);
  });
  idsBox.appendChild(sel);
}

async function showEvalItem(run, step, itemId) {
  const box = document.getElementById("eval-item");
  box.classList.remove("hidden");
  box.innerHTML = "<p class='muted'>加载该条…</p>";
  const data = await fetchJSON(
    `/api/runs/${encodeURIComponent(run)}/eval-samples/${step}/item/${encodeURIComponent(itemId)}`,
  );
  box.innerHTML = "";
  box.appendChild(el("h3", { text: `ID ${data.id}` }));
  if (data.view === "latent_probe") {
    if (data.metrics && Object.keys(data.metrics).length) {
      box.appendChild(kvTable(data.metrics));
    }
    const grid = el("div", { className: "img-grid" });
    for (const img of data.images || []) {
      grid.appendChild(el("div", {}, [
        el("div", { className: "muted", text: img.name }),
        clickableImage(img.url, { name: img.name, alt: img.name }),
      ]));
    }
    if (!(data.images || []).length) {
      box.appendChild(el("p", { className: "muted", text: "该条没有图像" }));
    } else {
      box.appendChild(grid);
    }
  } else {
    const sample = data.sample || {};
    const scalars = { ...sample };
    const text = scalars.text;
    delete scalars.text;
    box.appendChild(kvTable(scalars));
    box.appendChild(el("div", { className: "sample-text", text: text || "", style: "margin-top:0.75rem;max-height:none" }));
  }
}

function latestEvalStep(steps) {
  const nums = (steps || []).map(Number).filter((n) => Number.isFinite(n));
  if (!nums.length) return null;
  return Math.max(...nums);
}

function setupEvalPage(run, steps, preferredStep) {
  const sel = document.getElementById("eval-step-select");
  if (sel.dataset.loadedRun === run && sel.options.length) return;
  sel.innerHTML = "";
  sel.dataset.loadedRun = run;
  if (!steps.length) {
    document.getElementById("eval-params").innerHTML = "<p class='muted'>该 run 没有在线 eval 样本目录</p>";
    document.getElementById("eval-ids").innerHTML = "";
    document.getElementById("eval-item").classList.add("hidden");
    return;
  }
  const latest = latestEvalStep(steps);
  const want = String(preferredStep ?? "");
  const pick = (want && steps.some((st) => String(st) === want)) ? want : String(latest);
  for (const st of steps) {
    sel.appendChild(el("option", {
      value: String(st),
      text: String(st),
    }));
  }
  sel.value = pick;
  if (!sel.dataset.wired) {
    sel.dataset.wired = "1";
    sel.addEventListener("change", () => {
      const r = qs("run");
      if (r) loadEvalStep(r, parseInt(sel.value, 10));
    });
  }
  loadEvalStep(run, parseInt(sel.value, 10));
}

async function showDetail(run) {
  showView("detail");
  const header = document.getElementById("run-header");
  header.innerHTML = "<p class='muted'>加载 run 详情…</p>";

  const [listData, detail] = await Promise.all([
    fetchJSON("/api/runs"),
    fetchJSON(`/api/runs/${encodeURIComponent(run)}`),
  ]);
  window.__runs = listData.runs || [];
  const listed = (listData.runs || []).find((r) => r.run === run) || {};
  const merged = { ...listed, ...detail, run };
  window.__runDetail = merged;
  const back = document.getElementById("detail-back");
  if (merged.model) {
    const q = new URLSearchParams({ model: merged.model });
    if (merged.kind) q.set("kind", merged.kind);
    back.href = `/train.html?${q}`;
  }
  paintRunHeader(merged);
  startDetailProgressRefresh(run);
  wireDetailTabs();
  window.__evalSetup = { run, steps: detail.eval_steps || [], preferredStep: qs("estep") };
  await fetchPanelsFromServer();
  const tab = qs("tab") === "eval" ? "eval" : "charts";
  setDetailTab(tab);
}

function paintRunHeader(merged) {
  const header = document.getElementById("run-header");
  header.innerHTML = "";
  header.appendChild(runCard(merged, { clickable: false }));
}

function stopDetailProgressRefresh() {
  window.__detailProgGen = (window.__detailProgGen || 0) + 1;
  clearTimeout(window.__detailProgTimer);
  window.__detailProgTimer = null;
}

function startDetailProgressRefresh(run) {
  stopDetailProgressRefresh();
  const gen = window.__detailProgGen;
  const loop = async () => {
    if (window.__detailProgGen !== gen) return;
    try {
      await tickDetailProgress(run);
    } catch (_) {}
    if (window.__detailProgGen !== gen) return;
    const live = !!window.__runDetail?.live;
    window.__detailProgTimer = setTimeout(loop, live ? LIST_REFRESH_LIVE_MS : LIST_REFRESH_IDLE_MS);
  };
  loop();
}

async function tickDetailProgress(run) {
  const snap = await fetchJSON(`/api/progress?run=${encodeURIComponent(run)}`);
  const merged = { ...window.__runDetail, ...snap, run };
  window.__runDetail = merged;
  window.__runs = (window.__runs || []).map((r) => (
    r.run === run
      ? { ...r, live: snap.live, progress: snap.progress, last: snap.last, mtime: snap.mtime }
      : r
  ));
  const header = document.getElementById("run-header");
  const card = header?.querySelector(".card[data-run]");
  if (card) applyRunCardData(card, merged);
  else paintRunHeader(merged);
}

document.getElementById("add-panel").addEventListener("click", () => openChartModal(null));
document.getElementById("chart-modal-cancel").addEventListener("click", closeChartModal);
document.getElementById("chart-modal-backdrop").addEventListener("click", closeChartModal);
document.getElementById("chart-modal-save").addEventListener("click", saveChartModal);

async function main() {
  setActiveNav("train");
  const run = qs("run");
  const model = qs("model");
  const kind = qs("kind");
  if (run) {
    await showDetail(run);
    return;
  }
  try {
    const data = await fetchJSON(model ? runsListUrl({ model, kind }) : "/api/runs");
    window.__runs = data.runs || [];
    if (model) {
      renderHashes(data, kind, model);
      startHashRefresh(kind, model);
    } else {
      renderModels(data);
    }
  } catch (e) {
    const sec = document.getElementById("model-section");
    if (sec) sec.innerHTML = `<p class="muted">加载失败：${e.message}</p>`;
  }
  if (!run && !model) {
    setInterval(() => {
      loadRuns().catch(() => {});
    }, 15000);
  }
}

main().catch((e) => alert(e.message));
