import {
  fetchJSON,
  el,
  qs,
  setActiveNav,
  kindBadge,
  distinguishingKeys,
  formatRunDelta,
  clickableImage,
  inferKind,
} from "./common.js";

function evalHref({ kind, model, hash, step, gen } = {}) {
  const q = new URLSearchParams();
  if (kind) q.set("kind", kind);
  if (model) q.set("model", model);
  if (hash) q.set("hash", hash);
  if (step != null && step !== "") q.set("step", String(step));
  if (gen) q.set("gen", gen);
  const s = q.toString();
  return s ? `/eval.html?${s}` : "/eval.html";
}

function showView(name) {
  for (const id of ["model-view", "hash-view", "step-view", "detail-view", "sample-view"]) {
    const node = document.getElementById(id);
    if (node) node.classList.toggle("hidden", id !== `${name}-view`);
  }
}

function findModel(data, model, kind) {
  const models = data.models || [];
  if (kind) {
    return models.find((m) => m.model === model && m.kind === kind)
      || models.find((m) => m.model === model);
  }
  return models.find((m) => m.model === model);
}

function normalizeEvalData(data, runs) {
  const runByHash = {};
  for (const r of runs || []) {
    if (r.hash && !runByHash[r.hash]) runByHash[r.hash] = r;
  }
  const buckets = {};
  for (const m of data.models || []) {
    for (const h of m.hashes || []) {
      const run = runByHash[h.hash];
      const kind = inferKind(m.model, h.kind || m.kind || run?.kind);
      const steps = h.steps || [];
      const stepCount = h.step_count ?? steps.length;
      const runCount = h.run_count ?? steps.reduce(
        (n, s) => n + (s.run_count ?? (s.runs || []).length),
        0,
      );
      const latest = h.latest_step ?? steps.reduce(
        (acc, s) => (typeof s.step === "number" && (acc == null || s.step > acc) ? s.step : acc),
        null,
      );
      const key = `${kind}/${m.model}`;
      const bucket = buckets[key] || { kind, model: m.model, hashes: [] };
      buckets[key] = bucket;
      bucket.hashes.push({
        ...h,
        kind,
        model: m.model,
        identity: h.identity && Object.keys(h.identity).length
          ? h.identity
          : (run?.identity || {}),
        step_count: stepCount,
        run_count: runCount,
        latest_step: latest,
      });
    }
  }
  const models = Object.values(buckets).map((b) => ({
    ...b,
    count: b.hashes.length,
    step_count: b.hashes.reduce((n, h) => n + (h.step_count || 0), 0),
    run_count: b.hashes.reduce((n, h) => n + (h.run_count || 0), 0),
  }));
  models.sort((a, b) => `${a.kind}/${a.model}`.localeCompare(`${b.kind}/${b.model}`));
  return { models };
}

function currentKindFilter() {
  const btn = document.querySelector(".kind-filter.active");
  return btn ? (btn.dataset.kind || "") : "";
}

function applyModelFilter() {
  const data = window.__evalData;
  if (!data) return;
  const q = (document.getElementById("model-search")?.value || "").trim().toLowerCase();
  const kind = currentKindFilter();
  let models = data.models || [];
  if (kind) models = models.filter((m) => m.kind === kind);
  if (q) models = models.filter((m) => String(m.model).toLowerCase().includes(q));
  paintModelList(models, data.models?.length || 0);
}

function modelCard(m) {
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => {
    location.href = evalHref({ kind: m.kind, model: m.model });
  });
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("div", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
      kindBadge(m.kind),
      el("h3", { text: m.model, style: "margin:0" }),
    ]),
    el("span", {
      className: "badge badge-idle",
      text: `${m.count || 0} hash`,
    }),
  ]));
  card.appendChild(el("div", {
    className: "muted",
    text: `${m.step_count || 0} 个 step · ${m.run_count || 0} 组扫参 · 双击查看哈希`,
  }));
  return card;
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
    sec.appendChild(el("p", { className: "muted", text: "没有匹配的离线 eval" }));
    return;
  }
  const grid = el("div", { className: "list-stack" });
  models.forEach((m) => grid.appendChild(modelCard(m)));
  sec.appendChild(grid);
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

function renderModels() {
  showView("model");
  wireModelFilters();
  applyModelFilter();
}

function hashCard(h, { deltaKeys } = {}) {
  const delta = formatRunDelta(h, deltaKeys);
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => {
    location.href = evalHref({ kind: h.kind, model: h.model, hash: h.hash });
  });
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("h3", { text: delta.short, style: "margin:0", title: delta.full || delta.short }),
    el("span", {
      className: "badge badge-idle",
      text: h.latest_step != null ? `step ${h.latest_step}` : "无 step",
    }),
  ]));
  card.appendChild(el("div", {
    className: "muted",
    text: h.hash,
    title: h.hash,
  }));
  card.appendChild(el("div", {
    className: "muted",
    text: `${h.step_count || 0} 个 step · ${h.run_count || 0} 组扫参 · 双击查看步数`,
  }));
  return card;
}

function renderHashes(kind, model) {
  showView("hash");
  const sec = document.getElementById("hash-section");
  sec.innerHTML = "";
  const m = findModel(window.__evalData || {}, model, kind);
  const hashes = m?.hashes || [];
  const deltaKeys = distinguishingKeys(hashes);
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem" }, [
    kindBadge(kind || m?.kind || inferKind(model)),
    el("span", { text: `${model} · ${hashes.length} 个 hash` }),
  ]));
  if (!hashes.length) {
    sec.appendChild(el("p", { className: "muted", text: "该模型没有离线 eval hash" }));
    return;
  }
  const grid = el("div", { className: "list-stack" });
  hashes.forEach((h) => grid.appendChild(hashCard(h, { deltaKeys })));
  sec.appendChild(grid);
}

function stepCard(s, ctx) {
  const nRuns = s.run_count ?? (s.runs || []).length;
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => {
    location.href = evalHref({ ...ctx, step: s.step });
  });
  const badges = [
    el("span", { className: "badge badge-idle", text: `${nRuns} 组扫参` }),
  ];
  if (s.has_chart) {
    badges.push(el("span", { className: "badge badge-chart-default", text: "图" }));
  }
  if (s.has_table) {
    badges.push(el("span", { className: "badge badge-kind-lm", text: "表" }));
  }
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("h3", { text: `step ${s.step}`, style: "margin:0" }),
    el("div", { style: "display:flex;gap:0.35rem;flex-wrap:wrap" }, badges),
  ]));
  const names = (s.runs || []).map((r) => r.name).filter(Boolean);
  card.appendChild(el("div", {
    className: "muted",
    text: (names.length ? names.join(" · ") : "无扫参组名") + " · 双击查看结果",
  }));
  return card;
}

function renderSteps(kind, model, hash) {
  showView("step");
  const back = document.getElementById("step-back");
  if (back) back.href = evalHref({ kind, model });
  const sec = document.getElementById("step-section");
  sec.innerHTML = "";
  const m = findModel(window.__evalData || {}, model, kind);
  const h = (m?.hashes || []).find((x) => x.hash === hash);
  const steps = h?.steps || [];
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
    kindBadge(kind || m?.kind || inferKind(model)),
    el("span", { text: model }),
  ]));
  sec.appendChild(el("div", {
    className: "muted",
    text: hash,
    style: "margin-bottom:0.75rem",
  }));
  if (!steps.length) {
    sec.appendChild(el("p", { className: "muted", text: "该 hash 没有 eval step" }));
    return;
  }
  const grid = el("div", { className: "list-stack" });
  [...steps].sort((a, b) => (b.step || 0) - (a.step || 0)).forEach((s) => {
    grid.appendChild(stepCard(s, { kind, model, hash }));
  });
  sec.appendChild(grid);
}

function genCard(r, ctx) {
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => {
    location.href = evalHref({ ...ctx, gen: r.generate_hash });
  });
  const badges = [];
  if (r.has_samples) badges.push(el("span", { className: "badge badge-live", text: "samples" }));
  if (r.has_per_sample_csv) badges.push(el("span", { className: "badge badge-idle", text: "csv" }));
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("h3", { text: r.name || r.generate_hash, style: "margin:0" }),
    el("div", { style: "display:flex;gap:0.35rem;flex-wrap:wrap" }, badges),
  ]));
  card.appendChild(el("div", {
    className: "muted",
    text: `${r.generate_hash} · 双击查看逐条样本`,
  }));
  return card;
}

function metricsTable(rows) {
  const table = el("table");
  if (!rows.length) return table;
  const headers = Object.keys(rows[0]);
  table.appendChild(el("tr", {}, headers.map((h) => el("th", { text: h }))));
  for (const row of rows) {
    table.appendChild(el("tr", {}, headers.map((h) => el("td", { text: row[h] ?? "" }))));
  }
  return table;
}

async function renderStepDetail(kind, model, hash, step) {
  showView("detail");
  const back = document.getElementById("detail-back");
  if (back) back.href = evalHref({ kind, model, hash });
  const sec = document.getElementById("detail-section");
  sec.innerHTML = "<p class='muted'>加载该步 eval…</p>";
  const data = await fetchJSON(
    `/api/eval/${encodeURIComponent(model)}/${encodeURIComponent(hash)}/${step}`,
  );
  sec.innerHTML = "";
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
    kindBadge(kind || inferKind(model)),
    el("span", { text: `${model} · step ${step}` }),
  ]));
  sec.appendChild(el("div", { className: "muted", text: hash, style: "margin-bottom:0.75rem" }));

  if (data.chart_url) {
    sec.appendChild(el("div", { className: "card" }, [
      el("h3", { text: "results.png" }),
      clickableImage(data.chart_url, { name: "results.png", alt: "results", style: "max-width:100%" }),
    ]));
  }
  if (data.table_url) {
    sec.appendChild(el("div", { className: "card" }, [
      el("h3", { text: "results_table.png" }),
      clickableImage(data.table_url, { name: "results_table.png", alt: "table", style: "max-width:100%" }),
    ]));
  }

  const rows = data.metrics_rows || [];
  const csvCard = el("div", { className: "card" });
  csvCard.appendChild(el("h3", { text: "results.csv" }));
  if (rows.length) csvCard.appendChild(metricsTable(rows));
  else csvCard.appendChild(el("p", { className: "muted", text: "没有表格行" }));
  sec.appendChild(csvCard);

  sec.appendChild(el("h3", { text: `扫参组 (${(data.runs || []).length}) · 双击查看样本` }));
  const runs = data.runs || [];
  if (!runs.length) {
    sec.appendChild(el("p", { className: "muted", text: "该步没有扫参组" }));
    return;
  }
  const grid = el("div", { className: "list-stack" });
  runs.forEach((r) => grid.appendChild(genCard(r, { kind, model, hash, step })));
  sec.appendChild(grid);
}

async function renderSample(kind, model, hash, step, gen) {
  showView("sample");
  const back = document.getElementById("sample-back");
  if (back) back.href = evalHref({ kind, model, hash, step });
  const sec = document.getElementById("sample-section");
  sec.innerHTML = "<p class='muted'>加载逐条样本…</p>";
  const data = await fetchJSON(
    `/api/eval/${encodeURIComponent(model)}/${encodeURIComponent(hash)}/${step}/${encodeURIComponent(gen)}`,
  );
  sec.innerHTML = "";
  sec.appendChild(el("h2", { text: data.name || gen }));
  sec.appendChild(el("div", {
    className: "muted",
    text: `${model} · step ${step} · ${gen}`,
    style: "margin-bottom:0.75rem",
  }));

  const sumCard = el("div", { className: "card" });
  sumCard.appendChild(el("h3", { text: "汇总" }));
  sumCard.appendChild(el("pre", { className: "sample-text", text: JSON.stringify(data.summary, null, 2) }));
  sec.appendChild(sumCard);

  const samples = data.samples || [];
  const card = el("div", { className: "card" });
  card.appendChild(el("h3", { text: `逐条样本 (${samples.length})` }));
  const table = el("table");
  const keys = samples.length ? Object.keys(samples[0]) : ["text"];
  table.appendChild(el("tr", {}, keys.map((k) => el("th", { text: k }))));
  for (const s of samples) {
    table.appendChild(el("tr", {}, keys.map((k) => {
      const v = s[k];
      if (k === "text" || k === "text_preview") {
        return el("td", {}, [el("div", { className: "sample-text", text: v || "" })]);
      }
      return el("td", { text: v != null ? String(v) : "" });
    })));
  }
  card.appendChild(table);
  sec.appendChild(card);
}

async function main() {
  setActiveNav("eval");
  const model = qs("model");
  const hash = qs("hash");
  const step = qs("step");
  const gen = qs("gen");
  const kind = qs("kind") || "";

  const [evalData, runsData] = await Promise.all([
    fetchJSON("/api/eval"),
    fetchJSON("/api/runs").catch(() => ({ runs: [] })),
  ]);
  window.__evalData = normalizeEvalData(evalData, runsData.runs);

  if (model && hash && step && gen) {
    await renderSample(kind, model, hash, parseInt(step, 10), gen);
  } else if (model && hash && step) {
    await renderStepDetail(kind, model, hash, parseInt(step, 10));
  } else if (model && hash) {
    renderSteps(kind, model, hash);
  } else if (model) {
    renderHashes(kind, model);
  } else {
    renderModels();
  }
}

main().catch((e) => {
  showView("model");
  const sec = document.getElementById("model-section");
  if (sec) sec.innerHTML = `<p class="muted">加载失败：${e.message}</p>`;
});
