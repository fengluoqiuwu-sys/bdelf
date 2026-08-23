import {
  fetchJSON,
  el,
  fmtNum,
  qs,
  setActiveNav,
  kindBadge,
  distinguishingKeys,
  formatRunDelta,
} from "./common.js";

function href(parts = {}) {
  const q = new URLSearchParams();
  q.set("kind", "lm");
  if (parts.model) q.set("model", parts.model);
  if (parts.hash) q.set("hash", parts.hash);
  if (parts.run) q.set("run", parts.run);
  if (parts.ckpt) q.set("ckpt", parts.ckpt);
  const s = q.toString();
  return s ? `/generate.html?${s}` : "/generate.html";
}

function showView(name) {
  for (const id of ["remote-block", "model-view", "hash-view", "ckpt-view", "studio-view"]) {
    const node = document.getElementById(id);
    if (node) node.classList.toggle("hidden", id !== `${name}-view` && !(name === "remote" && id === "remote-block"));
  }
  if (name === "remote") {
    document.getElementById("remote-block")?.classList.remove("hidden");
    ["model-view", "hash-view", "ckpt-view", "studio-view"].forEach((id) => {
      document.getElementById(id)?.classList.add("hidden");
    });
  }
}

function lmRuns() {
  return (window.__runs || []).filter((r) => r.kind === "lm" && r.variant === "full");
}

function lmModels() {
  const map = new Map();
  for (const r of lmRuns()) {
    const key = r.model;
    const b = map.get(key) || { kind: "lm", model: r.model, count: 0, live_count: 0, live: false };
    b.count += 1;
    if (r.live) {
      b.live_count += 1;
      b.live = true;
    }
    map.set(key, b);
  }
  return [...map.values()].sort((a, b) => Number(b.live) - Number(a.live) || a.model.localeCompare(b.model));
}

function fmtBytes(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  if (x >= 1e9) return `${(x / 1e9).toFixed(2)} GB`;
  if (x >= 1e6) return `${(x / 1e6).toFixed(1)} MB`;
  if (x >= 1e3) return `${(x / 1e3).toFixed(0)} KB`;
  return `${x} B`;
}

function fmtWhen(ts) {
  if (!ts) return "";
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString();
}

function modelCard(m) {
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => { location.href = href({ model: m.model }); });
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("div", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
      kindBadge("lm"),
      el("h3", { text: m.model, style: "margin:0" }),
    ]),
    el("span", {
      className: `badge ${m.live ? "badge-live" : "badge-idle"}`,
      text: m.live ? `活跃 ${m.live_count}/${m.count}` : `${m.count} hash`,
    }),
  ]));
  card.appendChild(el("div", { className: "muted", text: `${m.count} 个 full hash · 双击查看哈希` }));
  return card;
}

function hashCard(r, { deltaKeys } = {}) {
  const delta = formatRunDelta(r, deltaKeys);
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => { location.href = href({ model: r.model, hash: r.hash, run: r.run }); });
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("h3", { text: delta.short, style: "margin:0", title: delta.full || delta.short }),
    el("span", { className: `badge ${r.live ? "badge-live" : "badge-idle"}`, text: r.live ? "训练中" : "已训完" }),
  ]));
  card.appendChild(el("div", { className: "muted", text: r.hash, title: r.run }));
  card.appendChild(el("div", {
    className: "muted",
    text: `step ${r.last?.step ?? "—"} · 双击选择 checkpoint`,
  }));
  return card;
}

function ckptCard(c, ctx) {
  const card = el("div", { className: "card clickable" });
  card.addEventListener("dblclick", () => {
    location.href = href({ ...ctx, ckpt: c.id });
  });
  card.appendChild(el("div", { className: "card-title-row" }, [
    el("h3", { text: c.name || c.file, style: "margin:0" }),
    el("span", { className: "badge badge-idle", text: fmtBytes(c.size) }),
  ]));
  card.appendChild(el("div", {
    className: "muted",
    text: `${c.file}${c.step != null ? ` · step ${c.step}` : ""} · ${fmtWhen(c.mtime)} · 双击打开`,
  }));
  return card;
}

function applyModelFilter() {
  const q = (document.getElementById("model-search")?.value || "").trim().toLowerCase();
  let models = lmModels();
  if (q) models = models.filter((m) => m.model.toLowerCase().includes(q));
  const sec = document.getElementById("model-section");
  sec.innerHTML = "";
  sec.appendChild(el("h2", { text: `模型 (显示 ${models.length} / ${lmModels().length} · 仅 LM)` }));
  if (!models.length) {
    sec.appendChild(el("p", { className: "muted", text: "没有匹配的 LM full run" }));
    return;
  }
  const grid = el("div", { className: "list-stack" });
  models.forEach((m) => grid.appendChild(modelCard(m)));
  sec.appendChild(grid);
}

function renderModels() {
  showView("model");
  const search = document.getElementById("model-search");
  if (search && !search.dataset.wired) {
    search.dataset.wired = "1";
    search.addEventListener("input", applyModelFilter);
  }
  applyModelFilter();
}

function renderHashes(model) {
  showView("hash");
  const sec = document.getElementById("hash-section");
  sec.innerHTML = "";
  const runs = lmRuns().filter((r) => r.model === model);
  const deltaKeys = distinguishingKeys(runs);
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem" }, [
    kindBadge("lm"),
    el("span", { text: `${model} · ${runs.length} 个 hash` }),
  ]));
  if (!runs.length) {
    sec.appendChild(el("p", { className: "muted", text: "该模型没有 LM hash" }));
    return;
  }
  const grid = el("div", { className: "list-stack" });
  runs.forEach((r) => grid.appendChild(hashCard(r, { deltaKeys })));
  sec.appendChild(grid);
}

function findRun(model, hash, runQ) {
  if (runQ) return lmRuns().find((r) => r.run === runQ);
  return lmRuns().find((r) => r.model === model && r.hash === hash);
}

async function renderCkpts(model, hash, runQ) {
  showView("ckpt");
  const back = document.getElementById("ckpt-back");
  if (back) back.href = href({ model });
  const sec = document.getElementById("ckpt-section");
  const listed = findRun(model, hash, runQ);
  const run = listed?.run || runQ;
  sec.innerHTML = "";
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
    kindBadge("lm"),
    el("span", { text: model }),
  ]));
  sec.appendChild(el("div", { className: "muted", text: hash, style: "margin-bottom:0.75rem" }));
  if (!run) {
    sec.appendChild(el("p", { className: "muted", text: "找不到对应 run" }));
    return;
  }
  try {
    const data = await fetchJSON(`/api/generate/checkpoints?run=${encodeURIComponent(run)}`);
    const list = data.checkpoints || [];
    if (!list.length) {
      sec.appendChild(el("p", { className: "muted", text: "该 hash 下没有 checkpoint_*.pt" }));
      return;
    }
    const grid = el("div", { className: "list-stack" });
    list.forEach((c) => grid.appendChild(ckptCard(c, { model, hash, run })));
    sec.appendChild(grid);
  } catch (e) {
    sec.appendChild(el("p", { className: "muted", text: `加载 checkpoint 失败：${e.message}` }));
  }
}

const SELECT_FIELDS = {
  sampling_method: ["sde", "ode"],
  time_schedule: ["logit_normal", "uniform", "linear"],
  dma_ace_order: ["after", "before", "skip_with_ace"],
  sampler: ["semi_ar"],
};

/** ACE 方向/步区间由 artifacts 自动查找，不在表单里配。 */
const HIDDEN_SAMPLING_KEYS = new Set(["ace_direction", "ace_step_lo", "ace_step_hi"]);

function samplingEntries(sampling) {
  return Object.entries(sampling || {}).filter(([k]) => !HIDDEN_SAMPLING_KEYS.has(k));
}

function fieldInput(key, value) {
  const wrap = el("div");
  wrap.appendChild(el("label", { text: key, for: `g-${key}` }));
  const choices = SELECT_FIELDS[key];
  if (choices) {
    const sel = el("select", { id: `g-${key}`, "data-key": key, "data-kind": "choice" });
    const cur = value == null ? "" : String(value);
    const opts = [...choices];
    if (cur && !opts.includes(cur)) opts.push(cur);
    opts.forEach((name) => {
      const opt = { value: name, text: name };
      if (name === cur) opt.selected = "selected";
      sel.appendChild(el("option", opt));
    });
    wrap.appendChild(sel);
    return wrap;
  }
  if (typeof value === "boolean") {
    wrap.appendChild(el("input", {
      id: `g-${key}`,
      type: "checkbox",
      ...(value ? { checked: "checked" } : {}),
      "data-key": key,
      "data-kind": "bool",
    }));
    return wrap;
  }
  const input = el("input", {
    id: `g-${key}`,
    type: typeof value === "number" ? "number" : "text",
    value: value == null ? "" : String(value),
    "data-key": key,
    "data-kind": value == null ? "null" : (typeof value === "number" ? "number" : "text"),
    placeholder: value == null ? "null" : "",
  });
  wrap.appendChild(input);
  return wrap;
}

function readSampling() {
  const box = document.getElementById("sampling-fields");
  const out = {};
  if (!box) return out;
  box.querySelectorAll("[data-key]").forEach((inp) => {
    const key = inp.dataset.key;
    const kind = inp.dataset.kind;
    if (HIDDEN_SAMPLING_KEYS.has(key)) return;
    if (kind === "bool") {
      out[key] = inp.checked;
      return;
    }
    const raw = inp.value.trim();
    if (raw === "") {
      out[key] = null;
      return;
    }
    if (kind === "number" || /^-?\d+(\.\d+)?$/.test(raw)) {
      const n = Number(raw);
      out[key] = Number.isFinite(n) ? n : raw;
      return;
    }
    if (raw === "true") out[key] = true;
    else if (raw === "false") out[key] = false;
    else if (raw === "null") out[key] = null;
    else out[key] = raw;
  });
  return out;
}

function metricCell(label, value, digits = 3) {
  let text = "—";
  if (typeof value === "boolean") text = value ? "是" : "否";
  else if (value != null && value !== "") text = fmtNum(value, digits);
  return el("div", { className: "metric-cell" }, [
    el("div", { className: "k", text: label }),
    el("div", { className: "v", text }),
  ]);
}

async function copyText(text, btn) {
  const done = () => {
    const old = btn.textContent;
    btn.textContent = "已复制";
    setTimeout(() => { btn.textContent = old; }, 1200);
  };
  try {
    await navigator.clipboard.writeText(text);
    done();
  } catch (_) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "readonly");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      done();
    } finally {
      ta.remove();
    }
  }
}

async function consumeNDJSON(response, onEvent) {
  if (!response.body) {
    const t = await response.text();
    for (const line of t.split("\n")) {
      const s = line.trim();
      if (!s) continue;
      try { onEvent(JSON.parse(s)); } catch (_) {}
    }
    return;
  }
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        onEvent(JSON.parse(line));
      } catch (_) {}
    }
  }
  const last = buf.trim();
  if (last) {
    try { onEvent(JSON.parse(last)); } catch (_) {}
  }
}

function previewText(text, lines = 4) {
  const parts = String(text || "").split("\n");
  if (parts.length <= lines) return text || "";
  return `${parts.slice(0, lines).join("\n")}\n…`;
}

function makeSampleCard(ev) {
  const seed = ev.seed ?? "—";
  const details = el("details", {
    className: "gen-sample card",
    "data-index": String(ev.index),
    "data-seed": String(seed),
  });
  const summary = el("summary", { className: "gen-sample-summary" });
  const head = el("div", { className: "gen-sample-head" });
  head.appendChild(el("span", { className: "gen-sample-title", text: `seed ${seed}` }));
  head.appendChild(el("span", { className: "muted", text: `nfe ${ev.nfe ?? "—"}` }));
  const copyBtn = el("button", { type: "button", className: "gen-copy secondary", text: "复制" });
  const text = ev.completion || ev.text || "";
  copyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    copyText(text, copyBtn);
  });
  head.appendChild(copyBtn);
  summary.appendChild(head);
  summary.appendChild(el("pre", { className: "sample-text gen-sample-preview", text: previewText(text) }));
  details.appendChild(summary);
  details.appendChild(el("pre", { className: "sample-text gen-sample-body", text }));
  details.appendChild(el("p", {
    className: "muted gen-eval-pending",
    text: `seed ${seed} · 评测等待全部生成结束…`,
  }));
  return details;
}

function paintSampleEval(card, ev) {
  const pending = card.querySelector(".gen-eval-pending");
  if (pending) pending.remove();
  const old = card.querySelector(".gen-eval");
  if (old) old.remove();
  const m = ev.metrics || {};
  const wrap = el("div", { className: "gen-eval" });
  wrap.appendChild(el("div", { className: "metrics-grid" }, [
    metricCell("seed", ev.seed, 0),
    metricCell("ppl", m.ppl),
    metricCell("gen-ppl", m.gen_ppl),
    metricCell("entropy", m.entropy),
    metricCell("gen-uniq", m.gen_uniq, 1),
    metricCell("src-entropy", m.src_entropy),
    metricCell("dist-1", m.dist1),
    metricCell("tokens", m.n_tokens, 0),
    metricCell("nonempty", m.nonempty),
    metricCell("seq-rep-4", m.seq_rep_4),
    metricCell("accept@human", m.accept_human),
  ]));
  card.appendChild(wrap);
}

function paintBatchSummary(batch, ev) {
  const old = batch.querySelector(".gen-batch-summary");
  if (old) old.remove();
  const m = ev.metrics || {};
  const wrap = el("div", { className: "card gen-batch-summary" });
  wrap.appendChild(el("h3", { text: "本批汇总" }));
  wrap.appendChild(el("div", { className: "metrics-grid" }, [
    metricCell("ppl", m.ppl),
    metricCell("gen-ppl", m.gen_ppl),
    metricCell("entropy", m.entropy),
    metricCell("gen-uniq", m.gen_uniq_mean, 1),
    metricCell("src-entropy", m.mean_src_entropy),
    metricCell("dist-1", m.mean_dist1),
    metricCell("nonempty", m.nonempty_frac),
    metricCell("median-rep", m.median_rep),
    metricCell("accept@human", m.accept_at_human),
    metricCell("nonword%", m.nonword_word_pct, 1),
    metricCell("ema", m.use_ema),
  ]));
  if (m.gpt2_skipped) {
    wrap.appendChild(el("p", { className: "muted", text: m.gpt2_reason || "未计算 gen-ppl" }));
  }
  batch.prepend(wrap);
}

async function refreshGpu(slot) {
  try {
    const g = await fetchJSON("/api/generate/gpu");
    const used = g.used_gib;
    const ok = !!g.ok;
    slot.className = ok ? "muted gpu-ok" : "muted gpu-bad";
    if (!g.gpus?.length) {
      slot.textContent = g.reason || "没有 GPU";
      return;
    }
    const gpu0 = g.gpus[0];
    slot.textContent = ok
      ? `GPU0 ${gpu0.name} · 占用 ${fmtNum(used, 2)} / ${fmtNum(gpu0.total_gib, 1)} GiB（门槛 < ${g.limit_gib} GiB）`
      : (g.reason || "显存检查未通过");
  } catch (e) {
    slot.className = "muted gpu-bad";
    slot.textContent = e.message;
  }
}

async function renderStudio(model, hash, ckpt, runQ) {
  showView("studio");
  const listed = findRun(model, hash, runQ);
  const run = listed?.run || runQ;
  const back = document.getElementById("studio-back");
  if (back) back.href = href({ model, hash, run });
  const sec = document.getElementById("studio-section");
  sec.innerHTML = "<p class='muted'>加载默认生成参数…</p>";
  if (!run) {
    sec.innerHTML = "<p class='muted'>找不到对应 run</p>";
    return;
  }
  let defaults;
  try {
    defaults = await fetchJSON(`/api/generate/defaults?run=${encodeURIComponent(run)}&profile=generate`);
  } catch (e) {
    sec.innerHTML = `<p class="muted">加载默认参数失败：${e.message}</p>`;
    return;
  }

  sec.innerHTML = "";
  sec.appendChild(el("h2", { style: "display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap" }, [
    kindBadge("lm"),
    el("span", { text: `${defaults.model || model} · ${ckpt}` }),
  ]));
  sec.appendChild(el("div", { className: "muted", text: `${run} · ${hash}`, style: "margin-bottom:0.75rem" }));

  const form = el("div", { className: "card generate-form" });
  form.appendChild(el("h3", { text: "生成参数" }));
  const row1 = el("div", { className: "form-row" });
  [
    ["num_tokens", defaults.num_tokens, "序列长", "8", "2048"],
    ["num_samples", defaults.num_samples ?? 1, "条数", "1", "16"],
    ["seed", defaults.seed, "seed", "0", ""],
  ].forEach(([id, val, label, min, max]) => {
    const w = el("div");
    w.appendChild(el("label", { text: label, for: `g-${id}` }));
    const attrs = { id: `g-${id}`, type: "number", value: String(val) };
    if (min !== "") attrs.min = min;
    if (max !== "") attrs.max = max;
    w.appendChild(el("input", attrs));
    row1.appendChild(w);
  });
  form.appendChild(row1);
  form.appendChild(el("p", {
    className: "muted",
    text: "内部仍逐条生成（seed、seed+1、…）；全部生成完再统一评测，避免反复加载 gpt2。生成完一条立刻显示文本。",
  }));

  const sampTitle = el("h3", { text: "采样（来自 generate YAML）", style: "margin-top:0.75rem" });
  form.appendChild(sampTitle);
  const samp = el("div", { id: "sampling-fields", className: "form-row" });
  samplingEntries(defaults.sampling).forEach(([k, v]) => samp.appendChild(fieldInput(k, v)));
  form.appendChild(samp);

  const promptWrap = el("div", { id: "prompt-wrap" });
  if (!defaults.supports_prefix) promptWrap.classList.add("hidden");
  promptWrap.appendChild(el("label", { text: "前缀（AR 续写；无条件生成模型不显示）", for: "g-prompt" }));
  promptWrap.appendChild(el("textarea", { id: "g-prompt", placeholder: "Once upon a time" }));
  form.appendChild(promptWrap);

  const gpuSlot = el("p", { className: "muted", id: "gpu-slot", text: "检查显存…" });
  const actions = el("div", { className: "generate-actions" });
  const btn = el("button", { type: "button", id: "g-run", text: "生成" });
  const status = el("span", { className: "muted", id: "g-status" });
  actions.appendChild(btn);
  actions.appendChild(status);
  form.appendChild(gpuSlot);
  form.appendChild(actions);
  sec.appendChild(form);

  const result = el("div", { id: "g-result" });
  sec.appendChild(result);

  refreshGpu(gpuSlot);

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    status.textContent = "检查显存并生成中…";
    await refreshGpu(gpuSlot);
    const seedEl = document.getElementById("g-seed");
    const seed = Number(seedEl?.value || 42);
    const nWanted = Math.max(1, Math.min(16, Number(document.getElementById("g-num_samples")?.value || 1)));
    const batch = el("div", { className: "gen-batch" });
    result.prepend(batch);
    const cards = new Map();
    let nextSeed = seed;
    let finished = false;
    try {
      const body = {
        run,
        checkpoint: ckpt,
        profile: "generate",
        num_tokens: Number(document.getElementById("g-num_tokens")?.value || 1024),
        num_samples: nWanted,
        seed,
        prompt: promptWrap.classList.contains("hidden") ? null : (document.getElementById("g-prompt")?.value || null),
        sampling: readSampling(),
      };
      const resp = await fetch("/api/generate/run", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || resp.statusText);
      }
      await consumeNDJSON(resp, (ev) => {
        if (!ev || typeof ev !== "object") return;
        if (ev.type === "status") {
          status.textContent = ev.message || "生成中…";
          return;
        }
        if (ev.type === "sample") {
          const card = makeSampleCard(ev);
          cards.set(ev.index, card);
          if (ev.seed != null) cards.set(`seed:${ev.seed}`, card);
          batch.appendChild(card);
          nextSeed = Number(ev.seed) + 1;
          if (seedEl) seedEl.value = String(nextSeed);
          status.textContent = `已生成 ${(ev.index ?? 0) + 1}/${ev.n ?? nWanted}（seed ${ev.seed}）`;
          return;
        }
        if (ev.type === "eval") {
          const card = (ev.seed != null && cards.get(`seed:${ev.seed}`)) || cards.get(ev.index);
          if (card) paintSampleEval(card, ev);
          status.textContent = `评测 seed ${ev.seed ?? ev.index ?? "—"}`;
          return;
        }
        if (ev.type === "done") {
          finished = true;
          if (ev.seed_next != null && seedEl) seedEl.value = String(ev.seed_next);
          nextSeed = ev.seed_next ?? nextSeed;
          paintBatchSummary(batch, ev);
          status.textContent = `完成 ${nWanted} 条 · 下次 seed ${nextSeed} · 已释放显存`;
          return;
        }
        if (ev.type === "error") {
          finished = true;
          status.textContent = ev.error || "生成失败";
          if (ev.log_tail) {
            batch.appendChild(el("pre", { className: "sample-text", text: ev.log_tail, style: "max-height:none" }));
          } else {
            batch.appendChild(el("p", { className: "muted", text: ev.error || "生成失败" }));
          }
        }
      });
      if (!finished) {
        status.textContent = cards.size
          ? `流已结束（已生成 ${cards.size} 条）· 下次 seed ${nextSeed}`
          : "流已结束，没有样本";
      }
    } catch (e) {
      status.textContent = e.message;
      batch.appendChild(el("p", { className: "muted", text: e.message }));
    } finally {
      btn.disabled = false;
      refreshGpu(gpuSlot);
    }
  });
}

async function main() {
  setActiveNav("generate");
  let role = "local";
  try {
    const inst = await fetchJSON("/api/instance");
    role = inst?.role === "remote" ? "remote" : "local";
  } catch (_) {}
  if (role !== "local") {
    showView("remote");
    return;
  }

  const model = qs("model");
  const hash = qs("hash");
  const ckpt = qs("ckpt");
  const runQ = qs("run");

  try {
    const data = await fetchJSON("/api/runs");
    window.__runs = (data.runs || []).filter((r) => r.variant === "full");
  } catch (e) {
    const sec = document.getElementById("model-section");
    if (sec) sec.innerHTML = `<p class="muted">加载失败：${e.message}</p>`;
    return;
  }

  if (model && hash && ckpt) {
    await renderStudio(model, hash, ckpt, runQ);
  } else if (model && hash) {
    await renderCkpts(model, hash, runQ);
  } else if (model) {
    renderHashes(model);
  } else {
    renderModels();
  }
}

main().catch((e) => {
  showView("model");
  const sec = document.getElementById("model-section");
  if (sec) sec.innerHTML = `<p class="muted">加载失败：${e.message}</p>`;
});
