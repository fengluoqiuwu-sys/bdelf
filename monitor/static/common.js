const API = "";

export async function fetchJSON(url) {
  const r = await fetch(API + url);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

export async function sendJSON(url, method, body) {
  const r = await fetch(API + url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) return r.json();
  return null;
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

export function pct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

export function fmtNum(x, digits = 2) {
  if (x == null || x === "") return "—";
  const n = Number(x);
  if (Number.isNaN(n)) return String(x);
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return n.toFixed(digits);
}

export function qs(name) {
  return new URLSearchParams(location.search).get(name);
}

export function kindBadge(kind) {
  const k = kind === "latent" ? "latent" : "lm";
  return el("span", {
    className: `badge badge-kind badge-kind-${k}`,
    text: k === "latent" ? "Latent" : "LM",
  });
}

export function distinguishingKeys(items) {
  const keys = new Set();
  for (const r of items || []) {
    Object.keys(r.identity || {}).forEach((k) => keys.add(k));
  }
  const out = [];
  for (const k of keys) {
    const vals = new Set((items || []).map((r) => String((r.identity || {})[k] ?? "")));
    if (vals.size > 1) out.push(k);
  }
  return out;
}

export function formatRunDelta(r, keys) {
  const ident = r.identity || {};
  const MAX_ITEMS = 4;
  const MAX_LEN = 96;
  let parts;
  if (keys && keys.length) {
    parts = keys.map((k) => `${k}=${ident[k] ?? "—"}`);
  } else {
    parts = [ident.config, ident.dataset, ident.preprocess].filter(Boolean);
  }
  if (!parts.length) return { short: "默认配置", full: "" };
  const extra = parts.length > MAX_ITEMS;
  let short = parts.slice(0, MAX_ITEMS).join(" · ");
  if (extra) short += " …";
  if (short.length > MAX_LEN) short = short.slice(0, MAX_LEN - 1) + "…";
  return { short, full: parts.join(" · ") };
}

export function inferKind(model, hint) {
  if (hint === "lm" || hint === "latent") return hint;
  const m = String(model || "");
  if (m.startsWith("latent_") || m === "cola_vae" || m === "latent_t5" || m === "latent_vae") {
    return "latent";
  }
  return "lm";
}

const _lb = {
  scale: 1,
  x: 0,
  y: 0,
  src: "",
  name: "",
  dragging: false,
  px: 0,
  py: 0,
};

function _lbApply() {
  const img = document.getElementById("img-lb-img");
  if (!img) return;
  img.style.transform = `translate(${_lb.x}px, ${_lb.y}px) scale(${_lb.scale})`;
}

function _lbReset() {
  _lb.scale = 1;
  _lb.x = 0;
  _lb.y = 0;
  _lbApply();
}

function _lbZoom(delta) {
  const next = Math.min(8, Math.max(0.25, _lb.scale + delta));
  _lb.scale = next;
  _lbApply();
}

function closeImageLightbox() {
  const root = document.getElementById("img-lightbox");
  if (root) root.classList.add("hidden");
  document.body.style.overflow = "";
  document.removeEventListener("keydown", _lbOnKey);
}

function _lbOnKey(ev) {
  if (ev.key === "Escape") closeImageLightbox();
  if (ev.key === "+" || ev.key === "=") _lbZoom(0.25);
  if (ev.key === "-" || ev.key === "_") _lbZoom(-0.25);
}

async function _lbDownload() {
  const name = String(_lb.name || "image").replace(/[\\/:*?"<>|]+/g, "-");
  const file = /\.[a-z0-9]+$/i.test(name) ? name : `${name}.png`;
  try {
    const r = await fetch(_lb.src);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = file;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (_) {
    window.open(_lb.src, "_blank");
  }
}

function _ensureLightbox() {
  if (document.getElementById("img-lightbox")) return;
  const root = el("div", { id: "img-lightbox", className: "img-lightbox hidden" });
  const backdrop = el("div", { className: "img-lightbox-backdrop" });
  backdrop.addEventListener("click", closeImageLightbox);
  const box = el("div", { className: "img-lightbox-box" });
  box.addEventListener("click", (e) => e.stopPropagation());
  const title = el("span", { className: "img-lightbox-title", id: "img-lb-title" });
  const toolbar = el("div", { className: "img-lightbox-toolbar" }, [
    title,
    el("button", { type: "button", className: "secondary", text: "−", onclick: () => _lbZoom(-0.25) }),
    el("button", { type: "button", className: "secondary", text: "+", onclick: () => _lbZoom(0.25) }),
    el("button", { type: "button", className: "secondary", text: "适应", onclick: _lbReset }),
    el("button", { type: "button", text: "下载", onclick: _lbDownload }),
    el("button", { type: "button", className: "secondary", text: "关闭", onclick: closeImageLightbox }),
  ]);
  const img = el("img", { id: "img-lb-img", alt: "" });
  const stage = el("div", { className: "img-lightbox-stage", id: "img-lb-stage" }, [img]);
  stage.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    _lbZoom(ev.deltaY < 0 ? 0.15 : -0.15);
  }, { passive: false });
  stage.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    _lb.dragging = true;
    _lb.px = ev.clientX - _lb.x;
    _lb.py = ev.clientY - _lb.y;
    stage.classList.add("dragging");
    ev.preventDefault();
  });
  window.addEventListener("mousemove", (ev) => {
    if (!_lb.dragging) return;
    _lb.x = ev.clientX - _lb.px;
    _lb.y = ev.clientY - _lb.py;
    _lbApply();
  });
  window.addEventListener("mouseup", () => {
    _lb.dragging = false;
    stage.classList.remove("dragging");
  });
  box.appendChild(toolbar);
  box.appendChild(stage);
  root.appendChild(backdrop);
  root.appendChild(box);
  document.body.appendChild(root);
}

export function openImageLightbox({ src, name = "image" } = {}) {
  if (!src) return;
  _ensureLightbox();
  _lb.src = src;
  _lb.name = name;
  _lbReset();
  const img = document.getElementById("img-lb-img");
  const title = document.getElementById("img-lb-title");
  img.src = src;
  img.alt = name;
  if (title) title.textContent = name;
  document.getElementById("img-lightbox").classList.remove("hidden");
  document.body.style.overflow = "hidden";
  document.removeEventListener("keydown", _lbOnKey);
  document.addEventListener("keydown", _lbOnKey);
}

export function clickableImage(src, { name = "", alt = "", className = "", style = "" } = {}) {
  const attrs = { src, alt: alt || name, loading: "lazy", className: `img-thumb ${className}`.trim() };
  if (style) attrs.style = style;
  const img = el("img", attrs);
  img.addEventListener("click", () => openImageLightbox({ src, name: name || alt || "image" }));
  return img;
}

export function setActiveNav(id) {
  document.querySelectorAll("nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.nav === id);
  });
  paintInstanceBadge();
}

export async function paintInstanceBadge() {
  const slot = document.getElementById("instance-badge");
  if (!slot) return;
  let role = "local";
  try {
    const data = await fetchJSON("/api/instance");
    if (data?.role === "remote") role = "remote";
  } catch (_) {}
  slot.hidden = false;
  slot.textContent = role;
  slot.className = `badge badge-instance badge-instance-${role}`;
  slot.title = role === "remote" ? "远端实例" : "本机实例";
  document.querySelectorAll('nav a[data-nav="generate"]').forEach((a) => {
    a.classList.toggle("nav-locked", role === "remote");
    a.title = role === "remote" ? "Generate 仅本机可用" : "本机生成";
  });
}
