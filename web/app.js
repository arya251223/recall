"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const api = async (path, opts) => {
  const r = await fetch(path, opts && { ...opts, headers: { "Content-Type": "application/json" }, body: opts.body && JSON.stringify(opts.body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
};
const toast = (msg) => { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 2800); };
const fmtMs = (v) => (v == null ? "–" : v < 10 ? v.toFixed(1) + " ms" : v < 1000 ? Math.round(v) + " ms" : (v / 1000).toFixed(2) + " s");
const STAGES = { guard: "Check", embed: "Embed", cache: "Cache", retrieve: "Retrieve", generate: "Generate", safety: "Safety" };
const VERDICT = { cache: "Answered from cache", llm: "Answered by the model", no_context: "Not in the knowledge base", blocked: "Blocked", error: "Model error" };
const EXAMPLES = ["How much does the Pro plan cost?", "What's the price of the Pro tier per user?", "How much does the Business plan cost?",
  "Can I get my money back on an annual plan?", "How long do deleted files stay recoverable?", "What is the capital of France?"];

function renderRoute(trace) {
  const ol = $("route"); ol.replaceChildren(); ol.classList.remove("pulse"); void ol.offsetWidth; ol.classList.add("pulse");
  trace.forEach((s, i) => {
    const li = el("li", s.status === "skipped" ? "" : s.status);
    li.style.setProperty("--i", i);
    li.append(STAGES[s.stage] || s.stage, el("span", "ms", s.status === "skipped" ? "skipped" : fmtMs(s.ms)));
    ol.append(li);
  });
}

function renderResult(r) {
  $("empty").hidden = true; $("result").hidden = false;
  renderRoute(r.trace);
  const v = $("verdict"); v.className = "verdict " + r.source;
  v.textContent = VERDICT[r.source] + (r.source === "cache" ? ` · similarity ${r.similarity.toFixed(2)} · ${fmtMs(r.total_ms)}` : ` · ${fmtMs(r.total_ms)}`);
  $("answer").textContent = r.answer;
  const rej = $("rejected"); rej.replaceChildren();
  const notes = [];
  if (r.source === "cache" && r.matched_query && r.matched_query !== r.question) notes.push(`Matched an earlier question: “${r.matched_query}”.`);
  (r.rejected || []).forEach((x) => notes.push(`Near-match “${x.query}” (similarity ${x.score}) was rejected: ${x.reason === "entity_conflict" ? "the questions differ on " + x.detail : "its answer came from different documents"}.`));
  if (r.flags && r.flags.redactions && r.flags.redactions.length) notes.push("Sensitive text was redacted: " + r.flags.redactions.join(", ") + ".");
  notes.forEach((n) => rej.append(el("div", null, n)));
  rej.hidden = !notes.length;
  const ul = $("sources"); ul.replaceChildren();
  (r.sources || []).forEach((s) => {
    const li = el("li"); li.append(el("div", "t", s.title));
    if (s.score != null) { const b = el("div", "bar"); const i = el("i"); i.style.width = Math.min(100, Math.max(2, s.score * 100)) + "%"; b.append(i); li.append(b); }
    li.append(el("div", null, s.text)); ul.append(li);
  });
  if (!(r.sources || []).length) ul.append(el("li", null, "No passages were used."));
}

async function ask(question) {
  const btn = $("ask-btn"); btn.disabled = true;
  try {
    renderResult(await api("/api/query", { method: "POST", body: { question, use_cache: $("use-cache").checked } }));
  } catch (e) { toast(e.message); } finally { btn.disabled = false; refresh(); }
}

function renderStats(s) {
  const c = s.config;
  const badges = $("badges"); badges.replaceChildren();
  [["embedder", c.embedder], ["model", c.llm], ["threshold", c.cache_threshold.toFixed(2)]].forEach(([k, v]) => { const b = el("span", "badge", k + " "); b.append(el("b", null, v)); badges.append(b); });
  const tiles = $("tiles"); tiles.replaceChildren();
  [["Requests", s.requests], ["Hit rate", Math.round(s.hit_rate * 100) + "%"], ["Model calls avoided", s.llm_calls_avoided], ["Tokens saved", s.tokens_saved.toLocaleString()]]
    .forEach(([k, v]) => { const d = el("div"); d.append(el("dt", null, k), el("dd", null, String(v))); tiles.append(d); });
  const lat = $("lat"); lat.replaceChildren();
  const mx = Math.max(s.avg_ms_cache || 0, s.avg_ms_llm || 0, 1);
  [["Cache", s.avg_ms_cache, "var(--cobalt)"], ["Model path", s.avg_ms_llm, "var(--plum)"]].forEach(([k, v, col]) => {
    const row = el("div"); const b = el("span", "b"); const i = el("i"); i.style.width = v ? Math.max(3, (v / mx) * 100) + "%" : "0"; i.style.background = col; b.append(i);
    row.append(el("span", null, k), b, el("span", null, fmtMs(v))); lat.append(row);
  });
  const strip = $("strip"); strip.replaceChildren();
  const rec = s.recent, top = Math.max(...rec.map((r) => r.ms), 1);
  rec.forEach((r) => { const i = el("i"); i.style.height = Math.max(6, (Math.log10(r.ms + 1) / Math.log10(top + 1)) * 100) + "%"; i.style.background = r.source === "cache" ? "var(--cobalt)" : r.source === "llm" ? "var(--plum)" : "var(--amber)"; i.title = `${r.source} · ${fmtMs(r.ms)} · ${r.q}`; strip.append(i); });
  const thr = $("thr"); if (document.activeElement !== thr) { thr.value = c.cache_threshold; $("thr-out").textContent = c.cache_threshold.toFixed(2); }
  $("g-entity").checked = c.entity_guard; $("g-consist").checked = c.consistency_check;
}

async function refresh() {
  try {
    renderStats(await api("/api/stats"));
    const [cache, docs] = await Promise.all([api("/api/cache"), api("/api/documents")]);
    $("cache-count").textContent = `(${cache.entries.length})`;
    const cl = $("cache-list"); cl.replaceChildren();
    if (!cache.entries.length) cl.append(el("li", null, "Nothing cached yet."));
    cache.entries.forEach((e) => { const li = el("li"); li.append(el("span", null, e.query), el("small", null, e.hits + (e.hits === 1 ? " hit" : " hits"))); cl.append(li); });
    $("kb-count").textContent = `(${docs.documents.length})`;
    const kl = $("kb-list"); kl.replaceChildren();
    docs.documents.forEach((d) => {
      const li = el("li"); const b = el("button", null, "remove"); b.type = "button"; b.setAttribute("aria-label", "Remove " + d.title);
      b.onclick = async () => { try { const r = await api("/api/documents/" + d.id, { method: "DELETE" }); toast(`Removed. ${r.cache_entries_invalidated} cached answer(s) dropped.`); } catch (e) { toast(e.message); } refresh(); };
      li.append(el("span", null, d.title), el("small", null, d.chunks + " chunks "), b); kl.append(li);
    });
  } catch (e) { /* server restarting */ }
}

async function setConfig(patch) { try { await api("/api/config", { method: "POST", body: patch }); } catch (e) { toast(e.message); } refresh(); }

$("ask-form").onsubmit = (e) => { e.preventDefault(); const q = $("q").value.trim(); if (q) ask(q); };
$("q").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask-form").requestSubmit(); } };
EXAMPLES.forEach((t) => { const b = el("button", "chip", t); b.type = "button"; b.onclick = () => { $("q").value = t; ask(t); }; $("chips").append(b); });
$("thr").oninput = (e) => { $("thr-out").textContent = (+e.target.value).toFixed(2); };
$("thr").onchange = (e) => setConfig({ cache_threshold: +e.target.value });
$("g-entity").onchange = (e) => setConfig({ entity_guard: e.target.checked });
$("g-consist").onchange = (e) => setConfig({ consistency_check: e.target.checked });
$("clear-cache").onclick = async () => { const r = await api("/api/cache/clear", { method: "POST", body: {} }); toast(`Cleared ${r.cleared} cached answer(s).`); refresh(); };
$("doc-form").onsubmit = async (e) => {
  e.preventDefault();
  try { const r = await api("/api/documents", { method: "POST", body: { title: $("doc-title").value, text: $("doc-text").value } }); toast(`Added “${r.title}”. ${r.cache_entries_invalidated} cached answer(s) dropped.`); $("doc-form").reset(); } catch (err) { toast(err.message); }
  refresh();
};
refresh(); setInterval(refresh, 5000);
