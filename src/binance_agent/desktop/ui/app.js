"use strict";
const $ = id => document.getElementById(id);
const readLocal = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (_) { return fallback; } };
const writeLocal = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* History remains on the server. */ } };
const uid = () => crypto.randomUUID();
const state = {
  config: null, mode: readLocal("product.mode", "report"), view: "research",
  drafts: readLocal("product.drafts", { report: { symbol: "BTCUSDT", market: "fapi" }, topic: { symbol: "BTCUSDT", market: "fapi", question: "", linked: false } }),
  recent: readLocal("product.recent", []), catalog: {}, catalogError: {},
  session: null, selected: null, report: null, source: null, selection: 0, resultVersion: 0, refreshVersion: 0,
  starting: false, followLive: false, refreshTimer: null, messageCommand: null,
};
const visible = (id, yes) => { $(id).hidden = !yes; };
function node(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") n.className = value;
    else if (key === "onClick") n.addEventListener("click", value);
    else if (key === "text") n.textContent = value;
    else if (key === "checked") n.checked = value;
    else n.setAttribute(key, value);
  });
  children.flat().filter(x => x !== null && x !== undefined).forEach(c => n.append(c instanceof Node ? c : document.createTextNode(String(c))));
  return n;
}
function safeHttpUrl(value) { try { const u = new URL(value); return ["https:", "http:"].includes(u.protocol) ? u.href : ""; } catch (_) { return ""; } }
async function api(path, body, options = {}) {
  const r = await fetch(path, { method: body === undefined ? "GET" : "POST", headers: body === undefined ? {} : { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body), ...options });
  if (!r.ok) {
    let error; try { error = (await r.json()).detail; } catch (_) { /* Generic error below. */ }
    if (Array.isArray(error)) error = error.map(x => x.msg).join("；");
    throw new Error(typeof error === "string" ? error : `请求未完成（${r.status}），请稍后重试`);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}
let toastTimer;
function toast(text) { $("toast").textContent = text; visible("toast", true); clearTimeout(toastTimer); toastTimer = setTimeout(() => visible("toast", false), 4500); }
function inputError(text = "") { $("input-error").textContent = text; visible("input-error", !!text); }
function settingsMessage(text) { $("settings-message").textContent = text; }
function scopeText(target, legacySymbol) { return target?.kind === "asset" ? `${target.symbol} · ${target.market === "api" ? "现货" : "合约"}` : target?.kind === "independent" ? "独立专题" : `${legacySymbol || ""} · 历史研究`; }
function timeText(value) { if (!value) return "时点未知"; const d = new Date(value); return Number.isNaN(d.getTime()) ? value || "时点未知" : d.toLocaleString("zh-CN", { hour12: false }); }
function switchView(view) {
  state.view = view; window.scrollTo?.(0, 0);
  ["research", "history", "settings"].forEach(v => visible(`${v}-view`, v === view));
  document.querySelectorAll("[data-view]").forEach(b => b.getAttribute("data-view") === view ? b.setAttribute("aria-current", "page") : b.removeAttribute("aria-current"));
  if (view === "history") loadHistory().catch(e => toast(e.message));
}
function saveDraft() {
  const d = state.drafts[state.mode];
  d.symbol = $("symbol").value; d.market = $("market").value;
  if (state.mode === "topic") { d.question = $("topic-question").value; d.linked = $("link-asset").checked; }
  writeLocal("product.drafts", state.drafts);
}
function setMode(mode, save = true) {
  if (save) { saveDraft(); visible("research-form", true); }
  state.mode = mode === "topic" ? "topic" : "report";
  writeLocal("product.mode", state.mode);
  const d = state.drafts[state.mode];
  $("symbol").value = d.symbol || ""; $("market").value = d.market || "fapi";
  $("topic-question").value = state.drafts.topic.question || "";
  $("link-asset").checked = !!state.drafts.topic.linked;
  $("mode-report").setAttribute("aria-pressed", String(state.mode === "report"));
  $("mode-topic").setAttribute("aria-pressed", String(state.mode === "topic"));
  visible("topic-fields", state.mode === "topic");
  $("generate").textContent = state.mode === "report" ? "生成研报" : "研究这个话题";
  inputError(); updateScope();
}
function isAsset() { return state.mode === "report" || $("link-asset").checked; }
function selectedAsset() {
  const raw = $("symbol").value.trim().toLowerCase();
  const rows = state.catalog[$("market").value] || [];
  return rows.find(x => x.symbol.toLowerCase() === raw) || rows.find(x => x.base_asset.toLowerCase() === raw && x.quote_asset === "USDT") || rows.find(x => x.name.toLowerCase() === raw || x.name.toLowerCase().split(" / ").includes(raw));
}
function updateScope() {
  const asset = isAsset(); visible("asset-fields", asset); visible("recent-assets", asset);
  const market = $("market").value;
  visible("catalog-retry", asset && !!state.catalogError[market]);
  if (!asset) { $("target-preview").textContent = "独立研究 · 无需选择币种"; return; }
  $("symbol-options").replaceChildren(...(state.catalog[market] || []).map(x => node("option", {value:x.symbol, label:`${x.name} · ${x.base_asset}/${x.quote_asset}`})));
  const selected = selectedAsset();
  $("target-preview").textContent = selected ? `${selected.symbol} · ${market === "api" ? "现货" : "合约 USDⓈ-M"}` : state.catalogError[market] || "选择实际交易对后开始研究";
  if (!state.catalog[market]) loadCatalog(market).catch(() => {});
  renderRecent();
}
async function loadCatalog(market, force = false) {
  if (state.catalog[market] && !force) return;
  if (state.catalogError[market] === "正在读取币种…") return;
  state.catalogError[market] = "正在读取币种…";
  try { const result = await api(`/symbols?market=${encodeURIComponent(market)}`); state.catalog[market] = result.symbols; delete state.catalogError[market]; }
  catch (e) { state.catalogError[market] = e.message; }
  if ($("market").value !== market || !isAsset()) return;
  $("symbol-options").replaceChildren(...(state.catalog[market] || []).map(x => node("option", { value: x.symbol, label: `${x.name} · ${x.base_asset}/${x.quote_asset}` })));
  const chosen = selectedAsset();
  $("target-preview").textContent = chosen ? `${chosen.symbol} · ${market === "api" ? "现货" : "合约 USDⓈ-M"}` : state.catalogError[market] || "请选择有效交易对";
  visible("catalog-retry", isAsset() && !!state.catalogError[market]);
}
function renderRecent() {
  $("recent-assets").replaceChildren(...state.recent.slice(0, 5).map(x => node("button", { type: "button", onClick: () => { $("symbol").value = x.symbol; $("market").value = x.market; saveDraft(); loadCatalog(x.market).then(updateScope); } }, `${x.symbol} · ${x.market === "api" ? "现货" : "合约"}`)));
}
function modelReady() {
  if (state.config?.profile === "demo") return true;
  const e = state.config?.llm?.endpoints?.find(x => x.name === state.config.llm.active);
  if (!e?.model) return false;
  let local = false; try { local = ["localhost", "127.0.0.1", "[::1]"].includes(new URL(e.base_url).hostname); } catch (_) { /* Invalid endpoint. */ }
  return !!e.has_key || local;
}
async function startResearch(event) {
  event?.preventDefault(); if (state.starting) return;
  saveDraft(); inputError();
  let target = { kind: "independent" };
  if (state.mode === "topic" && !$("topic-question").value.trim()) { inputError("写下你想研究的问题。"); $("topic-question").focus(); return; }
  if (isAsset()) {
    if (!$("symbol").value.trim()) { inputError("请选择一个币种。"); $("symbol").focus(); return; }
    const asset = selectedAsset();
    if (!asset) { inputError(state.catalogError[$("market").value] || "请选择目录中的有效币种或交易对。"); $("symbol").focus(); return; }
    target = { kind: "asset", symbol: asset.symbol, market: $("market").value };
  }
  if (!modelReady()) { switchView("settings"); settingsMessage("先连接模型，研究草稿已保留。"); return; }
  const payload = { mode: state.mode, target, question: state.mode === "topic" ? $("topic-question").value.trim() : "" };
  const previous = readLocal("product.pending-new", null);
  const command = previous?.signature === JSON.stringify(payload) ? previous : { signature: JSON.stringify(payload), createId: uid(), turnId: uid() };
  writeLocal("product.pending-new", command);
  const selection = state.selection;
  state.starting = true; $("generate").disabled = true;
  try {
    const session = await api("/research/sessions", { ...payload, request_id: command.createId });
    const run = await api(`/research/sessions/${encodeURIComponent(session.session_id)}/turns`, { action: "initial", request_id: command.turnId });
    writeLocal("product.pending-new", null);
    if (target.kind === "asset") { state.recent = [target, ...state.recent.filter(x => x.symbol !== target.symbol || x.market !== target.market)].slice(0, 5); writeLocal("product.recent", state.recent); renderRecent(); }
    if (selection === state.selection && state.view === "research") await openSession(session.session_id, run.run_id, true);
    else toast("新研究已开始，可在历史中打开。");
  } catch (e) { inputError(e.message); }
  finally { state.starting = false; $("generate").disabled = false; }
}
async function openSession(id, runId = null, followLive = false) {
  const generation = ++state.selection;
  state.resultVersion++; state.source?.close(); state.source = null;
  const session = await api(`/research/sessions/${encodeURIComponent(id)}`);
  if (generation !== state.selection) return;
  state.session = session; state.selected = runId || session.active_run_id || session.runs.at(-1)?.run_id || null; state.followLive = followLive; state.report = null;
  writeLocal("product.session", id); writeLocal("product.selected", state.selected);
  switchView("research"); visible("empty-report", false); visible("workspace", true); visible("research-form", false); visible("new-research-toggle", true);
  $("report-content").replaceChildren(); renderSession(); await loadSelectedReport(); subscribe(generation);
}
function currentRun() { return state.session?.runs.find(r => r.run_id === state.selected); }
function renderSession() {
  const s = state.session; if (!s) return;
  $("session-title").textContent = s.title || s.root_question || "历史研究";
  $("session-scope").textContent = scopeText(s.target, s.symbol);
  const options = s.runs.map((r, i) => node("option", { value: r.run_id }, `第 ${i + 1} 轮 · ${statusLabel(r.status)}`));
  $("turn-select").replaceChildren(...options); $("turn-select").value = state.selected || "";
  const r = currentRun(); const active = s.runs.find(x => x.run_id === s.active_run_id);
  $("run-status").textContent = r ? `${statusLabel(r.status)}${r.created_at ? ' · ' + timeText(r.created_at) : ''}` : "尚未开始研究";
  const selectedLive = !active || active.run_id === state.selected;
  visible("reading-old", !selectedLive); visible("return-live", !selectedLive);
  visible("stop-run", !!active); visible("resume-run", active?.status === "interrupted");
  $("update-report").textContent = s.mode === "topic" ? "更新当前话题" : "更新当前研报";
  $("update-report").disabled = !!active || state.starting || s.status !== "open";
  visible("followup-form", s.status === "open" && (!active || selectedLive));
  $("followup-label").textContent = active?.status === "running" ? "运行中补充" : "继续追问";
  $("send-followup").textContent = active?.status === "running" ? "补充到本轮" : "提交追问";
  $("send-followup").disabled = !!active && active.status !== "running";
  const messages = s.messages.filter(m => m.run_id === state.selected);
  $("message-status").replaceChildren(...messages.map(m => node("p", {}, `${m.status === "incorporated" ? "已纳入" : m.status === "not_processed" ? "本轮未纳入" : "已接收"}：${m.payload?.content || "补充信息"}`)));
  $("run-error").textContent = r?.status === "interrupted" ? "本轮研究已保存。恢复会沿用原预算和截止时间；配置不兼容时可停止本轮后重新研究。" : "";
  visible("run-error", r?.status === "interrupted");
  visible("export-actions", r?.status === "completed" && !!state.report);
}
function statusLabel(value) { return ({ running: "正在研究", stopping: "正在保存并停止", interrupted: "研究中断，可恢复", completed: "已完成", cancelled: "已停止", failed: "未完成" })[value] || value; }
function subscribe(generation) {
  if (!state.session || state.session.status !== "open") return;
  const id = state.session.session_id;
  const cursor = readLocal(`product.cursor.${id}`, 0);
  const source = new EventSource(`/research/sessions/${encodeURIComponent(id)}/events?after=${Number(cursor) || 0}`); state.source = source;
  const receive = event => {
    if (generation !== state.selection) return;
    if (event.lastEventId) writeLocal(`product.cursor.${id}`, Number(event.lastEventId));
    if (["final", "interrupted", "cancelled", "failed", "run", "message"].includes(event.type)) {
      clearTimeout(state.refreshTimer); state.refreshTimer = setTimeout(() => refreshSession(generation).catch(e => toast(e.message)), 80);
    }
  };
  ["final", "interrupted", "cancelled", "failed", "run", "message", "agent", "node", "research_progress", "budget"].forEach(type => source.addEventListener(type, receive));
  source.onerror = () => { if (generation === state.selection && state.session?.active_run_id) $("run-status").textContent = "连接暂时断开，正在重连；后台研究不会重复启动。"; };
  source.onopen = () => { if (generation === state.selection) refreshSession(generation).catch(() => {}); };
}
async function refreshSession(generation = state.selection) {
  const id = state.session?.session_id; if (!id) return;
  const refresh = ++state.refreshVersion;
  const session = await api(`/research/sessions/${encodeURIComponent(id)}`);
  if (generation !== state.selection || refresh !== state.refreshVersion || state.session?.session_id !== id) return;
  state.session = session;
  if (state.followLive) state.selected = session.active_run_id || session.runs.at(-1)?.run_id;
  renderSession(); await loadSelectedReport();
}
async function loadSelectedReport() {
  const run = currentRun(); const id = state.selected; const generation = ++state.resultVersion;
  if (!id || run?.status !== "completed") { state.report = null; visible("export-actions", false); $("report-content").replaceChildren(node("p", { class: "muted" }, run?.status === "running" ? "正在收集资料与核验依据，完成后结论将显示在这里。" : "这轮还没有已完成的研报。")); return; }
  try {
    const result = await api(`/runs/${encodeURIComponent(id)}/report`);
    if (generation !== state.resultVersion || id !== state.selected) return;
    state.report = result.report; renderReport(result.report, id); visible("export-actions", true);
  } catch (e) { if (generation === state.resultVersion) { $("report-content").replaceChildren(node("p", { class: "error" }, e.message)); } }
}
function renderReport(report, runId = null) {
  const article = $("report-content"); const completeness = ({ complete: "完整", partial: "部分完成", insufficient_evidence: "证据不足" })[report.completeness];
  const children = [node("div", { class: "report-meta" }, node("span", { class: "badge" }, completeness), node("span", {}, `分析时点 · ${timeText(report.analysis_at)}`))];
  if (report.basis?.length) children[0].append(node("button", {class:"text-button", onClick:()=>$("report-basis")?.scrollIntoView({block:"start"})}, `查看依据 · ${report.basis.length} 条`));
  if (report.synthetic) children.push(node("p", { class: "notice" }, "合成示例 · 不联网、不消耗模型额度，不代表真实市场或模型效果。"));
  children.push(node("p", { class: "report-summary" }, report.summary));
  if (report.risks?.length) children.push(node("div", { class: "risk-strip" }, node("strong", {}, "重要风险"), node("ul", {}, report.risks.map(x => node("li", {}, x)))));
  const addList = (title, rows) => { if (rows?.length) children.push(node("section", { class: "report-section" }, node("h2", {}, title), node("ul", {}, rows.map(x => node("li", {}, x))))); };
  addList("建议与条件", [...(report.recommendations || []), ...(report.conditions || [])]);
  const advice = report.advice;
  if (advice?.decision_type === "trade") {
    children.push(node("p", {class:"muted"}, `条件性方向 · ${advice.direction === "long" ? "偏多" : "偏空"}`));
    children.push(node("div", { class: "trade-levels" }, [["入场区间", advice.entry_zone], ["止损条件", advice.stop_loss], ["目标", advice.targets?.join(" / ")]].map(([label, value]) => node("div", {}, node("small", {}, label), value || "—"))));
  }
  if (report.basis?.length) children.push(node("section", { class: "report-section", id:"report-basis" }, node("h2", {}, "主要依据"), report.basis.map(f => node("div", { class: "basis-item" }, node("p", {}, f.statement), f.quote ? node("details", {}, node("summary", {}, "查看引用"), node("blockquote", {}, f.quote)) : null))));
  const sourceRows = (report.sources || []).map(s => {
    const href = safeHttpUrl(s.url);
    return node("div", { class: "source-row" }, href ? node("a", { href, target: "_blank", rel: "noopener noreferrer" }, s.title || s.source || "来源") : node("span", {}, s.title || s.source || "来源"), runId && s.evidence_id ? node("button", { onClick: () => openEvidence(runId, s.evidence_id) }, "阅读已保存原文") : null, node("span", { class: "muted" }, `${s.source || ""} · 发布 ${timeText(s.published_at)} · 获取 ${timeText(s.fetched_at)}`));
  });
  children.push(node("section", { class: "report-section" }, node("h2", {}, "来源与缺口"), sourceRows.length ? sourceRows : node("p", { class: "muted" }, "本轮没有可展示的已核验来源。"), (report.gaps || []).length ? node("ul", {}, report.gaps.map(x => node("li", {}, x))) : null));
  if (report.coverage?.length) children.push(node("details", {}, node("summary", {}, "查看资料覆盖"), node("ul", {}, report.coverage.map(x => node("li", {}, `${x.source} · ${x.status}`)))));
  children.push(node("p", { class: "disclaimer" }, report.disclaimer || "")); article.replaceChildren(...children);
}
async function showExample() {
  try { const { report } = await api("/reports/example"); state.source?.close(); state.selection++; state.resultVersion++; state.session = null; state.selected = null; state.report = report; visible("workspace", true); visible("empty-report", false); $("session-title").textContent = report.title; $("session-scope").textContent = "合成示例"; $("turn-select").replaceChildren(); $("run-status").textContent = "示例，不保存为真实研究"; ["followup-form", "export-actions", "resume-run", "stop-run", "return-live", "reading-old", "run-error"].forEach(id => visible(id, false)); $("update-report").disabled = true; renderReport(report); } catch (e) { toast(e.message); }
}
async function openEvidence(runId, evidenceId) {
  const dialog = $("evidence-dialog"); $("evidence-title").textContent = "读取来源原文…"; $("evidence-text").textContent = ""; $("evidence-meta").textContent = ""; dialog.showModal();
  try { const result = await api(`/runs/${encodeURIComponent(runId)}/evidence/${encodeURIComponent(evidenceId)}`); const d = result.document || result; $("evidence-title").textContent = d.title || "已保存原文"; $("evidence-meta").textContent = `${d.source || ""} · ${timeText(d.fetched_at)}`; $("evidence-text").textContent = d.text || "原文未保存或不可用"; } catch (e) { $("evidence-text").textContent = e.message; }
}
async function continueTurn(action, question = "") {
  if (!state.session || state.starting) return;
  const selection = state.selection; const id = state.session.session_id; const signature = JSON.stringify({ id, action, question });
  let command = readLocal("product.pending-turn", null); if (command?.signature !== signature) command = { signature, request_id: uid() };
  writeLocal("product.pending-turn", command); state.starting = true; renderSession();
  try { const run = await api(`/research/sessions/${encodeURIComponent(id)}/turns`, { action, question, request_id: command.request_id }); writeLocal("product.pending-turn", null); if (selection === state.selection && state.view === "research") { $("followup-question").value = ""; await openSession(id, run.run_id, true); } else toast("新轮次已开始，可在历史中打开。"); }
  catch (e) { toast(e.message); } finally { state.starting = false; renderSession(); }
}
async function sendFollowup(event) {
  event.preventDefault(); const content = $("followup-question").value.trim();
  if (!content) { toast("请输入需要研究或补充的问题。"); $("followup-question").focus(); return; }
  const active = state.session?.runs.find(r => r.run_id === state.session.active_run_id);
  if (active?.status === "running") {
    const signature = JSON.stringify({ run: active.run_id, content });
    if (state.messageCommand?.signature !== signature) state.messageCommand = { signature, request_id: uid() };
    const selection = state.selection;
    $("send-followup").disabled = true;
    try { await api(`/research/sessions/${encodeURIComponent(state.session.session_id)}/messages`, { run_id: active.run_id, content, request_id: state.messageCommand.request_id }); state.messageCommand = null; toast("已接收，处理后会标记为已纳入。"); if (selection === state.selection) { $("followup-question").value = ""; await refreshSession(); } }
    catch (e) { toast(e.message); } finally { $("send-followup").disabled = false; }
  } else await continueTurn("followup", content);
}
async function controlRun(action) {
  const rid = state.session?.active_run_id; if (!rid) return;
  const key = `product.control.${rid}.${action}`; let requestId = readLocal(key, null); if (!requestId) { requestId = uid(); writeLocal(key, requestId); }
  try { await api(`/runs/${encodeURIComponent(rid)}/${action}`, { request_id: requestId }); writeLocal(key, null); await refreshSession(); } catch (e) { toast(e.message); }
}
async function exportReport(copy) {
  const id = state.selected; if (!id || !state.report) return;
  try {
    if (copy) { const markdown = await api(`/runs/${encodeURIComponent(id)}/report.md`); await navigator.clipboard.writeText(markdown); $("export-status").textContent = "已复制当前版本"; }
    else if (window.pywebview?.api?.save_report) { const result = await window.pywebview.api.save_report(id); if (result.error) throw new Error(result.error); $("export-status").textContent = result.cancelled ? "已取消保存" : `已保存 ${result.name}`; }
    else { const markdown = await api(`/runs/${encodeURIComponent(id)}/report.md`); const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown;charset=utf-8" })); const a = node("a", { href: url, download: `研报-${id.slice(0, 8)}.md` }); a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); $("export-status").textContent = "已下载当前版本"; }
  } catch (e) { $("export-status").textContent = e.message; }
}
async function openLegacyReport(id) {
  state.source?.close(); const selection = ++state.selection; const resultVersion = ++state.resultVersion;
  const result = await api(`/runs/${encodeURIComponent(id)}/report`);
  if (selection !== state.selection || resultVersion !== state.resultVersion) return;
  state.session = null; state.selected = id; state.report = result.report;
  switchView("research"); visible("empty-report", false); visible("workspace", true); visible("research-form", false); visible("new-research-toggle", true);
  $("session-title").textContent = result.report.title; $("session-scope").textContent = "历史研究";
  $("turn-select").replaceChildren(); $("run-status").textContent = "已保存的旧版结果"; $("update-report").disabled = true;
  ["followup-form", "resume-run", "stop-run", "return-live", "reading-old", "run-error"].forEach(key => visible(key, false));
  renderReport(result.report, id); visible("export-actions", true);
}
async function loadHistory() {
  const [sessions, records] = await Promise.all([api("/research/sessions"), api("/runs?limit=100")]);
  const list = $("history-list"); const rows = sessions.sessions || [];
  const children = rows.map(s => {
    const latest = s.runs.at(-1); return node("div", { class: "history-item" }, node("div", { class: "history-description" }, node("h2", {}, s.title || s.root_question || "历史研究"), node("p", { class: "muted" }, `${scopeText(s.target, s.symbol)} · ${s.runs.length} 轮 · ${statusLabel(latest?.status || s.status)}`)), node("button", { onClick: () => openSession(s.session_id).catch(e => toast(e.message)) }, "打开"));
  });
  const legacy = (Array.isArray(records) ? records : records.runs || []).filter(r => !r.session_id);
  legacy.forEach(r => children.push(node("div", { class: "history-item" }, node("div", { class: "history-description" }, node("h2", {}, `${r.symbol || ""} 历史研究`), node("p", { class: "muted" }, timeText(r.created_at))), node("button", { onClick: () => openLegacyReport(r.run_id).catch(e => toast(e.message)) }, "查看"))));
  list.replaceChildren(...(children.length ? children : [node("p", { class: "muted" }, "还没有研究记录。生成第一份研报后会自动保存在这里。") ]));
}
async function loadConfig() {
  state.config = await api("/config"); const cfg = state.config;
  visible("welcome", !modelReady()); $("connection-status").textContent = cfg.profile === "demo" ? "合成演示" : modelReady() ? "模型已配置" : "先连接模型";
  $("model-service").replaceChildren(...cfg.llm.endpoints.map(e => node("option", { value: e.name }, e.name)), node("option", { value: "__new__" }, "自定义兼容服务…"));
  $("model-service").value = cfg.llm.active || "__new__"; populateEndpoint();
  $("proxy").value = cfg.network.proxy || ""; $("research-window").value = String(cfg.params.time_window_hours);
  $("on-top").checked = !!cfg.desktop?.on_top; $("hotkey-enabled").checked = !!cfg.desktop?.hotkey_enabled;
  $("max-requests").value = cfg.research.max_requests || 100; $("deadline").value = cfg.research.deadline_s || 600;
  $("app-version").textContent = cfg.version || "";
  $("search-state").textContent = cfg.intel.keyed.some(x => x.source === "tavily" && x.has_key) ? "已连接" : "未配置（可选）";
  renderRoles(); renderSources();
}
function populateEndpoint() {
  const endpoint = state.config.llm.endpoints.find(e => e.name === $("model-service").value);
  visible("custom-service", !endpoint); $("endpoint-name").value = endpoint?.name || ""; $("endpoint-url").value = endpoint?.base_url || "";
  $("endpoint-key").value = ""; $("endpoint-model").value = endpoint?.model || ""; $("key-state").textContent = endpoint?.has_key ? "已保存，留空保留" : "";
  $("model-options").replaceChildren(...(endpoint?.models || []).map(m => node("option", { value: m })));
  $("structured-output").checked = !!endpoint?.structured_output; $("endpoint-concurrency").value = endpoint?.max_concurrency || 3;
  $("unified-model").checked = !endpoint || !state.config.llm.endpoints.some(e => e.has_key) || state.config.llm.agents.every(a => a.endpoint === endpoint.name && a.model === endpoint.model);
}
function endpointDraft() {
  const chosen = state.config.llm.endpoints.find(e => e.name === $("model-service").value);
  return { name: chosen?.name || $("endpoint-name").value.trim(), base_url: $("endpoint-url").value.trim(), model: $("endpoint-model").value.trim(), api_key: $("endpoint-key").value.trim() || null, structured_output: $("structured-output").checked, max_concurrency: Number($("endpoint-concurrency").value), use_for_all_roles: $("unified-model").checked };
}
async function saveModel(event) {
  event.preventDefault(); const body = endpointDraft();
  if (!body.name || !safeHttpUrl(body.base_url) || !body.model) { settingsMessage("请填写服务名称、有效 API 地址和模型名称。"); return; }
  try { await api("/config/llm/endpoint", body); if (!body.use_for_all_roles) await api("/config/llm-endpoint/switch", { endpoint: body.name }); $("endpoint-key").value = ""; await loadConfig(); settingsMessage(""); switchView("research"); toast("模型连接已保存，研究草稿已保留。"); } catch (e) { settingsMessage(e.message); }
}
async function fetchModels() {
  const body = endpointDraft(); if (!safeHttpUrl(body.base_url)) { settingsMessage("先填写有效的 API 地址。"); return; }
  $("fetch-models").disabled = true; settingsMessage("正在读取服务提供的模型列表…");
  try { const result = await api("/config/llm/models", { base_url: body.base_url, api_key: body.api_key, endpoint: $("model-service").value === "__new__" ? null : body.name }); const models = result.models || []; $("model-options").replaceChildren(...models.map(m => node("option", { value: typeof m === "string" ? m : m.id }))); if (!$("endpoint-model").value && models.length) $("endpoint-model").value = typeof models[0] === "string" ? models[0] : models[0].id; settingsMessage(models.length ? `连接成功，读到 ${models.length} 个模型。选择模型后保存。` : "服务未返回模型列表，可以手动输入模型名称。"); } catch (e) { settingsMessage(e.message); } finally { $("fetch-models").disabled = false; }
}
const roleNames = { research_supervisor: "研究主管", news_researcher: "资料研究", evidence_verifier: "证据核验", technical: "技术分析", funding: "资金分析", news_sentiment: "资讯分析", bull_debater: "多方分析", bear_debater: "空方分析", research_manager: "研究汇总", strategy_synth: "策略合成" };
function renderRoles() {
  $("role-models").replaceChildren(...state.config.llm.agents.filter(a => roleNames[a.agent]).map(a => {
    const service = node("select", { "aria-label": `${roleNames[a.agent]}服务` }, state.config.llm.endpoints.map(e => node("option", { value: e.name }, e.name))); service.value = a.endpoint;
    const model = node("input", { value: a.model, "aria-label": `${roleNames[a.agent]}模型` });
    return node("div", { class: "role-row" }, node("span", {}, roleNames[a.agent]), service, model, node("button", { onClick: async () => { try { await api("/config/llm/agent", { agent: a.agent, endpoint: service.value, model: model.value.trim() }); await loadConfig(); settingsMessage("角色模型已保存。"); } catch (e) { settingsMessage(e.message); } } }, "保存"));
  }));
}
function renderSources() {
  const cfg = state.config;
  $("source-settings").replaceChildren(...(cfg.intel.catalog || []).map(s => {
    const input = node("input", { type: "checkbox", checked: s.enabled, "aria-label": s.label || s.source });
    input.addEventListener("change", async () => { try { await api("/config/intel/source", { source: s.id || s.source, enabled: input.checked }); settingsMessage("数据来源已保存。"); } catch (e) { input.checked = !input.checked; settingsMessage(e.message); } });
    return node("div", { class: "source-setting" }, node("label", { class: "checkbox" }, input, s.label || s.source), node("span", { class: "muted" }, s.requires_key && !cfg.intel.keyed.some(k => k.source === s.source && k.has_key) ? "需要密钥" : ""));
  }));
  $("source-keys").replaceChildren(...cfg.intel.keyed.filter(k => k.source !== "tavily").map(k => {
    const input = node("input", { type: "password", autocomplete: "new-password", placeholder: k.has_key ? "已连接；留空保留" : "可选密钥", "aria-label": `${k.source}密钥` });
    return node("div", {}, node("label", {}, k.source), node("div", { class: "inline-field" }, input, node("button", { onClick: async () => { if (!input.value.trim()) return; try { await api("/config/intel/key", { source: k.source, api_key: input.value.trim() }); input.value = ""; await loadConfig(); settingsMessage("数据源连接已保存。"); } catch (e) { settingsMessage(e.message); } } }, "保存")));
  }));
}
async function saveSettings(event, path, body, message) { event.preventDefault(); try { await api(path, body); settingsMessage(message); } catch (e) { settingsMessage(e.message); } }
$("new-research-toggle").addEventListener("click", () => { visible("research-form", $("research-form").hidden); if (!$("research-form").hidden) (state.mode === "topic" ? $("topic-question") : $("symbol")).focus(); });
document.querySelectorAll("[data-view]").forEach(b => b.addEventListener("click", () => switchView(b.dataset.view)));
$("mode-report").addEventListener("click", () => setMode("report")); $("mode-topic").addEventListener("click", () => setMode("topic"));
$("link-asset").addEventListener("change", () => { saveDraft(); updateScope(); });
$("symbol").addEventListener("input", () => { saveDraft(); inputError(); updateScope(); });
$("market").addEventListener("change", () => { saveDraft(); loadCatalog($("market").value, true).then(updateScope); });
$("topic-question").addEventListener("input", saveDraft); $("catalog-retry").addEventListener("click", () => loadCatalog($("market").value, true));
$("research-form").addEventListener("submit", startResearch); $("connect-model").addEventListener("click", () => switchView("settings"));
$("show-example").addEventListener("click", showExample); $("empty-example").addEventListener("click", showExample);
$("back-to-draft").addEventListener("click", () => switchView("research")); $("refresh-history").addEventListener("click", () => loadHistory().catch(e => toast(e.message)));
$("turn-select").addEventListener("change", () => { state.selected = $("turn-select").value; state.followLive = false; state.report = null; writeLocal("product.selected", state.selected); renderSession(); loadSelectedReport(); });
$("return-live").addEventListener("click", () => { state.followLive = true; refreshSession(); });
$("update-report").addEventListener("click", () => continueTurn("update")); $("followup-form").addEventListener("submit", sendFollowup);
$("stop-run").addEventListener("click", () => controlRun("stop")); $("resume-run").addEventListener("click", () => controlRun("resume"));
$("copy-report").addEventListener("click", () => exportReport(true)); $("save-report").addEventListener("click", () => exportReport(false));
$("close-evidence").addEventListener("click", () => $("evidence-dialog").close());
$("edit-service-address").addEventListener("click", () => visible("custom-service", true));
$("model-service").addEventListener("change", populateEndpoint); $("model-form").addEventListener("submit", saveModel); $("fetch-models").addEventListener("click", fetchModels);
$("search-form").addEventListener("submit", async event => { event.preventDefault(); if (!$("search-key").value.trim()) { settingsMessage("填写 Tavily 密钥即可启用可选搜索。"); return; } try { await api("/config/intel/key", { source: "tavily", api_key: $("search-key").value.trim() }); $("search-key").value = ""; await loadConfig(); settingsMessage("搜索连接已保存。"); } catch (e) { settingsMessage(e.message); } });
$("research-window").addEventListener("change", async () => { try { await api("/config/params", { time_window_hours: Number($("research-window").value) }); settingsMessage("资料时间窗已保存。"); } catch (e) { settingsMessage(e.message); } });
$("network-form").addEventListener("submit", e => saveSettings(e, "/config/network", { proxy: $("proxy").value.trim() }, "代理已保存。"));
$("desktop-form").addEventListener("submit", e => saveSettings(e, "/config/desktop", { on_top: $("on-top").checked, hotkey_enabled: $("hotkey-enabled").checked }, "桌面偏好已保存，重启应用后生效。"));
$("limits-form").addEventListener("submit", e => saveSettings(e, "/config/research", { max_requests: Number($("max-requests").value), deadline_s: Number($("deadline").value) }, "研究上限已保存，后续新轮次生效。"));
$("window-min").addEventListener("click", () => window.pywebview?.api?.minimize()); $("window-close").addEventListener("click", () => window.pywebview?.api?.close());
window.__externalAnalyze = symbol => { switchView("research"); setMode("report"); $("symbol").value = String(symbol); saveDraft(); updateScope(); $("generate").focus(); };
window.addEventListener("beforeunload", () => { saveDraft(); state.source?.close(); });
(async function init() {
  setMode(state.mode, false);
  try { await loadConfig(); const id = readLocal("product.session", null); if (id) await openSession(id, readLocal("product.selected", null)); }
  catch (e) { toast(e.message); visible("welcome", true); }
})();
