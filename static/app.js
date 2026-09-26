const $ = (q, el = document) => el.querySelector(q);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("en-US", { maximumFractionDigits: 3, minimumFractionDigits: 3 });
const COLORS = {
  honest: "#2f7d62",
  extremes_hackable: "#d4762c",
  extremes_unhackable: "#3d7ea6",
  oracle_upper: "#7d6aad",
  oracle_lower: "#b35454",
};
const PHASES = { search: "搜索", ramp: "推全", hold: "保持" };

const form = $("#setup");
let catalog = null;
let requestId = 0;

async function api(path, body) {
  const response = await fetch("/api/" + path, body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : {});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "请求失败");
  return payload;
}

function option(id, label) {
  return `<option value="${esc(id)}">${esc(label)}</option>`;
}

function fillSelect(name, items) {
  $(`[name=${name}]`, form).innerHTML = items.map((item) => option(item.id, item.label)).join("");
}

function renderCompare() {
  $("#compare").innerHTML = catalog.runs.map((run) => `
    <label class="check">
      <input type="checkbox" name="compare" value="${esc(run.id)}">
      <span>${esc(run.label)}<small>${esc(run.blurb)}</small></span>
    </label>`).join("");
}

function applyPreset(id) {
  const preset = catalog.presets[id];
  for (const [key, value] of Object.entries(preset)) {
    if (key === "compare") continue;
    const field = form.elements[key];
    if (field) field.value = value;
  }
  const chosen = new Set(preset.compare);
  form.querySelectorAll("[name=compare]").forEach((box) => { box.checked = chosen.has(box.value); });
  document.querySelectorAll("[data-preset]").forEach((button) => button.classList.toggle("active", button.dataset.preset === id));
  syncFields();
}

function syncFields() {
  const process = form.elements.process.value;
  const afterward = form.elements.afterward.value;
  form.querySelectorAll("[data-when]").forEach((row) => {
    const when = row.dataset.when;
    row.hidden = when === "hold" ? afterward !== "hold" : when === "ar1" ? process !== "ar1" : process === "static";
  });
  const metric = catalog.metrics.find((item) => item.id === form.elements.metric.value);
  $("#metric-hint").textContent = metric ? metric.label : "";
}

function readForm() {
  const integer = (name) => Number(form.elements[name].value);
  return {
    n_users: integer("n_users"),
    seed: integer("seed"),
    metric: form.elements.metric.value,
    process: form.elements.process.value,
    phi: Number(form.elements.phi.value),
    sigma: Number(form.elements.sigma.value),
    exposure_percent: integer("exposure_percent"),
    search_rounds: integer("search_rounds"),
    afterward: form.elements.afterward.value,
    hold_rounds: integer("hold_rounds"),
    compare: [...form.querySelectorAll("[name=compare]:checked")].map((box) => box.value),
  };
}

function readout(data) {
  const byId = Object.fromEntries(data.runs.map((run) => [run.id, run]));
  const hack = byId.extremes_hackable;
  const plain = byId.extremes_unhackable;
  const lines = [];
  if (hack && plain) {
    const hSearch = hack.series.filter((point) => point.phase === "search");
    const uSearch = plain.series.filter((point) => point.phase === "search");
    lines.push(`搜索期间，hackable 样本 AB 从 ${fmt(hSearch[0].ab_claim)} 到 ${fmt(hSearch.at(-1).ab_claim)}；unhackable 从 ${fmt(uSearch[0].ab_claim)} 到 ${fmt(uSearch.at(-1).ab_claim)}。`);
    lines.push(`搜索结束时的全体 AB：hackable ${fmt(hack.summary.ab_population_search_end)}，unhackable ${fmt(plain.summary.ab_population_search_end)}。`);
  }
  const anchor = hack || data.runs[0];
  if (anchor && data.phases.some((phase) => phase.name === "ramp")) {
    lines.push(`推全后，${anchor.label} 的 AB 是 ${fmt(anchor.summary.ab_claim_final)}。`);
  }
  if (anchor && data.phases.some((phase) => phase.name === "hold")) {
    lines.push(`分组保持后，${anchor.label} 的样本 AB 从 ${fmt(anchor.summary.ab_claim_search_end)} 到 ${fmt(anchor.summary.ab_claim_final)}。`);
  }
  if (anchor) lines.push(`群体 AA 是 ${fmt(anchor.summary.aa_latent_final)}。`);
  return lines;
}

function renderResults(data) {
  const feasible = data.runs.filter((run) => run.role === "feasible");
  const oracles = data.runs.filter((run) => run.role === "oracle");
  const phaseText = data.phases.map((phase) => `${PHASES[phase.name] || phase.name} ${phase.start}–${phase.end}`).join(" · ");
  $("#results").innerHTML = `
    <section class="readout">
      <h2>${esc(data.n_users.toLocaleString("en-US"))} 位用户 · 种子 ${esc(data.seed)} · ${esc(phaseText)}</h2>
      <ul>${readout(data).map((line) => `<li>${esc(line)}</li>`).join("")}</ul>
    </section>
    <div class="charts">
      ${feasible.length ? chartCard("可行策略的 AB", "实线是样本，虚线是全体", "feasible") : ""}
      ${oracles.length ? chartCard("参照界", "臂占比 20%，与五组 hack 对齐", "oracle") : ""}
      ${chartCard("群体 AA", "相对开局的均值变化", "aa", true)}
    </div>
    ${table(data)}`;
  if (feasible.length) drawChart($("#chart-feasible"), feasible, data.phases, ["ab_claim", "ab_population"]);
  if (oracles.length) drawChart($("#chart-oracle"), oracles, data.phases, ["ab_claim", "ab_population"]);
  drawChart($("#chart-aa"), sharedLatent(data.runs), data.phases, ["aa_latent"], { legendStyle: "solid-only" });
}

function chartCard(title, note, id, wide = false) {
  return `<section class="chart-card ${wide ? "wide" : ""}"><h2>${title}<span>${note}</span></h2><div id="chart-${id}"></div></section>`;
}

function sharedLatent(runs) {
  const first = runs[0];
  const same = runs.every((run) => run.series.every((row, index) => row.aa_latent === first.series[index].aa_latent));
  if (same) return [{ ...first, id: "latent", label: "群体 AA" }];
  return runs;
}

function table(data) {
  const headers = ["策略", "流量层", "臂占比", "搜索 · 样本 AB", "搜索 · 全体 AB", "结束 AB", "AA"];
  const rows = data.runs.map((run) => {
    const summary = run.summary;
    const cells = [
      run.label,
      run.system,
      `${Math.round(run.arm_fraction * 100)}%`,
      fmt(summary.ab_claim_search_end),
      fmt(summary.ab_population_search_end),
      fmt(summary.ab_claim_final),
      fmt(summary.aa_latent_final),
    ];
    return `<tr>${cells.map((cell) => `<td>${esc(cell)}</td>`).join("")}</tr>`;
  }).join("");
  return `<section class="table-card"><table><thead><tr>${headers.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></section>`;
}

function drawChart(host, runs, phases, fields, opts = {}) {
  const width = 640;
  const height = 250;
  const left = 48;
  const right = 12;
  const top = 12;
  const bottom = 28;
  const innerW = width - left - right;
  const innerH = height - top - bottom;
  const rounds = runs[0].series.map((row) => row.round);
  const values = runs.flatMap((run) => fields.flatMap((field) => run.series.map((row) => row[field]))).filter((value) => Number.isFinite(value));
  let min = Math.min(0, ...values);
  let max = Math.max(0, ...values);
  if (max - min < 1e-9) { min -= 0.25; max += 0.25; }
  const pad = (max - min) * 0.08;
  min -= pad;
  max += pad;
  const x = (round) => left + (rounds.length === 1 ? innerW / 2 : (round - rounds[0]) / (rounds.at(-1) - rounds[0]) * innerW);
  const y = (value) => top + (max - value) / (max - min) * innerH;
  let svg = `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img">`;
  phases.forEach((phase, index) => {
    if (index % 2 === 0) return;
    const x0 = x(phase.start) - (rounds.length > 1 ? innerW / (rounds.length - 1) / 2 : 0);
    const x1 = x(phase.end) + (rounds.length > 1 ? innerW / (rounds.length - 1) / 2 : innerW / 2);
    svg += `<rect x="${x0}" y="${top}" width="${Math.max(0, x1 - x0)}" height="${innerH}" fill="#f4f7f2"></rect>`;
  });
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = min + (max - min) * tick / 4;
    svg += `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" stroke="#e7eee6"></line>`;
    svg += `<text x="${left - 8}" y="${y(value) + 3}" text-anchor="end" fill="#8b998f" font-size="10" font-family="ui-monospace,monospace">${value.toFixed(2)}</text>`;
  }
  svg += `<line x1="${left}" y1="${y(0)}" x2="${width - right}" y2="${y(0)}" stroke="#23312d" stroke-width="0.8"></line>`;
  const xTicks = [...new Set([rounds[0], rounds[Math.floor((rounds.length - 1) / 2)], rounds.at(-1)])];
  xTicks.forEach((round) => {
    svg += `<text x="${x(round)}" y="${height - 8}" text-anchor="middle" fill="#8b998f" font-size="10" font-family="ui-monospace,monospace">${round}</text>`;
  });
  runs.forEach((run) => {
    const ordered = fields.map((field, index) => ({ field, index })).reverse();
    ordered.forEach(({ field, index }) => {
      const dash = index === 0 || opts.legendStyle === "solid-only" ? "" : ' stroke-dasharray="5 4"';
      const points = run.series.map((row) => `${x(row.round).toFixed(2)},${y(row[field]).toFixed(2)}`).join(" ");
      svg += `<polyline fill="none" stroke="${COLORS[run.id] || "#23312d"}" stroke-width="2" points="${points}"${dash}></polyline>`;
    });
  });
  svg += "</svg>";
  const legend = runs.map((run) => `<span><i style="background:${COLORS[run.id] || "#23312d"}"></i>${esc(run.label)}</span>`).join("");
  const styleNote = opts.legendStyle === "solid-only" || fields.length === 1 ? "" : `<span>实线样本 · 虚线全体</span>`;
  host.innerHTML = svg + `<div class="legend">${legend}${styleNote}</div>`;
}

async function run() {
  const id = ++requestId;
  const button = $(".primary", form);
  button.disabled = true;
  button.textContent = "运行中…";
  $("#form-error").textContent = "";
  try {
    const data = await api("run", readForm());
    if (id !== requestId) return;
    renderResults(data);
  } catch (error) {
    if (id !== requestId) return;
    $("#form-error").textContent = error.message;
  } finally {
    if (id === requestId) {
      button.disabled = false;
      button.textContent = "运行";
    }
  }
}

async function init() {
  catalog = await api("catalog");
  fillSelect("metric", catalog.metrics);
  fillSelect("process", catalog.processes);
  fillSelect("afterward", catalog.afterwards);
  renderCompare();
  applyPreset("selection");
  form.addEventListener("submit", (event) => { event.preventDefault(); run(); });
  form.addEventListener("change", syncFields);
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => { applyPreset(button.dataset.preset); run(); });
  });
  run();
}

init().catch((error) => {
  $("#results").innerHTML = `<div class="empty">无法连接本地服务：${esc(error.message)}</div>`;
});
