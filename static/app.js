(() => {
  let csrf = document.querySelector('meta[name="csrf-token"]')?.content;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const url = (u) => (/^https?:\/\//i.test(u || "") ? esc(u) : "");
  const link = (u, t) => (url(u) ? `<a href="${url(u)}" target="_blank" rel="noopener noreferrer">${esc(t)}</a>` : esc(t));
  const paperLink = (p) => link(p.doi ? "https://doi.org/" + p.doi : p.url, p.title);
  const refreshCsrf = async () => {
    const r = await fetch("/api/csrf-token", { headers: { Accept: "application/json", "X-Requested-With": "fetch" } });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data.csrf_token) throw new Error(data.error || "Your session expired. Sign in again and reopen the analysis.");
    csrf = data.csrf_token;
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) meta.content = csrf;
  };
  const api = async (u, opts = {}) => {
    const send = () => fetch(u, { ...opts, headers: { "Content-Type": "application/json", Accept: "application/json", "X-CSRFToken": csrf, "X-Requested-With": "fetch" } });
    let r = await send();
    let j = await r.json().catch(() => ({}));
    if (r.status === 400 && /csrf/i.test(j.error || "")) {
      await refreshCsrf();
      r = await send();
      j = await r.json().catch(() => ({}));
    }
    if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
    return j;
  };

  const hist = $("#hist");
  if (hist) api("/api/history").then((rows) => {
    if (!rows.length) $("#empty").hidden = false;
    hist.querySelector("tbody").innerHTML = rows.map((r) => `<tr><td>${esc(r.query)}</td><td>${esc(r.status)}</td><td>${esc(r.created.slice(0, 16).replace("T", " "))}</td>
      <td>${r.report_id ? `<a href="/?report=${r.report_id}">Open analysis</a>` : `<span class="muted">Not ready</span>`}</td></tr>`).join("");
  });

  const evalTable = $("#eval-history");
  if (evalTable) api("/api/evaluation").then((runs) => {
    const good = runs.filter((r) => r.agents?.length);
    if (!good.length) { $("#eval-empty").hidden = false; return; }
    const totals = new Map();
    good.forEach((r) => r.agents.forEach((a) => { const prev = totals.get(a.agent) || {value:0, unit:a.unit || "outputs"}; prev.value += a.measure || 0; totals.set(a.agent, prev); }));
    $("#eval-summary").innerHTML = [...totals].map(([name, v]) => `<article class="eval-card"><h3>${esc(name)}</h3><p><strong>${v.value}</strong> ${esc(v.unit)}</p><p class="muted">Summed across ${good.length} analyses.</p></article>`).join("");
    const max = Math.max(1, ...[...totals.values()].map((v) => v.value));
    $("#eval-chart").innerHTML = [...totals].map(([name, v]) => `<div class="eval-bar-row"><span>${esc(name)} <small>(${esc(v.unit)})</small></span><div class="eval-bar-track"><i style="width:${Math.max(2, v.value / max * 100)}%"></i></div><strong>${v.value}</strong></div>`).join("");
    const W=760,H=220,P=34, series=[{name:"Retrieved",color:"#963c70",agent:"Literature Search"},{name:"Structured",color:"#347f8d",agent:"Paper Understanding"}];
    const peak=Math.max(1,...good.flatMap(r=>series.map(s=>r.agents.find(a=>a.agent===s.agent)?.measure||0)));
    const pts=(s)=>good.map((r,i)=>{const v=r.agents.find(a=>a.agent===s.agent)?.measure||0;const x=P+(good.length===1?(W-2*P)/2:i*(W-2*P)/(good.length-1));const y=H-P-v/peak*(H-2*P);return `${x},${y}`;}).join(" ");
    $("#eval-trend").innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Retrieved and structured paper counts across analyses"><line x1="${P}" y1="${H-P}" x2="${W-P}" y2="${H-P}" stroke="#dce4ee"/><text x="4" y="${P}" fill="#617187" font-size="12">${peak}</text>${series.map(s=>`<polyline points="${pts(s)}" fill="none" stroke="${s.color}" stroke-width="3"/>`).join("")}${series.map(s=>`<circle cx="18" cy="${20+series.indexOf(s)*20}" r="5" fill="${s.color}"/><text x="30" y="${24+series.indexOf(s)*20}" fill="#455970" font-size="12">${s.name}</text>`).join("")}</svg>`;
    evalTable.querySelector("tbody").innerHTML = good.map((r) => `<tr><td><a href="/?report=${r.search_id}">${esc(r.query)}</a></td><td>${esc(r.status)}</td><td>${esc(r.created.slice(0,16).replace("T"," "))}</td><td>${r.agents.map((a) => `${esc(a.agent)}: ${esc(a.output)}`).join("<br>")}</td></tr>`).join("");
  }).catch((e) => { $("#eval-error").textContent = e.message; $("#eval-error").hidden = false; });

  const form = $("#run");
  if (!form) return;
  const stages = [...document.querySelectorAll("#track li")];
  const setStage = (s) => { const i = stages.findIndex((x) => x.dataset.s === s); stages.forEach((el, k) => (el.className = k < i ? "done" : k === i ? "now" : "")); };
  const showErr = (m) => { const e = $("#err"); e.textContent = m; e.hidden = !m; };
  const VERDICT = { NOVEL: ["Potentially unexplored", "ok"], PARTIALLY_EXPLORED: ["Partially explored", "mid"], ALREADY_ADDRESSED: ["Already addressed", "bad"], INSUFFICIENT_EVIDENCE: ["Insufficient evidence", "mid"] };
  const badge = (v) => (VERDICT[v] ? `<span class="pill ${VERDICT[v][1]}">${VERDICT[v][0]}</span>` : "");
  const chip = (t, cls = "") => `<span class="chip ${cls}">${esc(t)}</span>`;

  document.querySelectorAll("[data-topic]").forEach((button) => button.addEventListener("click", () => {
    const input = $("#q");
    input.value = button.dataset.topic || "";
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }));

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault(); showErr("");
    $("#go").disabled = true; $("#result").hidden = true; $("#progress").hidden = false; $("#hero").classList.remove("compact"); setStage("search"); $("#msg").textContent = "Starting…";
    try {
      const { search_id, report_id } = await api("/pipeline", { method: "POST", body: JSON.stringify({ query: $("#q").value }) });
      while (true) {
        await new Promise((r) => setTimeout(r, 2000));
        const s = await api(`/status/${search_id}`);
        setStage(s.stage); $("#msg").textContent = s.message;
        if (s.status === "failed") throw new Error(s.error || "The analysis failed.");
        if (s.status === "done") break;
      }
      await load(report_id);
    } catch (e) { showErr(e.message); $("#progress").hidden = true; }
    finally { $("#go").disabled = false; }
  });

  // ---------- renderers
  const evidence = (list) => `<ul class="ev">${(list || []).map((e) => `<li><span class="role">${esc(e.role || "")}</span>${paperLink(e)} <span class="muted">(${esc(e.year ?? "n.d.")})</span>
      <p class="muted">${esc(e.snippet)}</p></li>`).join("")}</ul>`;

  function gapPane(r) {
    const g = r.verified_gap, n = r.novelty;
    if (!g) return `<div class="empty"><h3>No gap passed the prior-art check</h3><p>Every candidate was either already addressed or could not be verified. Open <em>All candidates</em> to see what was found, or try a narrower topic.</p></div>`;
    const search = (n.search_log || []).map((s) => `<li><code>${esc(s.query)}</code> <span class="muted">→ ${esc(s.results ?? s.error)} results</span></li>`).join("");
    const closest = (n.closest || []).map((c) => `<li>${paperLink(c)} <span class="muted">(${esc(c.year ?? "n.d.")}, ${esc(c.type)})</span><div>${(c.matched || []).map((m) => chip(m, "hit")).join("")}
      ${(n.core_concepts || []).filter((m) => !(c.matched || []).includes(m)).map((m) => chip(m + " ✕", "miss")).join("")}</div>${c.reason ? `<p class="muted">${esc(c.reason)}</p>` : ""}</li>`).join("");
    return `<article class="gapcard">
      <p class="eyebrow">Recommended research gap ${badge(n.verdict)} <span class="pill">confidence ${esc(n.confidence)}</span></p>
      <h3>${esc(g.title)}</h3><p class="statement">${esc(g.statement)}</p>
      <div class="three"><div><h4>Already known</h4><p>${esc(g.known)}</p></div><div><h4>Not yet studied</h4><p>${esc(g.missing)}</p></div><div><h4>Why it matters</h4><p>${esc(g.why)}</p></div></div>
    </article>
    <h4 class="sec">Evidence this gap rests on</h4>${evidence(g.evidence)}
    <h4 class="sec">Prior-art check</h4>
    <p class="verdict-text">${esc(n.explanation)}</p>
    <p class="muted">A paper counts as “addressing” the gap only if it covers every concept: ${(n.core_concepts || []).map((c) => chip(c)).join(" ")}. ${esc(n.confidence_reason)}</p>
    <div class="two"><div><h5>Searches run</h5><ul class="plain">${search || "<li class='muted'>none</li>"}</ul></div><div><h5>Closest existing work</h5><ul class="plain">${closest || "<li class='muted'>none</li>"}</ul></div></div>`;
  }

  function planPane(r) {
    const p = r.plan;
    if (!p) return `<div class="empty"><h3>No experiment plan</h3><p>A plan is generated for a promising novel or partially explored direction. Insufficient evidence is clearly labelled and is not treated as verified novelty.</p></div>`;
    const tbl = (rows, cols) => `<table class="t"><thead><tr>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr></thead><tbody>${rows.map((x) => `<tr>${cols.map((c) => `<td>${c[1](x)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    return `${p.provisional ? `<p class="banner">Provisional plan: prior-art evidence was insufficient. This is an exploration proposal, not a verified novelty claim.</p>` : ""}<h3 class="rq">${esc(p.research_question)}</h3><p class="statement">${esc(p.hypothesis)}</p>
      <h4 class="sec">Procedure</h4><ol class="steps">${p.methodology.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
      <h4 class="sec">Datasets</h4>${p.datasets.length ? tbl(p.datasets, [["Dataset", (d) => link(d.url, d.name)], ["Access", (d) => esc(d.access)], ["Why this one", (d) => esc(d.reason)]]) : "<p class='muted'>No suitable public dataset was identified automatically. Search benchmark portals for this task before starting.</p>"}
      <h4 class="sec">Baselines</h4>${tbl(p.baselines, [["Baseline", (b) => `<strong>${esc(b.name)}</strong>`], ["Why", (b) => esc(b.reason)], ["Source", (b) => chip(b.source)]])}
      <h4 class="sec">Evaluation metrics</h4>${tbl(p.metrics, [["Metric", (m) => `<strong>${esc(m.name)}</strong>`], ["Why", (m) => esc(m.reason)]])}
      <div class="two"><div><h5>Ablations</h5><ul>${p.ablations.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div><div><h5>Risks</h5><ul>${p.risks.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div></div>
      <p class="muted">Plan generated by: ${esc(p.generated_by)}. Dataset, baseline and metric names come from the retrieved literature and a curated registry, not free generation.</p>`;
  }

  const candPane = (r) => (r.candidates || []).map((c) => `<details class="cand ${r.verified_gap && r.verified_gap.id === c.id ? "pick" : ""}"><summary><strong>${esc(c.title)}</strong> ${c.novelty ? badge(c.novelty.verdict) : ""}
      <span class="muted">${esc(c.type.replaceAll("_", " ").toLowerCase())}</span></summary><p>${esc(c.statement)}</p>${c.missing ? `<p><strong>Missing study:</strong> ${esc(c.missing)}</p>` : ""}
      ${c.novelty ? `<p class="muted">${esc(c.novelty.explanation)}</p>` : ""}${evidence(c.evidence)}</details>`).join("") || "<p class='muted'>No candidate gaps were produced.</p>";

  function papersPane(r) {
    const row = (p) => { const s = p.structured || {}; return `<li data-type="${esc(s.paper_type || "study")}"><details><summary>${paperLink(p)} <span class="muted">${esc(p.year ?? "n.d.")}</span> ${chip(s.paper_type || "study", s.paper_type === "review" ? "miss" : "hit")}</summary>
      <p class="muted">${esc(p.abstract)}</p><div>${(s.methods || []).map((m) => chip(m)).join("")}${(s.datasets || []).map((m) => chip(m, "ds")).join("")}${(s.aspects || []).map((m) => chip(m, "asp")).join("")}</div></details></li>`; };
    return `<div class="filters"><button data-f="all" class="on">All (${r.papers.length})</button><button data-f="study">Studies</button><button data-f="review">Reviews</button></div><ul class="papers">${r.papers.map(row).join("")}</ul>`;
  }

  function graphPane(r) {
    const g = r.graph;
    if (!g) return '<p class="muted">No graph was built for this analysis.</p>';
    return `<p class="muted">${esc(JSON.stringify(g.stats))}</p><h3>Research topics</h3><ul>${g.topics.map(t => `<li><strong>${esc(t.label)}</strong> · ${t.pids.length} papers</li>`).join("")}</ul><h3>Relationships</h3><div class="table-wrap"><table class="t"><thead><tr><th>From</th><th>Relationship</th><th>To</th></tr></thead><tbody>${g.edges.slice(0,500).map(e => `<tr><td>${esc(g.nodes.find(n=>n.id===e.source)?.title || g.nodes.find(n=>n.id===e.source)?.name || e.source)}</td><td>${esc(e.relation)}</td><td>${esc(g.nodes.find(n=>n.id===e.target)?.title || g.nodes.find(n=>n.id===e.target)?.name || e.target)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function evaluationPane(r) {
    const rows = r.agent_evaluation || [];
    if (!rows.length) return '<p class="muted">No evaluation record was saved for this analysis.</p>';
    const max = Math.max(1, ...rows.map(a => a.measure || 0));
    return `<p class="callout">These are measured execution outputs. Precision, recall and F1 require a labelled benchmark and are not inferred from output volume.</p><div class="eval-chart">${rows.map(a => `<div class="eval-bar-row"><span>${esc(a.agent)} <small>(${esc(a.unit || "outputs")})</small></span><div class="eval-bar-track"><i style="width:${Math.max(2,(a.measure||0)/max*100)}%"></i></div><strong>${a.measure || 0}</strong></div>`).join("")}</div><div class="table-wrap"><table class="t"><thead><tr><th>Agent</th><th>Status</th><th>Recorded output</th><th>Measurement note</th></tr></thead><tbody>${rows.map(a => `<tr><td>${esc(a.agent)}</td><td>${esc(a.status)}</td><td>${esc(a.output)}</td><td>${esc(a.note)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  let activeSearchId = null;
  function drawChat(rows) {
    const box = $("#chat-messages");
    if (!box) return;
    box.innerHTML = (rows || []).map(m => `<div class="chat-message ${m.role === "user" ? "user" : "assistant"}"><strong>${m.role === "user" ? "You" : "ResearchGapAI"}</strong><p>${esc(m.content).replace(/\n/g,"<br>")}</p></div>`).join("") || '<p class="muted">No questions yet. Ask your first question about this analysis.</p>';
    box.scrollTop = box.scrollHeight;
  }
  async function loadChat(searchId) {
    activeSearchId = searchId;
    try { drawChat(await api(`/api/analysis/${searchId}/chat`)); } catch (e) { $("#chat-error").textContent = e.message; $("#chat-error").hidden = false; }
  }
  const chatForm = $("#chat-form");
  if (chatForm) chatForm.addEventListener("submit", async (ev) => {
    ev.preventDefault(); if (!activeSearchId) return;
    const input = $("#chat-input"), error = $("#chat-error"), send = chatForm.querySelector("button");
    error.hidden = true; send.disabled = true;
    try {
      const result = await api(`/api/analysis/${activeSearchId}/chat`, {method:"POST", body:JSON.stringify({message:input.value})});
      const rows = await api(`/api/analysis/${activeSearchId}/chat`); drawChat(rows); input.value = "";
    } catch (e) { error.textContent = e.message; error.hidden = false; }
    finally { send.disabled = false; }
  });

  async function load(id) {
    const r = await api(`/report/${id}`);
    $("#progress").hidden = true; $("#result").hidden = false; $("#hero").classList.add("compact");
    $("#rtitle").textContent = r.query;
    const m = r.meta || {};
    $("#chips").innerHTML = [`${r.papers.length} papers`, m.corpus ? `${m.corpus.studies} studies / ${m.corpus.reviews} reviews` : "", `Reasoning: ${m.llm === "heuristic" ? "rule-based" : (m.llm || "n/a")}`, m.graph ? `Knowledge graph: ${m.graph.backend}` : ""].filter(Boolean).map((t) => chip(t)).join("");
    const warns = [...((m.corpus || {}).warnings || [])];
    $("#warn").innerHTML = warns.map((w) => `<p class="banner">${esc(w)}</p>`).join("");
    $("#pane-gap").innerHTML = gapPane(r); $("#pane-plan").innerHTML = planPane(r); $("#pane-cands").innerHTML = candPane(r);
    $("#pane-papers").innerHTML = papersPane(r);
    $("#pane-graph").innerHTML = graphPane(r);
    $("#pane-evaluation").innerHTML = evaluationPane(r);
    $("#pane-report").innerHTML = r.report_html; $("#dl").href = `/report/${id}?format=md`;
    if (r.search_id) loadChat(r.search_id);
    document.querySelectorAll(".filters button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll(".filters button").forEach((x) => x.classList.toggle("on", x === b));
      document.querySelectorAll(".papers li").forEach((li) => (li.hidden = b.dataset.f !== "all" && li.dataset.type !== b.dataset.f));
    }));
  }
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".pane").forEach((p) => (p.hidden = p.id !== "pane-" + b.dataset.t));
    if (b.dataset.t === "chat" && activeSearchId) loadChat(activeSearchId);
  }));
  const rid = new URLSearchParams(location.search).get("report");
  if (rid) load(rid).catch((e) => showErr(e.message));
})();
