/* ============================================================
   AI DJ — frontend controller
   ============================================================ */
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const state = {
  folder: null,
  running: false,
  startedAt: 0,
  serverElapsed: 0,
  eta: null,
  lastSync: 0,
  trackCount: 0,
  planRendered: false,
};

/* ---------- helpers ---------- */
function fmtTime(sec) {
  if (sec == null || isNaN(sec) || sec < 0) return "—";
  sec = Math.floor(sec);
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}
function toast(msg, kind = "err") {
  const t = $("#toast");
  $("#toastMsg").textContent = msg;
  t.hidden = false;
  t.classList.toggle("ok", kind === "ok");
  requestAnimationFrame(() => t.classList.add("show"));
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => (t.hidden = true), 400);
  }, 5000);
}
function setStatus(text, cls) {
  $("#statusText").textContent = text;
  const dot = $("#statusDot");
  dot.className = "dot" + (cls ? " " + cls : "");
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch (e) {}
    throw new Error(m);
  }
  return r.json();
}

/* ---------- source selection ---------- */
$("#browseBtn").addEventListener("click", async () => {
  setStatus("Opening picker…", "busy");
  try {
    const res = await api("/api/pick-folder", { method: "POST" });
    if (res.ok && res.folder) {
      setFolder(res.folder);
      quickScan(res.folder);
    } else {
      toast(res.error || "No folder chosen.");
      setStatus("Ready");
    }
  } catch (e) {
    toast("Native picker unavailable — paste a path instead.");
    setStatus("Ready");
  }
});

$("#scanBtn").addEventListener("click", () => {
  const p = $("#folderPath").value.trim();
  if (!p) return toast("Enter a folder path first.");
  setFolder(p);
  quickScan(p);
});
$("#folderPath").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#scanBtn").click();
});

/* webkitdirectory upload fallback */
$("#folderUpload").addEventListener("change", async (e) => {
  const files = [...e.target.files].filter((f) =>
    /\.(mp3|wav|flac|m4a|aac|ogg|opus|aiff?|wma)$/i.test(f.name)
  );
  if (!files.length) return toast("No audio files found in that folder.");
  setStatus(`Uploading ${files.length} files…`, "busy");
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    setFolder(res.folder);
    $("#scanCount").textContent = `${res.saved} tracks`;
    $("#scanNote").textContent = "uploaded to local workspace";
    $("#scanSummary").hidden = false;
    setStatus("Ready");
  } catch (err) {
    toast("Upload failed: " + err.message);
    setStatus("Ready", "err");
  }
});

function setFolder(path) {
  state.folder = path;
  $("#folderPath").value = path;
  $("#generateBtn").disabled = false;
  $("#dropZone").classList.remove("drag");
}

async function quickScan(folder) {
  setStatus("Scanning…", "busy");
  try {
    const res = await api("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder }),
    });
    $("#scanCount").textContent = `${res.count} tracks`;
    $("#scanNote").textContent = res.warnings.length
      ? res.warnings[0]
      : "ready to mix";
    $("#scanSummary").hidden = false;
    $("#generateBtn").disabled = res.count < 2;
    setStatus(res.count >= 2 ? "Ready" : "Need 2+ tracks", res.count >= 2 ? "" : "err");
  } catch (e) {
    toast("Scan failed: " + e.message);
    setStatus("Ready", "err");
  }
}

/* drag & drop just captures a hint (browsers can't give real paths) */
const dz = $("#dropZone");
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); })
);
dz.addEventListener("drop", () => toast("Drag-drop can't read folder paths — use Browse or Upload.", "ok"));

/* button glow follows cursor */
$$(".btn").forEach((b) =>
  b.addEventListener("mousemove", (e) => {
    const r = b.getBoundingClientRect();
    b.style.setProperty("--mx", e.clientX - r.left + "px");
    b.style.setProperty("--my", e.clientY - r.top + "px");
  })
);

/* ---------- settings ---------- */
$("#settingsToggle").addEventListener("click", () => {
  $("#settingsBody").classList.toggle("collapsed");
  $("#settingsChev").classList.toggle("up");
});
function bindRange(id, out, fmt) {
  const el = $(id), o = $(out);
  const upd = () => {
    el.style.setProperty("--fill", el.value + "%");
    o.textContent = fmt(el.value);
  };
  el.addEventListener("input", upd);
  upd();
}
bindRange("#transIntensity", "#transOut", (v) => v + "%");
bindRange("#fxIntensity", "#fxOut", (v) => v + "%");
bindRange("#harmPriority", "#harmOut", (v) => v + "%");
bindRange("#mixLength", "#lenOut", (v) => (v === "0" ? "Auto" : v + " min"));
$("#volume").style.setProperty("--fill", "90%");

$$(".segmented").forEach((seg) =>
  seg.querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      seg.dataset.value = btn.dataset.v;
    })
  )
);

function collectSettings() {
  return {
    energy_curve: $("#energyCurve").dataset.value,
    transition_intensity: +$("#transIntensity").value / 100,
    effect_intensity: +$("#fxIntensity").value / 100,
    harmonic_priority: +$("#harmPriority").value / 100,
    target_minutes: +$("#mixLength").value,
    preserve_quality: $("#preserveQuality").checked,
    aggressive: $("#aggressive").checked,
    output_format: $("#outFormat").dataset.value,
  };
}

/* ---------- generate ---------- */
$("#generateBtn").addEventListener("click", async () => {
  if (!state.folder) return toast("Select a music folder first.");
  if (state.running) return;
  $("#generateBtn").disabled = true;
  resetUI();
  try {
    await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder: state.folder, settings: collectSettings() }),
    });
    state.running = true;
    state.startedAt = performance.now();
    $("#progressCard").hidden = false;
    $("#progressCard").scrollIntoView({ behavior: "smooth", block: "start" });
    setStatus("Mixing…", "busy");
    startTimers();
    listenProgress();
  } catch (e) {
    toast("Could not start: " + e.message);
    $("#generateBtn").disabled = false;
    setStatus("Ready", "err");
  }
});

function resetUI() {
  ["#tracksCard", "#planCard", "#resultCard"].forEach((s) => ($(s).hidden = true));
  $("#trackList").innerHTML = "";
  $("#journey").innerHTML = "";
  $("#warnings").innerHTML = "";
  state.trackCount = 0;
  state.planRendered = false;
  finishSuccess._done = false; // allow generating another mix without reloading
  if (poll._t) { clearInterval(poll._t); poll._t = null; }
  const a = $("#audio");
  if (a) { a.pause(); a.removeAttribute("src"); }
  if (typeof vizRAF !== "undefined") cancelAnimationFrame(vizRAF);
}

/* ---------- progress stream ---------- */
function listenProgress() {
  let ws;
  try {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/progress`);
    ws.onmessage = (ev) => handleSnapshot(JSON.parse(ev.data));
    ws.onerror = () => { try { ws.close(); } catch (e) {} poll(); };
  } catch (e) {
    poll();
  }
}
function poll() {
  if (poll._t) return;
  poll._t = setInterval(async () => {
    try {
      const snap = await api("/api/progress");
      handleSnapshot(snap);
      if (snap.done) { clearInterval(poll._t); poll._t = null; }
    } catch (e) {}
  }, 500);
}

const STAGE_ORDER = ["scanning", "metadata", "analysis", "planning", "transitions", "rendering", "finalizing"];

function handleSnapshot(snap) {
  // progress bar
  const pct = Math.round((snap.overall || 0) * 100);
  $("#progressFill").style.width = pct + "%";
  $("#progressPct").textContent = pct + "%";
  $("#progressMessage").textContent = snap.message || "";
  $("#progressStageTitle").textContent = titleForStage(snap.stage);

  // sync timers
  if (snap.elapsed != null) { state.serverElapsed = snap.elapsed; state.lastSync = performance.now(); }
  state.eta = snap.eta_seconds;

  // stage stepper
  const curIdx = STAGE_ORDER.indexOf(snap.stage);
  $$(".stage-chip").forEach((chip) => {
    const i = STAGE_ORDER.indexOf(chip.dataset.s);
    chip.classList.toggle("active", chip.dataset.s === snap.stage);
    chip.classList.toggle("past", curIdx > -1 && i < curIdx);
  });

  // warnings
  if (snap.warnings && snap.warnings.length) {
    $("#warnings").innerHTML = snap.warnings
      .map((w) => `<div class="warn-line">⚠ ${escapeHtml(w)}</div>`)
      .join("");
  }

  // tracks
  if (snap.tracks && snap.tracks.length > state.trackCount) {
    renderTracks(snap.tracks);
    state.trackCount = snap.tracks.length;
  }

  // plan
  if (snap.plan && snap.plan.transition_details && !state.planRendered) {
    renderPlan(snap.plan);
    state.planRendered = true;
  }

  if (snap.stage === "error" || snap.error) {
    finishError(snap.error || snap.message);
  } else if (snap.done && snap.result) {
    finishSuccess(snap.result);
  }
}

function titleForStage(s) {
  return {
    scanning: "Scanning folder", metadata: "Reading metadata",
    analysis: "Analysing tracks", planning: "Planning the set",
    transitions: "Designing transitions", rendering: "Rendering the mix",
    finalizing: "Finalising", complete: "Complete", error: "Error",
  }[s] || "Working…";
}

/* ---------- timers ---------- */
function startTimers() {
  cancelAnimationFrame(startTimers._raf);
  const tick = () => {
    if (!state.running) return;
    const localElapsed = state.serverElapsed + (performance.now() - state.lastSync) / 1000;
    $("#elapsedTimer").textContent = fmtTime(localElapsed);
    let eta = state.eta;
    if (eta != null) eta = Math.max(0, eta - (performance.now() - state.lastSync) / 1000);
    $("#etaTimer").textContent = eta == null ? "—" : fmtTime(eta);
    startTimers._raf = requestAnimationFrame(tick);
  };
  tick();
}

/* ---------- track list ---------- */
function renderTracks(tracks) {
  $("#tracksCard").hidden = false;
  $("#trackCountPill").textContent = tracks.length;
  const list = $("#trackList");
  for (let i = state.trackCount; i < tracks.length; i++) {
    const t = tracks[i];
    const row = document.createElement("div");
    row.className = "track-row";
    row.style.animationDelay = (i - state.trackCount) * 0.04 + "s";
    row.innerHTML = `
      <span class="tr-idx">${i + 1}</span>
      <div class="tr-main">
        <div class="tr-title">${escapeHtml(t.title || "Untitled")}</div>
        <div class="tr-artist">${escapeHtml(t.artist || "Unknown")}${t.genre ? " · " + escapeHtml(t.genre) : ""}${(t.mood_tags && t.mood_tags.length) ? " · " + escapeHtml(t.mood_tags.slice(0, 2).join(", ")) : ""}</div>
      </div>
      <div class="tr-meta">
        <span class="chip bpm">${t.bpm ? t.bpm.toFixed(0) : "?"} BPM</span>
        <span class="chip key">${t.camelot || "?"}</span>
        <span class="chip">E ${Math.round((t.energy || 0) * 100)}</span>
        <div class="spark">${sparkline(t.energy_curve)}</div>
      </div>`;
    list.appendChild(row);
  }
}
function sparkline(curve) {
  if (!curve || !curve.length) return "";
  const step = Math.max(1, Math.floor(curve.length / 16));
  let bars = "";
  for (let i = 0; i < curve.length; i += step) {
    const h = Math.max(6, Math.min(100, curve[i] * 100));
    bars += `<i style="height:${h}%"></i>`;
  }
  return bars;
}

/* ---------- plan / journey ---------- */
function renderPlan(plan) {
  $("#planCard").hidden = false;
  $("#planSummary").textContent =
    `${plan.n_tracks} tracks · ${plan.energy_style} energy · ~${Math.round(plan.est_duration / 60)} min`;
  const jr = $("#journey");
  jr.innerHTML = "";
  const tracks = plan.tracks || [];
  const trans = plan.transition_details || [];
  tracks.forEach((t, i) => {
    const li = document.createElement("li");
    const tr = trans[i];
    li.innerHTML = `
      <div class="j-title">${i + 1}. ${escapeHtml(t.title)} <span class="chip bpm">${t.bpm} BPM</span> <span class="chip key">${t.camelot}</span></div>
      ${tr ? `<div class="j-reason">${escapeHtml(tr.reason)}</div>
              <span class="j-tech">${tr.technique} · ${tr.overlap}s ${tr.beatmatched ? "· beatmatched" : ""}</span>` : ""}`;
    jr.appendChild(li);
  });
}

/* ---------- finish ---------- */
function finishError(msg) {
  state.running = false;
  setStatus("Error", "err");
  $("#generateBtn").disabled = false;
  toast(msg || "The mix failed. Check the logs.");
}
function finishSuccess(result) {
  if (finishSuccess._done) return;
  finishSuccess._done = true;
  state.running = false;
  setStatus("Done", "");
  $("#statusDot").className = "dot";
  $("#generateBtn").disabled = false;
  $("#progressFill").style.width = "100%";
  $("#progressPct").textContent = "100%";
  $$(".stage-chip").forEach((c) => c.classList.add("past"));
  showResult(result);
  setTimeout(() => setStatus("Ready"), 1500);
}

function showResult(result) {
  const card = $("#resultCard");
  card.hidden = false;
  const wavUrl = "/outputs/" + result.wav.split("/").pop();
  const mp3Url = result.mp3 ? "/outputs/" + result.mp3.split("/").pop() : null;
  $("#resultMeta").textContent =
    `${fmtTime(result.duration)} · ${result.n_tracks} tracks · ${result.lufs} LUFS · peak ${result.peak_dbfs} dBFS · rendered in ${result.elapsed || "?"}s`;

  const audio = $("#audio");
  audio.src = wavUrl;
  $("#dlWav").href = wavUrl;
  const mp3 = $("#dlMp3");
  if (mp3Url) { mp3.href = mp3Url; mp3.style.display = ""; }
  else mp3.style.display = "none";

  // integrity check badges
  const cb = $("#checkBadges");
  cb.innerHTML = "";
  const checks = result.checks || {};
  [["no clip", checks.no_clip], ["no NaN", checks.no_nan],
   ["valid SR", checks.sr_ok], ["reloadable", checks.reloadable],
   ["length ok", checks.duration_ok]].forEach(([label, ok]) => {
    const b = document.createElement("span");
    b.className = "cbadge" + (ok ? "" : " fail");
    b.textContent = (ok ? "✓ " : "✕ ") + label;
    cb.appendChild(b);
  });

  card.scrollIntoView({ behavior: "smooth", block: "center" });
  setupPlayer();
}

/* ---------- custom audio player + visualizer ---------- */
let audioCtx, analyser, sourceNode, vizRAF;
function setupPlayer() {
  const audio = $("#audio");
  const playBtn = $("#playBtn");
  const playIcon = $("#playIcon");
  const seekTrack = $("#seekTrack");
  const seekFill = $("#seekFill");

  playBtn.onclick = async () => {
    if (!audioCtx) initViz(audio);
    if (audioCtx.state === "suspended") await audioCtx.resume();
    if (audio.paused) audio.play(); else audio.pause();
  };
  audio.onplay = () => { playIcon.innerHTML = '<path d="M6 5h4v14H6zM14 5h4v14h-4z"/>'; drawViz(); };
  audio.onpause = () => { playIcon.innerHTML = '<path d="M8 5v14l11-7z"/>'; };
  audio.onloadedmetadata = () => { $("#durTime").textContent = fmtTime(audio.duration); };
  audio.ontimeupdate = () => {
    const p = audio.currentTime / (audio.duration || 1);
    seekFill.style.width = p * 100 + "%";
    $("#curTime").textContent = fmtTime(audio.currentTime);
  };
  seekTrack.onclick = (e) => {
    const r = seekTrack.getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
  };
  const vol = $("#volume");
  vol.oninput = () => { audio.volume = vol.value / 100; vol.style.setProperty("--fill", vol.value + "%"); };
  audio.volume = vol.value / 100;
}

function initViz(audio) {
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    sourceNode = audioCtx.createMediaElementSource(audio);
    sourceNode.connect(analyser);
    analyser.connect(audioCtx.destination);
  } catch (e) { console.warn("viz init failed", e); }
}
function drawViz() {
  const canvas = $("#viz");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const resize = () => { canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr; };
  resize();
  const bins = analyser ? analyser.frequencyBinCount : 64;
  const data = new Uint8Array(bins);
  const render = () => {
    if ($("#audio").paused) { cancelAnimationFrame(vizRAF); return; }
    vizRAF = requestAnimationFrame(render);
    if (analyser) analyser.getByteFrequencyData(data);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const n = 64, w = canvas.width / n;
    for (let i = 0; i < n; i++) {
      const v = analyser ? data[Math.floor(i * bins / n)] / 255 : Math.random() * 0.5;
      const h = Math.max(2 * dpr, v * canvas.height * 0.92);
      const x = i * w, y = (canvas.height - h) / 2;
      const g = ctx.createLinearGradient(0, y, 0, y + h);
      g.addColorStop(0, "#21d4fd"); g.addColorStop(1, "#7c5cff");
      ctx.fillStyle = g;
      const r = Math.min(w * 0.35, h / 2);
      roundRect(ctx, x + w * 0.18, y, w * 0.64, h, r);
      ctx.fill();
    }
  };
  render();
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* ---------- util ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* boot */
setStatus("Ready");
api("/api/health").catch(() =>
  toast("Backend not reachable — start the server (see README).")
);
