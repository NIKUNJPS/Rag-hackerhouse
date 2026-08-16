// ---------- element refs ----------
const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");
const textForm = document.getElementById("textForm");
const textInput = document.getElementById("textInput");
const waveformCanvas = document.getElementById("waveform");
const waveformCtx = waveformCanvas.getContext("2d");
const apiDot = document.getElementById("apiDot");
const apiStatus = document.getElementById("apiStatus");
const stagesPanel = document.getElementById("stagesPanel");

const LATENCY_TARGET_MS = 200; // chunking + vector retrieval, per task spec

// ---------- backend health check ----------
async function checkHealth() {
  try {
    const res = await fetch("/health", { cache: "no-store" });
    if (res.ok) {
      apiDot.className = "api-dot up";
      apiStatus.textContent = "online";
      return;
    }
  } catch (_) {}
  apiDot.className = "api-dot down";
  apiStatus.textContent = "offline";
}
checkHealth();
setInterval(checkHealth, 15000);

// ---------- mic recording ----------
const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];
function pickMimeType() {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
  for (const m of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}
function extForMime(mime) {
  if (mime.includes("webm")) return "webm";
  if (mime.includes("mp4")) return "mp4";
  if (mime.includes("ogg")) return "ogg";
  return "wav";
}

let mediaRecorder;
let recChunks = [];
let audioCtx, analyser, waveRaf, mediaStream;

async function setupRecorder() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = pickMimeType();
  mediaRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);

  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    const mime = mediaRecorder.mimeType || "audio/webm";
    const blob = new Blob(recChunks, { type: mime });
    recChunks = [];
    await sendAudio(blob, mime);
  };

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(mediaStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
}

function drawWaveform() {
  const bufferLength = analyser.frequencyBinCount;
  const data = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(data);

  const w = waveformCanvas.width, h = waveformCanvas.height;
  waveformCtx.clearRect(0, 0, w, h);
  const barCount = 28;
  const step = Math.floor(bufferLength / barCount);
  const barWidth = w / barCount;

  for (let i = 0; i < barCount; i++) {
    const v = data[i * step] / 255;
    const barH = Math.max(2, v * h);
    waveformCtx.fillStyle = i % 3 === 0 ? "#8b5cf6" : "#39ff88";
    waveformCtx.fillRect(i * barWidth + 1, h - barH, barWidth - 2, barH);
  }
  waveRaf = requestAnimationFrame(drawWaveform);
}

recordBtn.addEventListener("mousedown", (e) => { e.preventDefault(); startRecording(); });
recordBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); }, { passive: false });
recordBtn.addEventListener("mouseup", stopRecording);
recordBtn.addEventListener("mouseleave", stopRecording);
recordBtn.addEventListener("touchend", stopRecording);
recordBtn.addEventListener("keydown", (e) => {
  if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); startRecording(); }
});
recordBtn.addEventListener("keyup", (e) => {
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); stopRecording(); }
});

async function startRecording() {
  if (recordBtn.classList.contains("recording")) return;
  try {
    if (!mediaRecorder) await setupRecorder();
  } catch (err) {
    statusEl.textContent = "Mic permission denied or unavailable — use the text box instead.";
    console.error(err);
    return;
  }
  recChunks = [];
  mediaRecorder.start();
  recordBtn.classList.add("recording");
  recordBtn.setAttribute("aria-pressed", "true");
  recordBtn.querySelector(".mic-label").textContent = "Release to send";
  statusEl.textContent = "Listening…";
  waveformCanvas.classList.add("active");
  drawWaveform();
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") return;
  mediaRecorder.stop();
  recordBtn.classList.remove("recording");
  recordBtn.setAttribute("aria-pressed", "false");
  recordBtn.querySelector(".mic-label").textContent = "Hold to talk";
  statusEl.textContent = "Transcribing + retrieving + answering…";
  waveformCanvas.classList.remove("active");
  cancelAnimationFrame(waveRaf);
  waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
}

async function sendAudio(blob, mime) {
  const formData = new FormData();
  formData.append("file", blob, `query.${extForMime(mime)}`);
  try {
    const res = await fetch("/ask/voice", { method: "POST", body: formData });
    const data = await res.json();
    statusEl.textContent = "";
    renderResult(data);
  } catch (err) {
    statusEl.textContent = "Request failed — check the backend is running.";
    console.error(err);
  }
}

// ---------- text query fallback ----------
textForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = textInput.value.trim();
  if (!q) return;
  await sendText(q);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const q = chip.dataset.query;
    textInput.value = q;
    sendText(q);
  });
});

async function sendText(query) {
  statusEl.textContent = "Retrieving + answering…";
  const formData = new FormData();
  formData.append("query", query);
  try {
    const res = await fetch("/ask/text", { method: "POST", body: formData });
    const data = await res.json();
    statusEl.textContent = "";
    renderResult(data);
  } catch (err) {
    statusEl.textContent = "Request failed — check the backend is running.";
    console.error(err);
  }
}

// ---------- rendering ----------
function renderResult(data) {
  const transcriptPanel = document.getElementById("transcriptPanel");
  const answerPanel = document.getElementById("answerPanel");
  const sourcesPanel = document.getElementById("sourcesPanel");
  const latencyPanel = document.getElementById("latencyPanel");
  const badge = document.getElementById("badge");

  document.getElementById("transcriptText").textContent = data.query || "(no transcript)";
  transcriptPanel.hidden = false;

  document.getElementById("answerText").textContent = data.answer || data.error || "No answer";
  answerPanel.hidden = false;

  badge.className = "badge";
  const BADGES = {
    ok: ["Grounded answer", "badge-ok"],
    offtopic: ["Off-topic for this dataset", "badge-warn"],
    ungrounded: ["Not confidently grounded", "badge-warn"],
    unsafe: ["Blocked by safety guardrail", "badge-err"],
    error: ["Pipeline error", "badge-err"],
  };
  const [label, cls] = BADGES[data.status] || ["Unknown status", "badge-err"];
  badge.textContent = label;
  badge.classList.add(cls);

  const sourcesList = document.getElementById("sourcesList");
  sourcesList.innerHTML = "";
  const chunksUsed = data.chunks_used || [];
  chunksUsed.forEach((c) => {
    const li = document.createElement("li");
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.innerHTML =
      `<span>score ${(c.score ?? 0).toFixed(3)}</span>` +
      `<span>${c.strategy || "unknown"}</span>` +
      `<span>doc ${c.doc_id ?? "?"}</span>`;
    const text = document.createElement("div");
    text.className = "source-text";
    text.textContent = c.text.length > 220 ? c.text.slice(0, 220) + "…" : c.text;
    li.appendChild(meta);
    li.appendChild(text);
    sourcesList.appendChild(li);
  });
  sourcesPanel.hidden = chunksUsed.length === 0;

  renderLatency(data);
  renderStages(data);

  stagesPanel.hidden = false;
  latencyPanel.hidden = false;

  answerPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderLatency(data) {
  const timings = data.timings || [];
  const byStage = {};
  timings.forEach((t) => { byStage[t.stage] = t.ms; });

  const bar = document.getElementById("latencyBar");
  bar.innerHTML = "";
  const timedTotal = Object.values(byStage).reduce((a, b) => a + b, 0) || 1;
  ["stt", "retrieval", "generation"].forEach((stage) => {
    if (byStage[stage] === undefined) return;
    const seg = document.createElement("div");
    seg.className = `latency-seg ${stage}`;
    seg.style.width = `${(byStage[stage] / timedTotal) * 100}%`;
    seg.title = `${stage}: ${byStage[stage].toFixed(1)} ms`;
    bar.appendChild(seg);
  });

  const target = document.getElementById("latencyTarget");
  if (byStage.retrieval !== undefined) {
    const pass = byStage.retrieval < 200;
    target.innerHTML =
      `retrieval (chunking + vector search): <strong class="${pass ? "pass" : "fail"}">` +
      `${byStage.retrieval.toFixed(1)} ms — ${pass ? "under" : "over"} the 200ms target` +
      `${pass ? " ✓" : " ✗"}</strong>`;
  } else {
    target.textContent = "retrieval was skipped for this query (blocked earlier in the pipeline).";
  }

  const latencyList = document.getElementById("latencyList");
  latencyList.innerHTML = "";
  const STAGE_LABEL = { stt: "speech-to-text (network)", retrieval: "chunking + vector retrieval", generation: "LLM generation (network)" };
  timings.forEach((t) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot-${t.stage}">${STAGE_LABEL[t.stage] || t.stage}</span><span>${t.ms.toFixed(1)} ms</span>`;
    latencyList.appendChild(li);
  });
  const totalLi = document.createElement("li");
  totalLi.innerHTML = `<span>total (incl. external network calls)</span><span>${(data.total_ms ?? 0).toFixed(1)} ms</span>`;
  latencyList.appendChild(totalLi);
}

function renderStages(data) {
  const timings = {};
  (data.timings || []).forEach((t) => { timings[t.stage] = t.ms; });
  const status = data.status;

  const set = (name, state, text) => {
    const el = stagesPanel.querySelector(`.stage[data-stage="${name}"]`);
    el.classList.remove("active", "skipped", "failed");
    if (state) el.classList.add(state);
    el.querySelector(".stage-ms").textContent = text;
  };

  if (timings.stt !== undefined) {
    set("stt", "active", `${timings.stt.toFixed(0)}ms`);
  } else {
    set("stt", "skipped", "text input");
  }

  if (status === "unsafe") {
    set("guardrail-in", "failed", "blocked");
    set("retrieval", "skipped", "—");
    set("guardrail-topic", "skipped", "—");
    set("generation", "skipped", "—");
    return;
  }
  set("guardrail-in", "active", "passed");

  if (timings.retrieval === undefined) {
    set("retrieval", "skipped", "—");
    set("guardrail-topic", "skipped", "—");
    set("generation", "skipped", "—");
    return;
  }
  set("retrieval", "active", `${timings.retrieval.toFixed(0)}ms`);

  if (status === "offtopic") {
    set("guardrail-topic", "failed", "off-topic");
    set("generation", "skipped", "—");
    return;
  }

  if (timings.generation !== undefined) {
    set("generation", "active", `${timings.generation.toFixed(0)}ms`);
  } else {
    set("generation", "skipped", "—");
  }

  if (status === "ungrounded") {
    set("guardrail-topic", "failed", "ungrounded");
  } else if (status === "ok") {
    set("guardrail-topic", "active", "passed");
  } else {
    set("guardrail-topic", "skipped", "—");
  }
}
