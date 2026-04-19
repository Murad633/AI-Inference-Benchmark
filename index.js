const MODELS = {
  "coco-v2": {
    label: "COCO-SSD MobileNetV2",
    task: "Object Detection",
    desc: "MobileNetV2 backbone · 80 COCO classes · object detection.",
    color: "var(--c-coco-v2)",
    load: () => cocoSsd.load({ base: "mobilenet_v2" }),
    run: (m, v) => m.detect(v),
    fmt: (ps) =>
      ps.map((p) => ({ label: p.class, conf: p.score, bbox: p.bbox })),
  },
  "coco-lite": {
    label: "COCO-SSD Lite (MNv1)",
    task: "Object Detection",
    desc: "Lite MNv1 backbone · faster, slight accuracy trade-off.",
    color: "var(--c-coco-lite)",
    load: () => cocoSsd.load({ base: "lite_mobilenet_v2" }),
    run: (m, v) => m.detect(v),
    fmt: (ps) =>
      ps.map((p) => ({ label: p.class, conf: p.score, bbox: p.bbox })),
  },
  mobilenet: {
    label: "MobileNet v2",
    task: "Image Classification",
    desc: "ImageNet 1000-class classification · top-3 shown.",
    color: "var(--c-mobilenet)",
    load: () => mobilenet.load({ version: 2, alpha: 1.0 }),
    run: (m, v) => m.classify(v, 3),
    fmt: (ps) =>
      ps.map((p) => ({
        label: p.className.split(",")[0],
        conf: p.probability,
        bbox: null,
      })),
  },
  blazeface: {
    label: "BlazeFace",
    task: "Face Detection",
    desc: "Sub-millisecond face detector · mobile GPU optimised.",
    color: "var(--c-blazeface)",
    load: () => blazeface.load(),
    run: (m, v) => m.estimateFaces(v, false),
    fmt: (ps) =>
      ps.map((p) => {
        const [x, y] = p.topLeft,
          [x2, y2] = p.bottomRight;
        return {
          label: "face",
          conf: p.probability[0],
          bbox: [x, y, x2 - x, y2 - y],
        };
      }),
  },
};
const BACKENDS = {
  webgl: {
    label: "WebGL (GPU)",
    color: "var(--c-webgl)",
    tfName: "webgl",
  },
  wasm: { label: "WebAssembly", color: "var(--c-wasm)", tfName: "wasm" },
  cpu: { label: "CPU (Pure JS)", color: "var(--c-cpu)", tfName: "cpu" },
};

let activeModel = null,
  activeModelK = null,
  activeBackend = "webgl";
let modelCache = {},
  loopRunning = false,
  rafHandle = null;
let lastTS = 0,
  fCount = 0,
  fAccum = 0,
  displayFPS = 0;
let latHist = [],
  fpsHist = [],
  objHist = [],
  respHist = [],
  smthHist = [],
  iaccHist = [],
  ramHist = [];
let recording = false,
  log = [];
let statsByModel = {},
  statsByBackend = {},
  stageData = {};

const stageKey = (mk, bk) => `${mk}:${bk}`;
function getStage(mk, bk) {
  const k = stageKey(mk, bk);
  if (!stageData[k])
    stageData[k] = {
      setup_ms: null,
      warmup_ms: null,
      pred: [],
      frameCount: 0,
    };
  return stageData[k];
}
const getHeapMB = () =>
  performance.memory
    ? (performance.memory.usedJSHeapSize / 1048576).toFixed(1)
    : null;
const getHeapLimitMB = () =>
  performance.memory
    ? (performance.memory.jsHeapSizeLimit / 1048576).toFixed(0)
    : null;

const video = document.getElementById("webcam"),
  canvas = document.getElementById("overlay"),
  ctx = canvas.getContext("2d");
const statusL = document.getElementById("status-layer"),
  statusMsg = document.getElementById("status-msg");
const recOverlay = document.getElementById("rec-overlay"),
  bkBadge = document.getElementById("backend-badge");
const vLat = document.getElementById("v-lat"),
  vFps = document.getElementById("v-fps");
const vWarmup = document.getElementById("v-warmup"),
  vSetup = document.getElementById("v-setup");
const vResp = document.getElementById("v-resp"),
  vSmth = document.getElementById("v-smth"),
  vIacc = document.getElementById("v-iacc");
const vRam = document.getElementById("v-ram"),
  vRamSub = document.getElementById("v-ram-sub");
const miName = document.getElementById("mi-name"),
  miTask = document.getElementById("mi-task");
const miDesc = document.getElementById("mi-desc"),
  detList = document.getElementById("det-list");

async function startWebcam() {
  setMsg("Requesting camera…");
  const s = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  video.srcObject = s;
  return new Promise((r) => {
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      r();
    };
  });
}

async function setBackend(bk) {
  setMsg(`Switching to ${BACKENDS[bk].label}…`);
  await tf.setBackend(BACKENDS[bk].tfName);
  await tf.ready();
  activeBackend = bk;
  bkBadge.textContent = BACKENDS[bk].label;
  bkBadge.style.color = bkBadge.style.borderColor = `var(--c-${bk})`;
  document
    .querySelectorAll(".backend-pill")
    .forEach((b) => b.classList.remove("active"));
  document.querySelector(`[data-backend="${bk}"]`).classList.add("active");
}

async function loadModel(mk) {
  const ck = `${activeBackend}:${mk}`,
    sd = getStage(mk, activeBackend);
  if (modelCache[ck]) return modelCache[ck];
  setMsg(`Loading ${MODELS[mk].label}…`);
  const t0 = performance.now();
  const m = await MODELS[mk].load();
  sd.setup_ms = performance.now() - t0;
  modelCache[ck] = m;
  document.querySelector(`[data-model="${mk}"]`).classList.add("cached");
  return m;
}

async function startModel(mk) {
  loopRunning = false;
  if (rafHandle) cancelAnimationFrame(rafHandle);
  resetPerfState();
  statusL.classList.remove("gone");
  activeModelK = mk;
  activeModel = await loadModel(mk);
  const reg = MODELS[mk];
  miName.textContent = reg.label;
  miTask.textContent = reg.task;
  miTask.style.color = miTask.style.borderColor = reg.color;
  miDesc.textContent = reg.desc;
  document
    .querySelectorAll(".model-pill")
    .forEach((b) => b.classList.remove("active"));
  document.querySelector(`[data-model="${mk}"]`).classList.add("active");
  statusL.classList.add("gone");
  loopRunning = true;
  rafHandle = requestAnimationFrame(inferLoop);
}

function resetPerfState() {
  lastTS = 0;
  fCount = 0;
  fAccum = 0;
  displayFPS = 0;
  latHist = [];
  fpsHist = [];
  objHist = [];
  respHist = [];
  smthHist = [];
  iaccHist = [];
  ramHist = [];
}

async function inferLoop(ts) {
  if (!loopRunning) return;
  const sd = getStage(activeModelK, activeBackend);
  sd.frameCount++;
  const t0 = performance.now();
  let preds = [];
  try {
    const raw = await MODELS[activeModelK].run(activeModel, video);
    preds = MODELS[activeModelK].fmt(raw);
  } catch (_) {}
  const lat = performance.now() - t0;
  if (sd.warmup_ms === null) {
    sd.warmup_ms = lat;
  } else {
    sd.pred.push(lat);
  }
  if (lastTS > 0) {
    const d = ts - lastTS;
    fAccum += d;
    fCount++;
    if (fAccum >= 500) {
      displayFPS = Math.round(fCount / (fAccum / 1000));
      fCount = 0;
      fAccum = 0;
    }
  }
  lastTS = ts;
  const avgConf = preds.length
    ? preds.reduce((s, p) => s + (p.conf || 0), 0) / preds.length
    : 0;
  const resp_rpm = 60000 / lat,
    smth_fps = displayFPS,
    iacc_norm = avgConf;
  const heapMB = getHeapMB(),
    heapNum = heapMB !== null ? parseFloat(heapMB) : 0;
  latHist.push(lat);
  fpsHist.push(displayFPS);
  objHist.push(preds.length);
  respHist.push(resp_rpm);
  smthHist.push(smth_fps);
  iaccHist.push(iacc_norm);
  ramHist.push(heapNum);
  if (latHist.length > 60) {
    latHist.shift();
    fpsHist.shift();
    objHist.shift();
  }
  if (respHist.length > 60) {
    respHist.shift();
    smthHist.shift();
    iaccHist.shift();
    ramHist.shift();
  }
  if (!statsByModel[activeModelK])
    statsByModel[activeModelK] = {
      lat: [],
      fps: [],
      obj: [],
      resp: [],
      smth: [],
      iacc: [],
    };
  const sm = statsByModel[activeModelK];
  sm.lat.push(lat);
  sm.fps.push(displayFPS);
  sm.obj.push(preds.length);
  sm.resp.push(resp_rpm);
  sm.smth.push(smth_fps);
  sm.iacc.push(iacc_norm);
  if (!statsByBackend[activeBackend])
    statsByBackend[activeBackend] = {
      lat: [],
      fps: [],
      resp: [],
      smth: [],
      iacc: [],
    };
  const sb = statsByBackend[activeBackend];
  sb.lat.push(lat);
  sb.fps.push(displayFPS);
  sb.resp.push(resp_rpm);
  sb.smth.push(smth_fps);
  sb.iacc.push(iacc_norm);
  drawFrame(preds);
  vLat.textContent = lat.toFixed(1);
  vLat.className = "val" + (lat > 200 ? " red" : lat > 80 ? " yel" : " grn");
  vFps.textContent = displayFPS || "—";
  vSetup.textContent =
    sd.setup_ms !== null ? sd.setup_ms.toFixed(0) + "ms" : "—";
  vWarmup.textContent =
    sd.warmup_ms !== null ? sd.warmup_ms.toFixed(0) + "ms" : "—";
  vResp.textContent = resp_rpm.toFixed(0);
  vSmth.textContent = smth_fps || "—";
  vIacc.textContent = iacc_norm.toFixed(3);
  if (heapMB !== null) {
    vRam.textContent = heapMB;
    const lim = getHeapLimitMB();
    vRamSub.textContent = lim ? `/ ${lim} MB` : "";
  } else {
    vRam.textContent = "N/A";
    vRamSub.textContent = "Chrome/Edge only";
  }
  const cls = [...new Set(preds.map((p) => p.label))];
  detList.innerHTML = cls
    .map((c) => {
      const col = MODELS[activeModelK].color;
      return `<li class="dtag" style="color:${col};border-color:${col}33">${c}</li>`;
    })
    .join("");
  updateCharts();
  updateStageTable();
  updateCmp();
  if (recording) {
    log.push({
      n: log.length + 1,
      time: new Date().toLocaleTimeString(),
      model: MODELS[activeModelK].label,
      task: MODELS[activeModelK].task,
      backend: BACKENDS[activeBackend].label,
      lat: lat.toFixed(2),
      fps: displayFPS,
      obj: preds.length,
      resp: resp_rpm.toFixed(1),
      smth: smth_fps,
      iacc: iacc_norm.toFixed(4),
      ram: heapMB !== null ? heapMB : "N/A",
    });
    renderLog();
  }
  rafHandle = requestAnimationFrame(inferLoop);
}

function drawFrame(preds) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,.02)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 64) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 64) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  const cM = {
    "coco-v2": "#22d3ee",
    "coco-lite": "#a78bfa",
    mobilenet: "#4ade80",
    blazeface: "#fb923c",
  };
  const col = cM[activeModelK];
  preds.forEach((p, i) => {
    if (!p.bbox) {
      ctx.fillStyle = "rgba(0,0,0,.5)";
      ctx.fillRect(8, 8 + i * 26, canvas.width - 16, 22);
      ctx.fillStyle = col;
      ctx.font = '600 12px "Azeret Mono",monospace';
      ctx.fillText(
        `${p.label}  ${(p.conf * 100).toFixed(1)}%`,
        16,
        24 + i * 26,
      );
      return;
    }
    const [x, y, w, h] = p.bbox,
      conf = (p.conf * 100).toFixed(1);
    ctx.shadowColor = col;
    ctx.shadowBlur = 8;
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.8;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0;
    const tk = 9;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    [
      [
        [x, y + tk],
        [x, y],
        [x + tk, y],
      ],
      [
        [x + w - tk, y],
        [x + w, y],
        [x + w, y + tk],
      ],
      [
        [x, y + h - tk],
        [x, y + h],
        [x + tk, y + h],
      ],
      [
        [x + w - tk, y + h],
        [x + w, y + h],
        [x + w, y + h - tk],
      ],
    ].forEach((pts) => {
      ctx.beginPath();
      pts.forEach(([px, py], j) =>
        j === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py),
      );
      ctx.stroke();
    });
    ctx.font = '600 10px "Azeret Mono",monospace';
    const tw = ctx.measureText(`${p.label} ${conf}%`).width;
    const py2 = y > 22 ? y - 22 : y + 4;
    ctx.fillStyle = col;
    ctx.fillRect(x, py2, tw + 10, 18);
    ctx.fillStyle = "#000";
    ctx.fillText(`${p.label} ${conf}%`, x + 5, py2 + 13);
  });
}

let charts = {};
function initCharts() {
  const make = (id, lbl, col) => {
    const o = {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          grid: { color: "#1a2235" },
          ticks: {
            color: "#4d5e7a",
            font: { size: 8, family: "Azeret Mono" },
          },
        },
      },
    };
    return new Chart(document.getElementById(id), {
      type: "line",
      data: {
        labels: Array(60).fill(""),
        datasets: [
          {
            label: lbl,
            data: [],
            borderColor: col,
            backgroundColor: col + "18",
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
          },
        ],
      },
      options: o,
    });
  };
  charts.lat = make("c-lat", "Pred ms", "#22d3ee");
  charts.fps = make("c-fps", "FPS", "#94a3b8");
  charts.obj = make("c-obj", "Objects", "#fb923c");
  charts.resp = make("c-resp", "Resp", "#38bdf8");
  charts.smth = make("c-smth", "Smth", "#a3e635");
  charts.ram = make("c-ram", "Heap MB", "#34d399");
}

function updateCharts() {
  if (!charts.lat) return;
  const cM = {
    "coco-v2": "#22d3ee",
    "coco-lite": "#a78bfa",
    mobilenet: "#4ade80",
    blazeface: "#fb923c",
  };
  const col = cM[activeModelK];
  [
    [charts.lat, latHist, col],
    [charts.fps, fpsHist, "#94a3b8"],
    [charts.obj, objHist, "#fb923c"],
    [charts.resp, respHist, "#38bdf8"],
    [charts.smth, smthHist, "#a3e635"],
    [charts.ram, ramHist, "#34d399"],
  ].forEach(([c, d, clr]) => {
    c.data.datasets[0].data = [...d];
    c.data.datasets[0].borderColor = clr;
    c.data.datasets[0].backgroundColor = clr + "18";
    c.update("none");
  });
}

function stddev(a) {
  if (a.length < 2) return 0;
  const m = a.reduce((s, x) => s + x, 0) / a.length;
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length);
}
function f(v, d = 1) {
  return v !== null && v !== undefined ? v.toFixed(d) : "—";
}

function updateStageTable() {
  const body = document.getElementById("lat-body");
  if (!Object.keys(stageData).length) {
    body.innerHTML =
      '<p class="no-data-msg">Run a model to populate this table.</p>';
    return;
  }
  const cM = {
    "coco-v2": "#22d3ee",
    "coco-lite": "#a78bfa",
    mobilenet: "#4ade80",
    blazeface: "#fb923c",
  };
  let html = "";
  for (const mk of Object.keys(MODELS)) {
    const bkEntries = Object.entries(BACKENDS)
      .map(([bk, bkInfo]) => {
        const sd = stageData[stageKey(mk, bk)];
        if (!sd) return null;
        return { bk, bkInfo, sd };
      })
      .filter(Boolean);
    if (!bkEntries.length) continue;
    const col = cM[mk];
    html += `<div class="lat-model-block"><div class="lat-model-title"><span style="color:${col}">${MODELS[mk].label}</span><span class="task-tag" style="color:${col};border-color:${col}">${MODELS[mk].task}</span></div><div class="tbl-wrap"><table class="stage-table"><thead><tr><th>Backend</th><th class="setup-h">Setup ms</th><th class="warmup-h">Warmup ms</th><th class="pred-h">Avg ms</th><th class="pred-h">Min ms</th><th class="pred-h">Max ms</th><th class="pred-h">σ ms</th><th>Frames</th></tr></thead><tbody>`;
    for (const { bk, bkInfo, sd } of bkEntries) {
      const pa = sd.pred.length
        ? sd.pred.reduce((s, x) => s + x, 0) / sd.pred.length
        : null;
      const pmn = sd.pred.length ? Math.min(...sd.pred) : null,
        pmx = sd.pred.length ? Math.max(...sd.pred) : null;
      const psd = sd.pred.length ? stddev(sd.pred) : null;
      const s = sd.setup_ms ?? 0,
        w = sd.warmup_ms ?? 0,
        p = pa ?? 0,
        total = s + w + p || 1;
      const sp = Math.round((s / total) * 100),
        wp = Math.round((w / total) * 100),
        pp = 100 - sp - wp;
      html += `<tr><td class="bk-label" style="color:${bkInfo.color}">${bkInfo.label}</td><td class="stage-val setup-v">${sd.setup_ms !== null ? f(sd.setup_ms, 0) : '<span class="na-cell">cached</span>'}</td><td class="stage-val warmup-v">${sd.warmup_ms !== null ? f(sd.warmup_ms, 1) : '<span class="na-cell">—</span>'}</td><td class="stage-val pred-v">${f(pa, 1)}</td><td style="color:var(--muted)">${f(pmn, 1)}</td><td style="color:var(--muted)">${f(pmx, 1)}</td><td style="color:var(--muted)">${f(psd, 1)}</td><td style="color:var(--muted)">${sd.frameCount}</td></tr><tr><td colspan="8" style="padding:3px 8px 8px;border-bottom:1px solid var(--border)"><div class="stage-bar-row">Setup ${sp}% · Warmup ${wp}% · Prediction ${pp}%</div><div class="stage-bars"><div class="seg s" style="width:${sp}%"></div><div class="seg w" style="width:${wp}%"></div><div class="seg p" style="width:${pp}%"></div></div></td></tr>`;
    }
    html += `</tbody></table></div></div>`;
  }
  document.getElementById("lat-body").innerHTML =
    html || '<p class="no-data-msg">Run a model to populate this table.</p>';
}

const avg = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
const fmt2 = (v) => (v !== null ? v.toFixed(1) : "—");

function updateCmp() {
  const cM = {
    "coco-v2": "#22d3ee",
    "coco-lite": "#a78bfa",
    mobilenet: "#4ade80",
    blazeface: "#fb923c",
  };
  const bM = {
    webgl: "var(--c-webgl)",
    wasm: "var(--c-wasm)",
    cpu: "var(--c-cpu)",
  };
  const mK = Object.keys(statsByModel);
  if (mK.length)
    document.getElementById("cmp-model").innerHTML = mK
      .map((k) => {
        const s = statsByModel[k],
          c = cM[k];
        return `<div class="cmp-card" style="border-color:${c}44"><h4 style="color:${c}">${MODELS[k].label}</h4><div class="cmp-row"><span class="k">Avg Pred. ms</span><span class="v" style="color:${c}">${fmt2(avg(s.lat))}</span></div><div class="cmp-row"><span class="k">Min/Max ms</span><span class="v">${fmt2(Math.min(...s.lat))}/${fmt2(Math.max(...s.lat))}</span></div><div class="cmp-row"><span class="k">Avg FPS</span><span class="v">${fmt2(avg(s.fps))}</span></div><div class="cmp-row"><span class="k" style="color:var(--c-resp)">Responsiveness</span><span class="v" style="color:var(--c-resp)">${fmt2(avg(s.resp))}</span></div><div class="cmp-row"><span class="k" style="color:var(--c-smth)">Smoothness</span><span class="v" style="color:var(--c-smth)">${fmt2(avg(s.smth))}</span></div><div class="cmp-row"><span class="k" style="color:var(--c-iacc)">Infer. Acc.</span><span class="v" style="color:var(--c-iacc)">${avg(s.iacc) !== null ? avg(s.iacc).toFixed(3) : "—"}</span></div><div class="cmp-row"><span class="k">Samples</span><span class="v">${s.lat.length}</span></div></div>`;
      })
      .join("");
  const bK = Object.keys(statsByBackend);
  if (bK.length)
    document.getElementById("cmp-backend").innerHTML = bK
      .map((k) => {
        const s = statsByBackend[k],
          c = bM[k];
        return `<div class="cmp-card" style="border-color:${c}44"><h4 style="color:${c}">${BACKENDS[k].label}</h4><div class="cmp-row"><span class="k">Avg Pred. ms</span><span class="v" style="color:${c}">${fmt2(avg(s.lat))}</span></div><div class="cmp-row"><span class="k">Min/Max ms</span><span class="v">${fmt2(Math.min(...s.lat))}/${fmt2(Math.max(...s.lat))}</span></div><div class="cmp-row"><span class="k">Avg FPS</span><span class="v">${fmt2(avg(s.fps))}</span></div><div class="cmp-row"><span class="k" style="color:var(--c-resp)">Responsiveness</span><span class="v" style="color:var(--c-resp)">${fmt2(avg(s.resp))}</span></div><div class="cmp-row"><span class="k" style="color:var(--c-smth)">Smoothness</span><span class="v" style="color:var(--c-smth)">${fmt2(avg(s.smth))}</span></div><div class="cmp-row"><span class="k" style="color:var(--c-iacc)">Infer. Acc.</span><span class="v" style="color:var(--c-iacc)">${avg(s.iacc) !== null ? avg(s.iacc).toFixed(3) : "—"}</span></div><div class="cmp-row"><span class="k">Samples</span><span class="v">${s.lat.length}</span></div></div>`;
      })
      .join("");
}

function renderLog() {
  document.getElementById("log-count").textContent = `${log.length} samples`;
  const cM = {
    "coco-v2": "#22d3ee",
    "coco-lite": "#a78bfa",
    mobilenet: "#4ade80",
    blazeface: "#fb923c",
  };
  const bM = {
    "WebGL (GPU)": "#22d3ee",
    WebAssembly: "#f59e0b",
    "CPU (Pure JS)": "#f87171",
  };
  document.getElementById("log-body").innerHTML = [...log]
    .reverse()
    .slice(0, 30)
    .map((r) => {
      const mk =
        Object.keys(MODELS).find((k) => MODELS[k].label === r.model) ||
        "coco-v2";
      return `<tr><td>${r.n}</td><td style="color:var(--muted)">${r.time}</td><td style="color:${cM[mk]}">${r.model}</td><td style="color:var(--muted)">${r.task}</td><td style="color:${bM[r.backend] || "#fff"}">${r.backend}</td><td style="color:var(--c-pred)">${r.lat}</td><td>${r.fps}</td><td>${r.obj}</td><td style="color:var(--c-resp)">${r.resp}</td><td style="color:var(--c-smth)">${r.smth}</td><td style="color:var(--c-iacc)">${r.iacc}</td><td style="color:var(--c-ram)">${r.ram}</td></tr>`;
    })
    .join("");
}

function exportCSV() {
  if (!log.length) {
    alert("No data recorded. Press ⏺ Record first.");
    return;
  }
  let stageRows = [
    ["=== Stage Latency Summary (Setup / Warmup / Prediction) ==="],
    [
      "Model",
      "Backend",
      "Setup_ms",
      "Warmup_ms",
      "Pred_avg_ms",
      "Pred_min_ms",
      "Pred_max_ms",
      "Pred_sigma_ms",
      "Frames_n",
    ],
  ];
  for (const [k, sd] of Object.entries(stageData)) {
    const [mk, bk] = k.split(":");
    const pa = sd.pred.length
      ? sd.pred.reduce((s, x) => s + x, 0) / sd.pred.length
      : null;
    const pmn = sd.pred.length ? Math.min(...sd.pred) : null,
      pmx = sd.pred.length ? Math.max(...sd.pred) : null;
    const psd = sd.pred.length ? stddev(sd.pred) : null;
    stageRows.push([
      `"${MODELS[mk]?.label ?? mk}"`,
      `"${BACKENDS[bk]?.label ?? bk}"`,
      sd.setup_ms !== null ? sd.setup_ms.toFixed(1) : "cached",
      sd.warmup_ms !== null ? sd.warmup_ms.toFixed(1) : "—",
      pa !== null ? pa.toFixed(1) : "—",
      pmn !== null ? pmn.toFixed(1) : "—",
      pmx !== null ? pmx.toFixed(1) : "—",
      psd !== null ? psd.toFixed(1) : "—",
      sd.frameCount,
    ]);
  }
  const stageCsv = stageRows.map((r) => r.join(",")).join("\n");
  const hdr = [
    "#",
    "Time",
    "Model",
    "Task",
    "Backend",
    "Pred_ms",
    "FPS",
    "Detections",
    "Responsiveness_runs_per_min",
    "Smoothness_fps",
    "InferenceAccuracy_0to1",
    "HeapMB",
  ];
  const rows = log.map((r) => [
    r.n,
    r.time,
    `"${r.model}"`,
    `"${r.task}"`,
    `"${r.backend}"`,
    r.lat,
    r.fps,
    r.obj,
    r.resp,
    r.smth,
    r.iacc,
    r.ram,
  ]);
  const frameCsv = [hdr, ...rows].map((r) => r.join(",")).join("\n");
  const full = stageCsv + "\n\n=== Per-Frame Log ===\n" + frameCsv;
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([full], { type: "text/csv" })),
    download: `ai_benchmark_${Date.now()}.csv`,
  });
  a.click();
}

document
  .querySelectorAll(".model-pill")
  .forEach((b) =>
    b.addEventListener("click", () => startModel(b.dataset.model)),
  );
document.querySelectorAll(".backend-pill").forEach((b) =>
  b.addEventListener("click", async () => {
    const bk = b.dataset.backend;
    if (bk === activeBackend) return;
    const was = loopRunning;
    loopRunning = false;
    if (rafHandle) cancelAnimationFrame(rafHandle);
    statusL.classList.remove("gone");
    try {
      await setBackend(bk);
      resetPerfState();
      if (was && activeModelK) {
        activeModel = await loadModel(activeModelK);
        statusL.classList.add("gone");
        loopRunning = true;
        rafHandle = requestAnimationFrame(inferLoop);
      } else {
        statusL.classList.add("gone");
      }
    } catch (e) {
      setMsg(`Failed: ${e.message}`);
    }
  }),
);
document.getElementById("rec-btn").addEventListener("click", function () {
  recording = !recording;
  this.textContent = recording ? "⏹ Stop" : "⏺ Record";
  this.classList.toggle("on", recording);
  recOverlay.classList.toggle("show", recording);
});
document.getElementById("exp-btn").addEventListener("click", exportCSV);
function setMsg(m) {
  statusMsg.textContent = m;
}

(async () => {
  try {
    await tf.ready();
    await startWebcam();
    initCharts();
    await setBackend("webgl");
    await startModel("coco-v2");
  } catch (e) {
    setMsg("Error: " + e.message);
    console.error(e);
  }
})();
