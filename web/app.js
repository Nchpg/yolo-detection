// Traffic Vision — user interface.
// The detection pipeline itself lives in detector.js.

const CLASSES = [
  { name: 'Car',               color: '#ffb000' },
  { name: 'Number Plate',      color: '#00e5ff' },
  { name: 'Blur Number Plate', color: '#8b7bff' },
  { name: 'Two Wheeler',       color: '#46d17a' },
  { name: 'Auto',              color: '#ff8a3d' },
  { name: 'Bus',               color: '#ff4d6d' },
  { name: 'Truck',             color: '#d4ff3d' },
];

const DEFAULT_MODEL = 'models/best.onnx';
const NMS_IOU = 0.45;   // standard value; boxes overlapping more than this are merged
const WARMUP_MS = 10000;
const FORCE_WASM = 'traffic-vision:force-wasm';
const SAMPLE_VIDEOS = ['demo/sample1.mp4', 'demo/sample2.mp4'];
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';

const $ = (id) => document.getElementById(id);

const els = {
  video: $('video'), overlay: $('overlay'), stage: $('stage'), hint: $('hint'),
  ledModel: $('led-model'), ledVideo: $('led-video'), ledRun: $('led-run'),
  conf: $('conf'), outConf: $('out-conf'),
  optLabels: $('opt-labels'), optScores: $('opt-scores'),
  classes: $('classes'), samples: $('samples'), samplesList: $('samples-list'),
  btnPlay: $('btn-play'), btnSnap: $('btn-snap'),
  seek: $('seek'), time: $('t-time'),
  tMs: $('t-ms'), tFps: $('t-fps'), tN: $('t-n'),
};

const state = {
  session: null,
  inputW: 640,
  inputH: 640,
  nc: CLASSES.length,
  enabled: CLASSES.map(() => true),
  busy: false,
  detections: [],
  frameReady: false,
  msEma: null,
  fpsEma: null,
  lastRun: 0,
  running: false,
};

const ctx = els.overlay.getContext('2d');
const pre = document.createElement('canvas');
const preCtx = pre.getContext('2d', { willReadFrequently: true });

// Inference is far slower than playback, so the video runs ahead of the boxes.
// Two buffers: one holds the frame being analysed, the other the frame whose
// detections are on screen. They swap only once a run completes, so the picture
// and the boxes always come from the same frame.
const frames = [document.createElement('canvas'), document.createElement('canvas')];
const frameCtxs = frames.map((c) => c.getContext('2d'));
let shownFrame = 0;

/* ---------------- interface ---------------- */

function setLed(el, mode, title) {
  el.className = 'led' + (mode ? ' ' + mode : '');
  if (title) el.title = title;
}

function say(msg, kind = '') {
  els.hint.textContent = msg;
  els.hint.className = 'hint' + (kind ? ' ' + kind : '');
}

function buildClassList() {
  els.classes.innerHTML = '';
  CLASSES.forEach((c, i) => {
    const li = document.createElement('li');
    li.innerHTML =
      `<span class="swatch" style="background:${c.color}"></span>` +
      `<span class="cname">${c.name}</span><span class="cnum">0</span>`;
    li.onclick = () => {
      state.enabled[i] = !state.enabled[i];
      li.classList.toggle('off', !state.enabled[i]);
      draw();
    };
    els.classes.appendChild(li);
  });
}

function updateClassCounts(dets) {
  const tally = new Array(CLASSES.length).fill(0);
  dets.forEach((d) => { tally[d.cls]++; });
  [...els.classes.children].forEach((li, i) => {
    li.querySelector('.cnum').textContent = tally[i];
    li.classList.toggle('hit', tally[i] > 0);
  });
}

function bindSliders() {
  const sync = () => { els.outConf.textContent = Number(els.conf.value).toFixed(2); };
  els.conf.addEventListener('input', sync);
  sync();
  els.optLabels.addEventListener('change', draw);
  els.optScores.addEventListener('change', draw);
}

/** The samples are known up front, so they are on screen before anything is
 *  fetched: waiting for the model or for the probes below made them appear
 *  seconds late and pushed the rest of the panel down. The probes only take
 *  away what turns out to be missing from web/demo/. */
function buildSamples() {
  els.samplesList.innerHTML = '';
  for (const src of SAMPLE_VIDEOS) {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = src.split('/').pop();
    b.onclick = () => loadVideo(src, b.textContent);
    els.samplesList.appendChild(b);
    checkSample(src, b);
  }
  els.samples.hidden = !els.samplesList.children.length;
}

async function checkSample(src, chip) {
  let ok = false;
  try { ok = (await fetch(src, { method: 'HEAD' })).ok; } catch { /* missing */ }
  if (ok) return;
  chip.remove();
  els.samples.hidden = !els.samplesList.children.length;
}

function bindDrop(zone, accept, handler) {
  ['dragenter', 'dragover'].forEach((e) =>
    zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((e) =>
    zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', (ev) => {
    const file = ev.dataTransfer.files[0];
    if (file && accept(file)) handler(file);
    else say('Unexpected file type.', 'err');
  });
}

/* ---------------- model loading ---------------- */

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => {
      const err = new Error(`${label} timed out after ${ms} ms`);
      err.timedOut = true;
      reject(err);
    }, ms)),
  ]);
}

function pinned(key) {
  try { return sessionStorage.getItem(key) === '1'; } catch { return false; }
}

function pin(key) {
  try { sessionStorage.setItem(key, '1'); } catch { /* private mode, ignore */ }
}

/** Create a session and prove the backend actually runs.
 *  A backend can accept the model and then fail or hang on the first run, which
 *  would leave the page stuck with no error. The warm-up also compiles the graph,
 *  so the first real frame is no slower than the rest. */
async function openSession(source, providers) {
  const session = await ort.InferenceSession.create(source, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });
  const [, , h, w] = inputShape(session) || [1, 3, 640, 640];
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor('float32', new Float32Array(3 * h * w), [1, 3, h, w]);
  await withTimeout(session.run(feeds), WARMUP_MS, providers[0]);
  return { session, h, w };
}

async function loadModel(source, label) {
  setLed(els.ledModel, 'busy');
  say(`Loading model ${label} ...`);
  try {
    ort.env.wasm.wasmPaths = ORT_DIST;
    // WASM multithreading requires cross-origin isolation (see serve.py)
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;

    let opened = null;
    let ep = null;
    const forced = pinned(FORCE_WASM);
    for (const providers of forced ? [['wasm']] : [['webgpu'], ['wasm']]) {
      try {
        opened = await openSession(source, providers);
        ep = providers[0];
        break;
      } catch (err) {
        console.warn(`backend ${providers[0]} unusable:`, err.message);
        // A backend that fails outright leaves the runtime usable, so the next
        // one can be tried in place. One that hangs keeps the runtime locked
        // ("Session already started"), and only a fresh page recovers.
        if (err.timedOut) {
          pin(FORCE_WASM);
          say(`${providers[0]} did not respond, reloading on WASM...`, 'err');
          location.reload();
          return;
        }
      }
    }
    if (!opened) throw new Error('no execution backend available');

    state.session = opened.session;
    state.inputH = opened.h;
    state.inputW = opened.w;
    pre.width = state.inputW;
    pre.height = state.inputH;

    // the backend is worth knowing but not worth a panel: it lands in the tooltip
    const threads = ep === 'wasm' ? ` \u00b7 ${ort.env.wasm.numThreads} thread(s)` : '';
    setLed(els.ledModel, 'on', `${label} \u00b7 ${state.inputW}\u00d7${state.inputH} \u00b7 ${ep}${threads}`);
    say('Model loaded. Drop a video or pick a sample.', 'ok');
    refreshPlayButton();
  } catch (err) {
    console.error(err);
    setLed(els.ledModel, 'err');
    say(`Could not load the model: ${err.message}`, 'err');
  }
}

function inputShape(session) {
  // depending on the onnxruntime-web version, metadata is not always exposed
  const meta = session.inputMetadata;
  if (!meta || !meta[0]) return null;
  const dims = meta[0].shape || meta[0].dimensions;
  if (!dims || dims.length !== 4) return null;
  const [, , h, w] = dims;
  return Number.isInteger(h) && Number.isInteger(w) && h > 0 && w > 0 ? [1, 3, h, w] : null;
}

/* ---------------- video loading ---------------- */

function loadVideo(src, label) {
  els.video.src = src;
  els.video.load();
  els.video.onloadedmetadata = () => {
    els.overlay.width = els.video.videoWidth;
    els.overlay.height = els.video.videoHeight;
    for (const c of frames) {
      c.width = els.video.videoWidth;
      c.height = els.video.videoHeight;
    }
    els.stage.classList.remove('empty');
    setLed(els.ledVideo, 'on', label);
    state.detections = [];
    state.frameReady = false;
    say(`${label} · ${els.video.videoWidth}×${els.video.videoHeight}`, 'ok');
    refreshPlayButton();
    updateSeek();
    draw();
  };
  els.video.onerror = () => {
    setLed(els.ledVideo, 'err');
    say('The browser cannot read this video (try an H.264 MP4).', 'err');
  };
}

/** A loaded video has no decoded frame to draw from until it plays or seeks:
 *  drawImage then yields a blank canvas and the model finds nothing. Nudging
 *  the position forces a decode. */
function frameAvailable() {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    const nudge = () => {
      els.video.addEventListener('seeked', finish, { once: true });
      const d = els.video.duration;
      els.video.currentTime = Number.isFinite(d)
        ? Math.min(els.video.currentTime + 0.001, d)
        : els.video.currentTime;
      setTimeout(finish, 300);   // a no-op seek fires no event in some browsers
    };
    if (els.video.readyState >= 2) nudge();
    else els.video.addEventListener('loadeddata', nudge, { once: true });
  });
}

function clock(seconds) {
  if (!Number.isFinite(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  return `${m}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

function updateSeek() {
  if (state.scrubbing) return;
  const d = els.video.duration;
  els.seek.value = Number.isFinite(d) && d > 0 ? (els.video.currentTime / d) * 1000 : 0;
  els.time.textContent = `${clock(els.video.currentTime)} / ${clock(d)}`;
}

function refreshPlayButton() {
  const ready = state.session && els.video.src;
  els.btnPlay.disabled = !ready;
  els.btnSnap.disabled = !ready;
  els.seek.disabled = !ready;
}

/* ---------------- rendering ---------------- */

function draw() {
  const W = els.overlay.width;
  const H = els.overlay.height;
  ctx.clearRect(0, 0, W, H);

  // Cover the running video with the frame the boxes were computed from, so
  // picture and boxes never disagree. Seeking re-runs inference, so the frame
  // stays correct while paused too.
  if (state.frameReady) ctx.drawImage(frames[shownFrame], 0, 0);
  if (!state.detections.length) {
    els.tN.textContent = 0;
    return;
  }

  // stroke width follows resolution so it stays readable from 480p to 4K
  const unit = Math.max(1, Math.round(Math.min(W, H) / 360));
  const fontSize = Math.max(11, Math.round(Math.min(W, H) / 34));
  ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = 'alphabetic';

  let shown = 0;
  for (const d of state.detections) {
    if (!state.enabled[d.cls]) continue;
    shown++;
    const color = CLASSES[d.cls] ? CLASSES[d.cls].color : '#ffffff';
    const x = Math.max(0, d.x);
    const y = Math.max(0, d.y);
    const w = Math.min(W - x, d.w);
    const h = Math.min(H - y, d.h);

    ctx.strokeStyle = color;
    ctx.lineWidth = unit * 1.6;
    ctx.strokeRect(x, y, w, h);

    // corner brackets: instantly readable even against a busy background
    const arm = Math.min(w, h) * 0.22;
    ctx.lineWidth = unit * 3;
    ctx.beginPath();
    ctx.moveTo(x, y + arm); ctx.lineTo(x, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x + w - arm, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - arm);
    ctx.stroke();

    if (!els.optLabels.checked) continue;

    const name = CLASSES[d.cls] ? CLASSES[d.cls].name : `class ${d.cls}`;
    const text = els.optScores.checked ? `${name} ${(d.score * 100).toFixed(0)}%` : name;
    const padX = fontSize * 0.34;
    const boxH = fontSize * 1.32;
    const boxW = ctx.measureText(text).width + padX * 2;
    const above = y - boxH >= 0;
    const by = above ? y - boxH : y;

    ctx.fillStyle = color;
    ctx.fillRect(x - unit * 0.8, by, boxW, boxH);
    ctx.fillStyle = '#0b0d0e';
    ctx.fillText(text, x - unit * 0.8 + padX, by + boxH - fontSize * 0.34);
  }
  els.tN.textContent = shown;
}

/* ---------------- loop ---------------- */

async function infer() {
  if (state.busy || !state.session) return;
  state.busy = true;
  setLed(els.ledRun, 'busy');
  const t0 = performance.now();
  try {
    // freeze the frame into the spare buffer: the video keeps playing meanwhile
    const next = 1 - shownFrame;
    frameCtxs[next].drawImage(els.video, 0, 0, frames[next].width, frames[next].height);

    const { detections, nc } = await Detector.run(
      state.session, preCtx, frames[next],
      state.inputW, state.inputH,
      frames[next].width, frames[next].height,
      { conf: parseFloat(els.conf.value), iou: NMS_IOU },
    );

    // swap picture and boxes together, never one without the other
    shownFrame = next;
    state.frameReady = true;
    state.nc = nc;
    state.detections = detections;
    updateClassCounts(detections);

    const ms = performance.now() - t0;
    state.msEma = state.msEma === null ? ms : state.msEma * 0.8 + ms * 0.2;
    const fps = 1000 / Math.max(ms, 1);
    state.fpsEma = state.fpsEma === null ? fps : state.fpsEma * 0.8 + fps * 0.2;
    els.tMs.textContent = state.msEma.toFixed(0);
    els.tFps.textContent = state.fpsEma.toFixed(1);
    setLed(els.ledRun, 'on');
  } catch (err) {
    console.error(err);
    setLed(els.ledRun, 'err');
    say(`Inference error: ${err.message}`, 'err');
    state.running = false;
  } finally {
    state.busy = false;
  }
}

function onFrame() {
  if (!els.video.paused && !els.video.ended) infer();
  draw();
  updateSeek();
  schedule();
}

function schedule() {
  if (!state.running) return;
  if (els.video.requestVideoFrameCallback) els.video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

/* ---------------- actions ---------------- */

els.btnPlay.onclick = async () => {
  if (!els.video.paused) {
    els.video.pause();
    return;
  }
  // analyse the first frame before playing, otherwise the video runs at full
  // speed until the first result lands and only then drops to inference pace
  if (!state.frameReady) {
    await frameAvailable();
    await infer();
    draw();
  }
  els.video.play().catch((e) => say(e.message, 'err'));
};

els.seek.addEventListener('input', () => {
  state.scrubbing = true;
  const d = els.video.duration;
  if (Number.isFinite(d)) els.video.currentTime = (els.seek.value / 1000) * d;
  els.time.textContent = `${clock(els.video.currentTime)} / ${clock(d)}`;
});
els.seek.addEventListener('change', () => { state.scrubbing = false; });
els.video.addEventListener('timeupdate', updateSeek);

// the loop follows the video's real state, so the button and the native
// controls (play, pause, seeking) drive it interchangeably
els.video.addEventListener('play', () => {
  state.running = true;
  els.btnPlay.textContent = 'Pause';
  schedule();
});
['pause', 'ended'].forEach((ev) => els.video.addEventListener(ev, () => {
  state.running = false;
  els.btnPlay.textContent = 'Start';
}));
els.video.addEventListener('seeked', () => { if (els.video.paused) infer().then(draw); });

els.btnSnap.onclick = () => {
  const c = document.createElement('canvas');
  c.width = els.overlay.width;
  c.height = els.overlay.height;
  const g = c.getContext('2d');
  g.drawImage(els.video, 0, 0, c.width, c.height);
  g.drawImage(els.overlay, 0, 0);
  const a = document.createElement('a');
  a.download = `traffic-vision-${Date.now()}.png`;
  a.href = c.toDataURL('image/png');
  a.click();
};

$('btn-video').onclick = () => $('file-video').click();

$('file-video').onchange = (e) => {
  const f = e.target.files[0];
  if (f) loadVideo(URL.createObjectURL(f), f.name);
};

const isVideo = (f) => f.type.startsWith('video/');
bindDrop($('drop-video'), isVideo, (f) => loadVideo(URL.createObjectURL(f), f.name));
bindDrop(els.stage, isVideo, (f) => loadVideo(URL.createObjectURL(f), f.name));

/* ---------------- boot ---------------- */

async function boot() {
  buildClassList();
  bindSliders();
  buildSamples();
  els.stage.classList.add('empty');

  try {
    const head = await fetch(DEFAULT_MODEL, { method: 'HEAD' });
    if (head.ok) await loadModel(DEFAULT_MODEL, DEFAULT_MODEL.split('/').pop());
    else say(`${DEFAULT_MODEL} is missing. Put best.onnx there and reload.`, 'err');
  } catch {
    say(`${DEFAULT_MODEL} is missing. Put best.onnx there and reload.`, 'err');
  }
}

boot();
