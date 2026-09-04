/* ============================================================
   TRAFFIC VISION — interface
   Le pipeline de detection lui-meme vit dans detector.js.
   ============================================================ */

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
const SAMPLE_VIDEOS = ['demo/sample1.mp4', 'demo/sample2.mp4'];
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';

const $ = (id) => document.getElementById(id);

const els = {
  video: $('video'), overlay: $('overlay'), stage: $('stage'), hint: $('hint'),
  ledModel: $('led-model'), ledVideo: $('led-video'), ledRun: $('led-run'),
  mName: $('m-name'), mInput: $('m-input'), mOutput: $('m-output'), mEp: $('m-ep'),
  conf: $('conf'), iou: $('iou'), stride: $('stride'),
  outConf: $('out-conf'), outIou: $('out-iou'), outStride: $('out-stride'),
  optLabels: $('opt-labels'), optScores: $('opt-scores'),
  classes: $('classes'), samples: $('samples'), samplesList: $('samples-list'),
  btnPlay: $('btn-play'), btnSnap: $('btn-snap'),
  tMs: $('t-ms'), tFps: $('t-fps'), tN: $('t-n'),
};

const state = {
  session: null,
  inputW: 640,
  inputH: 640,
  nc: CLASSES.length,
  enabled: CLASSES.map(() => true),
  busy: false,
  frame: 0,
  detections: [],
  msEma: null,
  fpsEma: null,
  lastRun: 0,
  running: false,
};

const ctx = els.overlay.getContext('2d');
const pre = document.createElement('canvas');
const preCtx = pre.getContext('2d', { willReadFrequently: true });

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
  const link = (input, out, fmt) => {
    const sync = () => { out.textContent = fmt(input.value); };
    input.addEventListener('input', sync);
    sync();
  };
  link(els.conf, els.outConf, (v) => Number(v).toFixed(2));
  link(els.iou, els.outIou, (v) => Number(v).toFixed(2));
  link(els.stride, els.outStride, (v) => v);
  els.optLabels.addEventListener('change', draw);
  els.optScores.addEventListener('change', draw);
}

function bindDrop(zone, accept, handler) {
  ['dragenter', 'dragover'].forEach((e) =>
    zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((e) =>
    zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', (ev) => {
    const file = ev.dataTransfer.files[0];
    if (file && accept(file)) handler(file);
    else say('Type de fichier inattendu.', 'err');
  });
}

/* ---------------- chargement du modele ---------------- */

async function loadModel(source, label) {
  setLed(els.ledModel, 'busy');
  say(`Chargement du modèle ${label} ...`);
  try {
    ort.env.wasm.wasmPaths = ORT_DIST;
    // le multithread WASM exige l'isolation cross-origin (voir serve.py)
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 1;

    let session = null;
    let ep = null;
    for (const providers of [['webgpu'], ['wasm']]) {
      try {
        session = await ort.InferenceSession.create(source, {
          executionProviders: providers,
          graphOptimizationLevel: 'all',
        });
        ep = providers[0];
        break;
      } catch (err) {
        console.warn(`backend ${providers[0]} indisponible :`, err.message);
      }
    }
    if (!session) throw new Error('aucun backend disponible');

    state.session = session;

    const shape = inputShape(session);
    if (shape) { state.inputH = shape[2]; state.inputW = shape[3]; }
    pre.width = state.inputW;
    pre.height = state.inputH;

    els.mName.textContent = label;
    els.mInput.textContent = `${state.inputW}×${state.inputH}`;
    els.mOutput.textContent = '—';
    els.mEp.textContent = ep + (ep === 'wasm' ? ` · ${ort.env.wasm.numThreads} thread(s)` : '');
    setLed(els.ledModel, 'on', label);
    say('Modèle chargé. Déposez une vidéo pour lancer la détection.', 'ok');
    refreshPlayButton();
  } catch (err) {
    console.error(err);
    setLed(els.ledModel, 'err');
    say(`Échec du chargement du modèle : ${err.message}`, 'err');
  }
}

function inputShape(session) {
  // selon la version d'onnxruntime-web, les metadonnees ne sont pas toujours exposees
  const meta = session.inputMetadata;
  if (!meta || !meta[0]) return null;
  const dims = meta[0].shape || meta[0].dimensions;
  if (!dims || dims.length !== 4) return null;
  const [, , h, w] = dims;
  return Number.isInteger(h) && Number.isInteger(w) && h > 0 && w > 0 ? [1, 3, h, w] : null;
}

/* ---------------- chargement video ---------------- */

function loadVideo(src, label) {
  els.video.src = src;
  els.video.load();
  els.video.onloadedmetadata = () => {
    els.overlay.width = els.video.videoWidth;
    els.overlay.height = els.video.videoHeight;
    els.stage.classList.remove('empty');
    setLed(els.ledVideo, 'on', label);
    state.frame = 0;
    state.detections = [];
    say(`${label} · ${els.video.videoWidth}×${els.video.videoHeight}`, 'ok');
    refreshPlayButton();
    draw();
  };
  els.video.onerror = () => {
    setLed(els.ledVideo, 'err');
    say('Vidéo illisible par le navigateur (essayez un MP4 H.264).', 'err');
  };
}

function refreshPlayButton() {
  const ready = state.session && els.video.src;
  els.btnPlay.disabled = !ready;
  els.btnSnap.disabled = !ready;
}

/* ---------------- rendu ---------------- */

function draw() {
  const W = els.overlay.width;
  const H = els.overlay.height;
  ctx.clearRect(0, 0, W, H);
  if (!state.detections.length) return;

  // l'epaisseur suit la resolution pour rester lisible en 480p comme en 4K
  const unit = Math.max(1, Math.round(Math.min(W, H) / 360));
  const fontSize = Math.max(11, Math.round(Math.min(W, H) / 34));
  ctx.font = `600 ${fontSize}px 'Saira Condensed', sans-serif`;
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

    // equerres d'angle : lecture instantanee meme sur fond charge
    const arm = Math.min(w, h) * 0.22;
    ctx.lineWidth = unit * 3;
    ctx.beginPath();
    ctx.moveTo(x, y + arm); ctx.lineTo(x, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x + w - arm, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - arm);
    ctx.stroke();

    if (!els.optLabels.checked) continue;

    const name = CLASSES[d.cls] ? CLASSES[d.cls].name : `classe ${d.cls}`;
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

/* ---------------- boucle ---------------- */

async function infer() {
  if (state.busy || !state.session) return;
  state.busy = true;
  setLed(els.ledRun, 'busy');
  const t0 = performance.now();
  try {
    const { detections, dims, nc } = await Detector.run(
      state.session, preCtx, els.video,
      state.inputW, state.inputH,
      els.video.videoWidth, els.video.videoHeight,
      { conf: parseFloat(els.conf.value), iou: parseFloat(els.iou.value) },
    );
    state.nc = nc;
    state.detections = detections;
    updateClassCounts(detections);

    if (els.mOutput.textContent === '—') els.mOutput.textContent = `[${dims.join(', ')}]`;

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
    say(`Erreur d'inférence : ${err.message}`, 'err');
    state.running = false;
  } finally {
    state.busy = false;
  }
}

function onFrame() {
  if (!els.video.paused && !els.video.ended) {
    state.frame++;
    if (state.frame % parseInt(els.stride.value, 10) === 0) infer();
  }
  draw();
  schedule();
}

function schedule() {
  if (!state.running) return;
  if (els.video.requestVideoFrameCallback) els.video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

/* ---------------- actions ---------------- */

els.btnPlay.onclick = () => {
  if (els.video.paused) els.video.play().catch((e) => say(e.message, 'err'));
  else els.video.pause();
};

// la boucle suit l'etat reel de la video : le bouton et les controles natifs
// (lecture, pause, deplacement dans la timeline) la pilotent indifferemment
els.video.addEventListener('play', () => {
  state.running = true;
  els.btnPlay.textContent = '⏸ Pause';
  schedule();
});
['pause', 'ended'].forEach((ev) => els.video.addEventListener(ev, () => {
  state.running = false;
  els.btnPlay.textContent = '▶ Lancer';
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

$('btn-model').onclick = () => $('file-model').click();
$('btn-video').onclick = () => $('file-video').click();

$('file-model').onchange = (e) => {
  const f = e.target.files[0];
  if (f) f.arrayBuffer().then((buf) => loadModel(new Uint8Array(buf), f.name));
};
$('file-video').onchange = (e) => {
  const f = e.target.files[0];
  if (f) loadVideo(URL.createObjectURL(f), f.name);
};

bindDrop($('drop-model'), (f) => f.name.endsWith('.onnx'), (f) =>
  f.arrayBuffer().then((buf) => loadModel(new Uint8Array(buf), f.name)));
bindDrop($('drop-video'), (f) => f.type.startsWith('video/'), (f) =>
  loadVideo(URL.createObjectURL(f), f.name));
bindDrop(els.stage, (f) => f.type.startsWith('video/') || f.name.endsWith('.onnx'), (f) => {
  if (f.name.endsWith('.onnx')) f.arrayBuffer().then((b) => loadModel(new Uint8Array(b), f.name));
  else loadVideo(URL.createObjectURL(f), f.name);
});

/* ---------------- demarrage ---------------- */

async function boot() {
  buildClassList();
  bindSliders();
  els.stage.classList.add('empty');

  // modele depose dans web/models/ : on le prend automatiquement
  try {
    const head = await fetch(DEFAULT_MODEL, { method: 'HEAD' });
    if (head.ok) await loadModel(DEFAULT_MODEL, DEFAULT_MODEL.split('/').pop());
    else say('Aucun modèle dans models/. Déposez votre best.onnx dans le panneau 01.');
  } catch {
    say('Aucun modèle dans models/. Déposez votre best.onnx dans le panneau 01.');
  }

  // videos d'exemple si elles ont ete copiees dans web/demo/
  const found = [];
  for (const src of SAMPLE_VIDEOS) {
    try {
      const r = await fetch(src, { method: 'HEAD' });
      if (r.ok) found.push(src);
    } catch { /* absente, on ignore */ }
  }
  if (found.length) {
    els.samples.hidden = false;
    found.forEach((src) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.textContent = src.split('/').pop();
      b.onclick = () => loadVideo(src, b.textContent);
      els.samplesList.appendChild(b);
    });
  }
}

boot();
