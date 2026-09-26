/**
 * Application shell: camera, capture loop, live tracking, state machine, library, settings.
 *
 * One page, five states (ready / scanning / processing / result / error). The browser owns the
 * webcam and the WebGL viewer; the Python backend owns frames, COLMAP and the filesystem.
 *
 * Anything a person typed (the find record) or a device reported (the camera label) is put on
 * the page with textContent only, never innerHTML.
 */

import { PointCloudViewer } from './viewer.js';

const $ = (id) => document.getElementById(id);

const el = {
  pillCamera: $('pill-camera'),
  pillEngine: $('pill-engine'),
  btnSettings: $('btn-settings'),
  btnLibrary: $('btn-library'),
  shell: $('shell'),

  views: Object.fromEntries(
    [...document.querySelectorAll('[data-view]')].map((node) => [node.dataset.view, node]),
  ),
  rowReady: document.querySelector('[data-when="ready"]'),
  rowScanning: document.querySelector('[data-when="scanning"]'),

  viewport: $('viewport'),
  preview: $('preview'),
  grabber: $('grabber'),
  featureLayer: $('feature-layer'),
  cameraEmpty: $('camera-empty'),
  cameraEmptyTitle: $('camera-empty-title'),
  cameraEmptyBody: $('camera-empty-body'),
  btnRetryCamera: $('btn-retry-camera'),
  scanOverlay: $('scan-overlay'),
  viewportBadge: $('viewport-badge'),
  scanTimer: $('scan-timer'),
  hud: $('hud'),
  hudState: $('hud-state'),
  hudFeatures: $('hud-features'),
  hudLinks: $('hud-links'),
  hudBoard: $('hud-board'),
  meter: $('meter'),
  filmstrip: $('filmstrip'),
  live: $('live'),
  liveMap: $('live-map'),
  liveAge: $('live-age'),
  liveCanvas: $('live-canvas'),
  liveEmpty: $('live-empty'),
  livePlaced: $('live-placed'),
  liveCoverage: $('live-coverage'),
  coverageAge: $('coverage-age'),
  compass: $('compass'),
  coverageNote: $('coverage-note'),

  cameraSelect: $('camera-select'),
  presetToggle: $('preset-toggle'),
  btnRecordPending: $('btn-record-pending'),
  pendingRecordLabel: $('pending-record-label'),
  btnStart: $('btn-start'),
  btnFinish: $('btn-finish'),
  scanElapsed: $('scan-elapsed'),
  scanFrames: $('scan-frames'),
  metricPlaced: $('metric-placed'),
  scanPlaced: $('scan-placed'),
  scanScore: $('scan-score'),
  scanGuidance: $('scan-guidance'),

  mosaic: $('mosaic'),
  stageList: $('stage-list'),
  progressFill: $('progress-fill'),
  progressPct: $('progress-pct'),
  progressDetail: $('progress-detail'),
  processingSub: $('processing-sub'),
  techLog: $('tech-log'),
  btnCancel: $('btn-cancel'),

  canvasWrap: $('canvas-wrap'),
  viewerCanvas: $('viewer-canvas'),
  viewerHint: $('viewer-hint'),
  viewerLoading: $('viewer-loading'),
  resultName: $('result-name'),
  resultCheck: $('result-check'),
  resultVerdict: $('result-verdict'),
  resultWarning: $('result-warning'),
  pointSize: $('point-size'),
  showCameras: $('show-cameras'),
  autoRotate: $('auto-rotate'),
  statPoints: $('stat-points'),
  statImages: $('stat-images'),
  statTime: $('stat-time'),
  timeline: $('timeline'),
  scaleBadge: $('scale-badge'),
  scaleBar: $('scale-bar'),
  scaleBarLine: $('scale-bar-line'),
  scaleBarLabel: $('scale-bar-label'),
  measureReadout: $('measure-readout'),
  btnMeasure: $('btn-measure'),
  btnRecord: $('btn-record'),
  btnReport: $('btn-report'),
  btnResetView: $('btn-reset-view'),
  btnExport: $('btn-export'),
  btnOpenFolder: $('btn-open-folder'),
  btnNewScan: $('btn-new-scan'),

  compareSync: $('compare-sync'),
  btnCompareReset: $('btn-compare-reset'),
  btnCompareClose: $('btn-compare-close'),
  compareCanvas: [$('compare-canvas-a'), $('compare-canvas-b')],
  compareLoading: [$('compare-loading-a'), $('compare-loading-b')],
  compareLabel: [$('compare-label-a'), $('compare-label-b')],
  compareHead: [$('compare-head-a'), $('compare-head-b')],
  compareTable: $('compare-table'),

  errorTitle: $('error-title'),
  errorMessage: $('error-message'),
  errorHints: $('error-hints'),
  errorTech: $('error-tech'),
  errorLog: $('error-log'),
  btnErrorRetry: $('btn-error-retry'),
  btnErrorSettings: $('btn-error-settings'),

  library: $('library'),
  libraryList: $('library-list'),
  libraryEmpty: $('library-empty'),
  libraryEmptyText: $('library-empty-text'),
  librarySearch: $('library-search'),
  libraryCount: $('library-count'),
  libraryPick: $('library-pick'),
  btnCompare: $('btn-compare'),

  settingsModal: $('settings-modal'),
  btnCloseSettings: $('btn-close-settings'),
  engineStatus: $('engine-status'),
  engineNote: $('engine-note'),
  engineSearched: $('engine-searched'),
  colmapPath: $('colmap-path'),
  btnSaveColmap: $('btn-save-colmap'),
  btnDetectColmap: $('btn-detect-colmap'),
  importFolder: $('import-folder'),
  btnImport: $('btn-import'),
  importNote: $('import-note'),
  markerSize: $('marker-size'),
  btnMarkerSize: $('btn-marker-size'),
  btnRemeasure: $('btn-remeasure'),
  markerNote: $('marker-note'),

  recordModal: $('record-modal'),
  recordForm: $('record-form'),
  recordContext: $('record-context'),
  recordMaterial: $('record-material'),
  recordOther: $('record-other'),
  recordError: $('record-error'),
  btnCloseRecord: $('btn-close-record'),
  btnRecordClear: $('btn-record-clear'),
  aboutVersions: $('about-versions'),

  toast: $('toast'),
};

const state = {
  system: null,
  preset: 'fast',
  stream: null,
  deviceId: null,
  scanId: null,
  scanStartedAt: 0,
  acceptedFrames: 0,
  capturing: false,
  captureTimer: null,
  tickTimer: null,
  pollTimer: null,
  inFlight: false,
  viewer: null,
  recent: [], // last few frame outcomes, for debounced guidance
  shownTracking: 'starting',
  frameUrls: [], // object URLs of accepted frames (filmstrip + processing mosaic)
  scans: [],
  query: '', // library search
  pendingRecord: null, // find record for the next scan, typed before pressing Start
  result: null, // the scan shown in the result view (its status payload)
};

/** Which record the form edits: `{ kind: 'pending' }` or `{ kind: 'scan', id }`. */
const recordForm = { target: null };

/** Live preview while scanning: the latest server round and the small inset viewer. */
const preview = { timer: null, inFlight: false, data: null, round: null, viewer: null, clockOffset: 0 };

/** Side-by-side comparison: picking state in the library, then two viewers. */
const compare = { picking: false, picked: [], scans: [], viewers: [], returnTo: null };

// ── plumbing ────────────────────────────────────────────────────────────────

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!response.ok) {
    const detail = body?.detail;
    const error = new Error(
      (typeof detail === 'object' && detail?.message) ||
        (typeof detail === 'string' && detail) ||
        `Request failed (${response.status})`,
    );
    error.field = typeof detail === 'object' ? detail?.field : undefined;
    throw error;
  }
  return body;
}

function putJSON(path, payload) {
  return api(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });
}

function postJSON(path, payload) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });
}

let toastTimer = null;
function toast(message) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  requestAnimationFrame(() => el.toast.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.toast.classList.remove('show');
    setTimeout(() => (el.toast.hidden = true), 250);
  }, 3200);
}

function setView(name) {
  // Every way out of compare mode passes through here, so the two WebGL contexts are always
  // given back rather than piling up against the browser's context limit.
  if (name !== 'compare') closeCompareViewers();
  for (const [key, node] of Object.entries(el.views)) node.hidden = key !== name;
  document.documentElement.dataset.state = name;
  document.documentElement.dataset.busy = String(name === 'processing' || state.capturing);
}

const pad = (n) => String(n).padStart(2, '0');
const clock = (seconds) => `${pad(Math.floor(seconds / 60))}:${pad(Math.floor(seconds % 60))}`;

const presetLabel = (name) =>
  state.system?.presets.find((preset) => preset.name === name)?.label || name;

const scanDate = (scan) =>
  new Date((scan.createdAt || 0) * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

/** The find number when there is one: that is how a find is known, not by its scan date. */
const scanTitle = (scan) => scan?.record?.findNumber || `Scan · ${scanDate(scan)}`;

/** Two significant figures, so a small uncertainty never rounds to a claim of "0.00". */
const sig2 = (value) => Number(value.toPrecision(2)).toString();

function formatMm(mm) {
  if (mm >= 100) return mm.toFixed(0);
  if (mm >= 10) return mm.toFixed(1);
  return mm.toFixed(2);
}

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private window or blocked storage: the preference just isn't remembered */
  }
}

// ── system / settings ───────────────────────────────────────────────────────

async function loadSystem() {
  state.system = await api('/api/system');
  state.preset = storageGet('preset') || state.system.defaultPreset;
  if (!state.system.presets.some((preset) => preset.name === state.preset)) {
    state.preset = state.system.defaultPreset;
  }
  renderPresets();
  renderEngine();
  setupRecordForm(state.system.record);
  el.markerSize.value = storageGet('markerSizeMm') || '';
  el.aboutVersions.textContent =
    `App ${state.system.version} · Python ${state.system.python}` +
    (state.system.colmap.version ? ` · COLMAP ${state.system.colmap.version}` : '');
}

function renderPresets() {
  el.presetToggle.innerHTML = '';
  for (const preset of state.system.presets) {
    const button = document.createElement('button');
    button.type = 'button';
    button.role = 'radio';
    button.textContent = preset.label;
    button.title = preset.description;
    button.setAttribute('aria-checked', String(preset.name === state.preset));
    button.addEventListener('click', () => {
      state.preset = preset.name;
      storageSet('preset', preset.name);
      renderPresets();
    });
    el.presetToggle.appendChild(button);
  }
}

function renderEngine() {
  const info = state.system.colmap;
  el.pillEngine.dataset.ok = info.available ? 'true' : 'error';

  if (info.available) {
    el.engineStatus.innerHTML = `<span class="ok">Detected</span> · version ${info.version} · GPU: ${info.gpu}`;
    el.engineNote.textContent = `Using ${info.path} (found via ${info.source}).`;
    el.colmapPath.value = info.path || '';
  } else {
    const blocked = info.error?.startsWith('COLMAP was found');
    el.engineStatus.innerHTML = blocked
      ? '<span class="bad">Found, but it cannot start</span>'
      : '<span class="bad">Not found</span>';
    el.engineNote.textContent = blocked
      ? info.error
      : 'COLMAP must be installed to build a map. Install it, then enter the full path to ' +
        'COLMAP.bat (Windows) or the colmap binary below and press Use.';
    el.engineSearched.textContent = (info.searched || []).join('\n');
  }
}

function openSettings() {
  el.settingsModal.hidden = false;
}

function closeSettings() {
  el.settingsModal.hidden = true;
}

function setLibraryOpen(open) {
  el.shell.dataset.library = open ? 'open' : 'closed';
  el.btnLibrary.setAttribute('aria-pressed', String(open));
  storageSet('library', open ? 'open' : 'closed');
  // The viewer canvases change width with the sidebar.
  requestAnimationFrame(() => {
    state.viewer?.resize();
    compare.viewers.forEach((viewer) => viewer.resize());
  });
}

// ── camera ──────────────────────────────────────────────────────────────────

function stopStream() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
}

async function startCamera(deviceId) {
  stopStream();
  const constraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
  };

  state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  el.preview.srcObject = state.stream;
  await el.preview.play().catch(() => {});

  const track = state.stream.getVideoTracks()[0];
  state.deviceId = track.getSettings().deviceId || deviceId || null;
  el.cameraEmpty.hidden = true;
  el.pillCamera.dataset.ok = 'true';
  el.pillCamera.querySelector('.pill-label').textContent = track.label || 'Camera';
  el.btnStart.disabled = false;
  return track;
}

async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === 'videoinput');
  el.cameraSelect.innerHTML = '';
  cameras.forEach((device, index) => {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    option.selected = device.deviceId === state.deviceId;
    el.cameraSelect.appendChild(option);
  });
  el.cameraSelect.disabled = cameras.length < 2;
  return cameras;
}

function cameraFailure(error) {
  el.pillCamera.dataset.ok = 'error';
  el.btnStart.disabled = true;
  el.cameraEmpty.hidden = false;

  const name = error?.name || '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    el.cameraEmptyTitle.textContent = 'Camera access blocked';
    el.cameraEmptyBody.textContent =
      'Allow camera access for this page — click the camera icon in the address bar, choose ' +
      'Allow, then press Retry.';
  } else if (name === 'NotFoundError' || name === 'OverconstrainedError' || name === 'DevicesNotFoundError') {
    el.cameraEmptyTitle.textContent = 'No camera detected';
    el.cameraEmptyBody.textContent = 'Connect a webcam and try again.';
  } else if (name === 'NotReadableError' || name === 'TrackStartError') {
    el.cameraEmptyTitle.textContent = 'Camera is in use';
    el.cameraEmptyBody.textContent =
      'Another application is using the webcam. Close it, then press Retry.';
  } else {
    el.cameraEmptyTitle.textContent = 'Could not start the camera';
    el.cameraEmptyBody.textContent = error?.message || String(error);
  }
}

async function initCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraFailure({ name: 'NotFoundError' });
    return;
  }
  try {
    await startCamera(state.deviceId);
    await listCameras();
  } catch (error) {
    cameraFailure(error);
  }
}

// ── live tracking feedback ──────────────────────────────────────────────────

const TRACKING = {
  starting: { label: 'Locking on…', tone: 'neutral', tip: 'Hold steady for a moment, then start moving around the subject.' },
  good: { label: 'Tracking', tone: 'good', tip: 'Looking good. Keep circling slowly with the subject in view.' },
  low_texture: { label: 'Low detail', tone: 'warn', tip: 'Not much detail to lock onto. Aim at something textured and well lit.' },
  rotating: { label: 'No depth', tone: 'warn', tip: 'You are turning in place. Step sideways around the subject to add depth.' },
  lost: { label: 'Lost track', tone: 'bad', tip: 'The views stopped overlapping. Go back a little and move slower.' },
};

const REJECTIONS = {
  blurry: { tone: 'warn', tip: 'Frames are blurry. Slow down, or add light.' },
  dark: { tone: 'warn', tip: 'Too dark to see detail. Add light.' },
  bright: { tone: 'warn', tip: 'Overexposed. Avoid pointing at bright lights or windows.' },
  duplicate: { tone: 'neutral', tip: 'The view has barely changed. Keep moving.' },
};

const TONE_RGB = { good: '59,232,176', warn: '244,191,95', bad: '255,107,107', neutral: '155,165,176' };
const METER_SEGMENTS = 14;

const overlay = { points: [], tone: 'neutral', born: 0, raf: null };

function buildMeter() {
  el.meter.innerHTML = '';
  for (let i = 0; i < METER_SEGMENTS; i++) el.meter.appendChild(document.createElement('i'));
}

function renderMeter(score) {
  const lit = Math.round((score / 100) * METER_SEGMENTS);
  [...el.meter.children].forEach((segment, i) => segment.classList.toggle('on', i < lit));
}

/**
 * Turn the last few frame outcomes into one stable message.
 *
 * Verdicts arrive 2-3 times a second; showing each one raw makes the guidance flicker. A state
 * is only displayed once it wins two of the last three frames.
 */
function applyTracking(result) {
  const tracking = result.tracking;
  state.recent.push({ reason: result.reason, state: tracking?.state || 'starting' });
  if (state.recent.length > 3) state.recent.shift();

  const majority = (key) => {
    const counts = {};
    for (const entry of state.recent) counts[entry[key]] = (counts[entry[key]] || 0) + 1;
    return Object.entries(counts).find(([, count]) => count >= 2)?.[0];
  };

  const trackingState = majority('state');
  if (trackingState) state.shownTracking = trackingState;
  const shown = TRACKING[state.shownTracking] || TRACKING.starting;

  // COLMAP's own placed count, once there is enough of it to trust, outranks the ORB guesses.
  const rejection = REJECTIONS[majority('reason')];
  const tip = placementTip() || rejection || shown;
  el.scanGuidance.textContent = tip.tip;
  el.scanGuidance.dataset.tone = tip.tone;

  el.hud.dataset.tone = shown.tone;
  el.hudState.textContent = shown.label;
  if (typeof result.board === 'number') {
    el.hudBoard.dataset.on = String(result.board > 0);
    el.hudBoard.textContent = result.board
      ? `board · ${result.board} marker${result.board === 1 ? '' : 's'}`
      : 'no board';
  }
  if (tracking) {
    el.hudFeatures.textContent = tracking.features;
    el.hudLinks.textContent = tracking.inliers;
    el.scanScore.textContent = tracking.score;
    renderMeter(tracking.score);
    overlay.points = tracking.keypoints || [];
    overlay.tone = shown.tone;
    overlay.born = performance.now();
  }
}

/** Where the video actually sits inside the viewport (object-fit: contain letterboxing). */
function videoRect(width, height) {
  const vw = el.preview.videoWidth || 16;
  const vh = el.preview.videoHeight || 9;
  const scale = Math.min(width / vw, height / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}

function drawFeatures() {
  const canvas = el.featureLayer;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const age = performance.now() - overlay.born;
  const alpha = Math.max(0, 1 - age / 1100);
  if (alpha > 0 && overlay.points.length) {
    const rect = videoRect(width, height);
    const rgb = TONE_RGB[overlay.tone] || TONE_RGB.neutral;
    const grow = 1 + Math.min(age / 1100, 1) * 0.6;
    for (const [nx, ny] of overlay.points) {
      const x = rect.x + nx * rect.w;
      const y = rect.y + ny * rect.h;
      ctx.fillStyle = `rgba(${rgb},${0.18 * alpha})`;
      ctx.beginPath();
      ctx.arc(x, y, 4.5 * grow, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(${rgb},${0.95 * alpha})`;
      ctx.fillRect(x - 1.2, y - 1.2, 2.4, 2.4);
    }
  }
  overlay.raf = requestAnimationFrame(drawFeatures);
}

function startOverlay() {
  cancelAnimationFrame(overlay.raf);
  overlay.points = [];
  overlay.raf = requestAnimationFrame(drawFeatures);
}

function stopOverlay() {
  cancelAnimationFrame(overlay.raf);
  overlay.raf = null;
  overlay.points = [];
  const ctx = el.featureLayer.getContext('2d');
  ctx.clearRect(0, 0, el.featureLayer.width, el.featureLayer.height);
}

/** Keep an evenly spread sample of accepted frames for the filmstrip and processing mosaic. */
function rememberFrame(blob) {
  const url = URL.createObjectURL(blob);
  state.frameUrls.push(url);
  if (state.frameUrls.length > 40) {
    // Drop every other frame: the kept set stays spread across the whole scan.
    const kept = [];
    state.frameUrls.forEach((frameUrl, i) => {
      if (i % 2 === 0 || i === state.frameUrls.length - 1) kept.push(frameUrl);
      else URL.revokeObjectURL(frameUrl);
    });
    state.frameUrls = kept;
  }

  const img = document.createElement('img');
  img.src = url;
  img.alt = '';
  el.filmstrip.appendChild(img);
  while (el.filmstrip.children.length > 10) el.filmstrip.firstElementChild.remove();
}

function forgetFrames() {
  state.frameUrls.forEach((url) => URL.revokeObjectURL(url));
  state.frameUrls = [];
  el.filmstrip.innerHTML = '';
  el.mosaic.innerHTML = '';
}

function fillMosaic(urls) {
  el.mosaic.innerHTML = '';
  if (!urls.length) return;
  for (let i = 0; i < 48; i++) {
    const img = document.createElement('img');
    img.src = urls[i % urls.length];
    img.alt = '';
    img.style.animationDelay = `${(i * 0.37) % 6}s`;
    el.mosaic.appendChild(img);
  }
}

// ── live 3D preview + coverage ──────────────────────────────────────────────

const PREVIEW_POLL_MS = 2000;

const COVERAGE_REASONS = {
  too_few_cameras: 'Waiting for more placed views',
  no_common_subject:
    'No single subject in the middle of the views. The guide works when you walk around an object, not when panning across a room.',
};

function startPreview(scanId) {
  stopPreview();
  el.live.hidden = false;
  renderPreview(null);
  preview.timer = setInterval(() => pollPreview(scanId), PREVIEW_POLL_MS);
}

/**
 * A disposed viewer has forced its WebGL context lost, and a canvas hands that dead context to
 * anything that asks again - so the next viewer needs a new canvas element.
 */
function freshCanvas(canvas) {
  const replacement = canvas.cloneNode(false);
  canvas.replaceWith(replacement);
  return replacement;
}

function stopPreview() {
  clearInterval(preview.timer);
  preview.timer = null;
  if (preview.viewer) {
    preview.viewer.dispose();
    el.liveCanvas = freshCanvas(el.liveCanvas);
  }
  preview.viewer = null;
  preview.data = null;
  preview.round = null;
  el.live.hidden = true;
  el.scanPlaced.textContent = '–';
  el.metricPlaced.dataset.tone = 'neutral';
}

async function pollPreview(scanId) {
  if (preview.inFlight) return;
  preview.inFlight = true;
  try {
    const data = await api(`/api/scans/${scanId}/preview`);
    if (scanId !== state.scanId || !state.capturing || data.state === 'off') return;
    preview.data = data;
    preview.clockOffset = Date.now() / 1000 - data.serverTime;
    renderPreview(data);
    if (data.round && data.round !== preview.round) await loadPreviewRound(data);
  } catch (error) {
    console.warn('preview poll failed', error);
  } finally {
    preview.inFlight = false;
  }
}

async function loadPreviewRound(data) {
  preview.round = data.round;
  if (!preview.viewer) {
    preview.viewer = new PointCloudViewer(el.liveCanvas, { gizmo: false, intro: false });
    preview.viewer.start();
  }
  const viewer = preview.viewer;
  let cameras = [];
  try {
    cameras = (await api(data.camerasUrl)).cameras || [];
  } catch {
    /* orientation only; the cloud still loads */
  }
  try {
    await viewer.load(data.plyUrl, cameras);
    if (viewer !== preview.viewer) return; // capture ended while this was loading
    viewer.setPointSize(2.4);
    viewer.resize();
  } catch (error) {
    // A newer round can prune this one before it is fetched; the next poll catches up.
    console.warn('preview cloud not loaded', error);
  }
}

/** Placed-of-total as a tone. Early rounds (few frames) are too noisy to raise an alarm. */
function placementTone(data) {
  if (!data?.total) return 'neutral';
  const ratio = data.placed / data.total;
  if (data.total < data.trustedFrames) return ratio >= 0.6 ? 'good' : 'neutral';
  if (ratio >= 0.85) return 'good';
  if (ratio >= 0.6) return 'warn';
  return 'bad';
}

function placementTip() {
  const data = preview.data;
  const tone = placementTone(data);
  if (tone === 'bad') {
    return {
      tone,
      tip:
        `Only ${data.placed} of ${data.total} frames fit the 3D map. Go back to where the map ` +
        'was growing, then move slower and sideways.',
    };
  }
  if (tone === 'warn') {
    return {
      tone,
      tip: `${data.total - data.placed} of ${data.total} frames do not fit the 3D map. Slow down and keep more overlap.`,
    };
  }
  return null;
}

function renderPreview(data) {
  const map = el.liveMap;
  map.dataset.state = data?.state || 'collecting';
  el.liveEmpty.hidden = data?.state === 'ready';

  if (!data || data.state === 'collecting') {
    const have = data?.frames ?? 0;
    const need = data?.minFrames ?? 12;
    el.liveEmpty.textContent = `Collecting frames… ${Math.min(have, need)} of ${need}`;
    el.livePlaced.textContent = 'First preview after a few seconds';
  } else if (data.state === 'waiting') {
    // A failed early round is normal (no good initial pair yet): never shown as an error.
    const first = data.running && data.rounds <= 1;
    el.liveEmpty.textContent = first ? 'Building first preview…' : 'Waiting for enough views';
    el.livePlaced.textContent = first
      ? 'This takes a few seconds'
      : data.running
        ? 'Trying again with more frames…'
        : 'Keep moving around the subject';
  }

  const tone = data?.state === 'ready' ? placementTone(data) : 'neutral';
  map.dataset.tone = tone;
  el.metricPlaced.dataset.tone = tone;
  if (data?.state === 'ready') {
    const early = data.total < data.trustedFrames ? ' · early estimate' : '';
    el.livePlaced.innerHTML = `<b>${data.placed} of ${data.total}</b> frames placed${early}`;
    el.scanPlaced.textContent = `${data.placed}/${data.total}`;
  }
  renderCoverage(data);
  renderPreviewAge();
}

/** "12 s ago" from the server's own clock, so client/server clock skew cannot mislead. */
function renderPreviewAge() {
  const data = preview.data;
  for (const node of [el.liveAge, el.coverageAge]) {
    node.dataset.running = String(Boolean(data?.running));
    if (!data?.updatedAt) {
      node.textContent = data?.running ? 'working' : '';
      continue;
    }
    const age = Math.max(0, Date.now() / 1000 - preview.clockOffset - data.updatedAt);
    node.textContent = `${Math.round(age)} s ago`;
  }
}

function renderCoverage(data) {
  const card = el.liveCoverage;
  const coverage = data?.coverage;
  if (!data || data.state !== 'ready') {
    card.dataset.state = 'waiting';
    el.coverageNote.textContent = 'Waiting for first preview';
    return;
  }
  if (!coverage) {
    card.dataset.state = 'waiting';
    el.coverageNote.textContent = COVERAGE_REASONS[data.coverageReason] || 'Waiting for more placed views';
    return;
  }
  card.dataset.state = 'ready';
  drawCompass(coverage);
  el.coverageNote.innerHTML = coverage.complete
    ? 'Full circle covered.'
    : `Walk <b>${coverage.turn}</b> around the subject · ${coverage.covered} of ${coverage.sectorCount} sides covered`;
}

/**
 * Top-down compass, rotated so you stand at the bottom looking at the subject in the middle.
 * Azimuth grows towards your right, so screen-right is your right-hand side.
 */
function drawCompass(coverage) {
  const width = 360 / coverage.sectorCount;
  const inner = 24;
  const outer = 44;
  const point = (angle, radius) => {
    const rad = (angle * Math.PI) / 180;
    return [radius * Math.sin(rad), radius * Math.cos(rad)];
  };
  const rel = (azimuth) => azimuth - coverage.current;
  const f = (n) => n.toFixed(2);

  const parts = [`<circle class="ring" r="${outer + 8}"></circle>`];
  coverage.sectors.forEach((count, index) => {
    const from = rel(index * width - width / 2) + 1.2;
    const to = rel(index * width + width / 2) - 1.2;
    const [x1, y1] = point(from, outer);
    const [x2, y2] = point(to, outer);
    const [x3, y3] = point(to, inner);
    const [x4, y4] = point(from, inner);
    // Angles grow counter-clockwise on screen here, so the outer arc sweeps with flag 0.
    const path =
      `M${f(x1)} ${f(y1)} A${outer} ${outer} 0 0 0 ${f(x2)} ${f(y2)} ` +
      `L${f(x3)} ${f(y3)} A${inner} ${inner} 0 0 1 ${f(x4)} ${f(y4)}Z`;
    const cls = ['sector', count ? 'on' : '', index === coverage.target ? 'target' : ''].join(' ');
    parts.push(`<path class="${cls}" d="${path}"><title>${count} views</title></path>`);
  });

  if (coverage.target !== null) {
    // An arc from where you stand to the next open side, in the direction to walk.
    const sign = coverage.turn === 'right' ? 1 : -1;
    let span = (((coverage.target * width - coverage.current) * sign) % 360 + 360) % 360;
    span = Math.max(span, 24);
    const end = sign * span;
    const radius = outer + 8;
    const [sx, sy] = point(0, radius);
    const [ex, ey] = point(end, radius);
    const large = span > 180 ? 1 : 0;
    parts.push(
      `<path class="arrow" d="M${f(sx)} ${f(sy)} A${radius} ${radius} 0 ${large} ${sign > 0 ? 0 : 1} ${f(ex)} ${f(ey)}"></path>`,
    );
    const rad = (end * Math.PI) / 180;
    const tx = sign * Math.cos(rad);
    const ty = -sign * Math.sin(rad);
    const tip = [ex + tx * 5, ey + ty * 5];
    const left = [ex - tx * 2 - ty * 4, ey - ty * 2 + tx * 4];
    const right = [ex - tx * 2 + ty * 4, ey - ty * 2 - tx * 4];
    parts.push(
      `<path class="arrow-head" d="M${f(tip[0])} ${f(tip[1])} L${f(left[0])} ${f(left[1])} L${f(right[0])} ${f(right[1])}Z"></path>`,
    );
  }

  parts.push('<circle class="subject" r="4"></circle>');
  const [yx, yy] = point(0, outer + 8);
  parts.push(`<circle class="you" cx="${f(yx)}" cy="${f(yy)}" r="4.5"><title>you</title></circle>`);
  el.compass.innerHTML = parts.join('');
}

// ── capture ─────────────────────────────────────────────────────────────────

function grabFrame() {
  const video = el.preview;
  if (!video.videoWidth) return null;

  const maxWidth = state.system.capture.width;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = el.grabber;
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) =>
    canvas.toBlob(resolve, 'image/jpeg', state.system.capture.jpegQuality),
  );
}

async function sendFrame() {
  // Skip this tick if the previous upload is still going: capture cadence should never
  // queue up behind a slow quality check.
  if (!state.capturing || state.inFlight) return;
  state.inFlight = true;
  try {
    const blob = await grabFrame();
    if (!blob) return;

    const form = new FormData();
    form.append('frame', blob, 'frame.jpg');
    const result = await api(`/api/scans/${state.scanId}/frames`, { method: 'POST', body: form });
    if (!state.capturing) return;

    state.acceptedFrames = result.acceptedFrames;
    el.scanFrames.textContent = state.acceptedFrames;
    if (result.accepted) rememberFrame(blob);
    applyTracking(result);

    if (result.limitReached) {
      toast('Frame limit reached — finishing the scan.');
      state.inFlight = false; // this upload is done; finishScan waits for in-flight ones
      await finishScan();
    }
  } catch (error) {
    console.warn('frame upload failed', error);
  } finally {
    state.inFlight = false;
  }
}

async function startScan() {
  try {
    const track = state.stream?.getVideoTracks()[0];
    const scan = await postJSON('/api/scans', {
      preset: state.preset,
      record: state.pendingRecord,
      camera: track?.label || '',
      markerSizeMm: markerSizeSetting(),
    });
    // The record now belongs to this scan; the next find gets a fresh one.
    state.pendingRecord = null;
    renderPendingRecord();
    forgetFrames();
    state.scanId = scan.id;
    state.scanStartedAt = Date.now();
    state.acceptedFrames = 0;
    state.capturing = true;
    state.recent = [];
    state.shownTracking = 'starting';

    el.scanFrames.textContent = '0';
    el.scanScore.textContent = '–';
    el.scanElapsed.textContent = '00:00';
    el.scanTimer.textContent = '00:00';
    el.scanGuidance.textContent = TRACKING.starting.tip;
    el.scanGuidance.dataset.tone = 'neutral';
    el.hud.dataset.tone = 'neutral';
    el.hudState.textContent = TRACKING.starting.label;
    el.hudFeatures.textContent = '0';
    el.hudLinks.textContent = '0';
    el.hudBoard.dataset.on = 'false';
    el.hudBoard.textContent = 'no board';
    renderMeter(0);
    el.rowReady.hidden = true;
    el.rowScanning.hidden = false;
    el.scanOverlay.hidden = false;
    el.viewportBadge.hidden = false;
    el.hud.hidden = false;
    el.cameraSelect.disabled = true;
    document.documentElement.dataset.scanning = 'true';
    setView('capture');
    startOverlay();
    startPreview(scan.id);

    state.captureTimer = setInterval(sendFrame, state.system.capture.intervalMs);
    state.tickTimer = setInterval(() => {
      const seconds = (Date.now() - state.scanStartedAt) / 1000;
      el.scanElapsed.textContent = clock(seconds);
      el.scanTimer.textContent = clock(seconds);
      renderPreviewAge();
    }, 250);
  } catch (error) {
    showError({ title: 'Could not start the scan', message: error.message, hints: [] });
  }
}

function stopCapture() {
  state.capturing = false;
  clearInterval(state.captureTimer);
  clearInterval(state.tickTimer);
  state.captureTimer = null;
  state.tickTimer = null;
  stopOverlay();
  stopPreview();
  el.rowScanning.hidden = true;
  el.rowReady.hidden = false;
  el.scanOverlay.hidden = true;
  el.viewportBadge.hidden = true;
  el.hud.hidden = true;
  el.filmstrip.innerHTML = '';
  el.cameraSelect.disabled = false;
  document.documentElement.dataset.scanning = 'false';
}

async function finishScan() {
  if (!state.scanId) return;
  const scanId = state.scanId;
  stopCapture();
  // Let a frame that is already uploading land first; otherwise it reaches the server after
  // Finish and is refused with a 409.
  for (let waited = 0; state.inFlight && waited < 3000; waited += 50) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  const seconds = (Date.now() - state.scanStartedAt) / 1000;
  el.processingSub.textContent =
    `${state.acceptedFrames} frames · ${clock(seconds)} of scanning · ` +
    `${presetLabel(state.preset)} quality`;

  showProcessing();
  fillMosaic(state.frameUrls);

  try {
    await postJSON(`/api/scans/${scanId}/finish`);
    startPolling(scanId);
    refreshLibrary();
  } catch (error) {
    showError({ title: 'Could not start reconstruction', message: error.message, hints: [] });
  }
}

// ── processing ──────────────────────────────────────────────────────────────

const STAGE_ORDER = ['preparing', 'extracting', 'matching', 'mapping', 'exporting'];
const PROCESSING = new Set(['queued', ...STAGE_ORDER]);

function showProcessing() {
  setView('processing');
  renderStages(null);
  el.progressFill.style.width = '0%';
  el.progressPct.textContent = '0%';
  el.progressDetail.textContent = '';
  el.techLog.textContent = 'Waiting for output…';
}

function renderStages(status) {
  const labels = {
    preparing: 'Preparing images',
    extracting: 'Detecting features',
    matching: 'Matching views',
    mapping: 'Building 3D map',
    exporting: 'Preparing viewer',
  };
  const currentIndex = status ? STAGE_ORDER.indexOf(status) : -1;

  el.stageList.innerHTML = '';
  STAGE_ORDER.forEach((key, index) => {
    const item = document.createElement('li');
    const marker = document.createElement('span');
    marker.className = 'marker';
    item.appendChild(marker);
    item.appendChild(document.createTextNode(labels[key]));

    let markerState = 'pending';
    if (status === 'complete') markerState = 'done';
    else if (currentIndex >= 0 && index < currentIndex) markerState = 'done';
    else if (currentIndex >= 0 && index === currentIndex) markerState = 'active';
    item.dataset.state = markerState;

    el.stageList.appendChild(item);
  });
}

function startPolling(scanId) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(() => pollStatus(scanId), 700);
  pollStatus(scanId);
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function pollStatus(scanId) {
  let status;
  try {
    status = await api(`/api/scans/${scanId}/status`);
  } catch (error) {
    stopPolling();
    showError({ title: 'Lost contact with the server', message: error.message, hints: [] });
    return;
  }
  if (scanId !== state.scanId) return; // the user moved on while this request was in flight

  renderStages(status.status);
  el.progressFill.style.width = `${status.progress}%`;
  el.progressPct.textContent = `${status.progress}%`;
  el.progressDetail.textContent = status.stageDetail || '';

  if (status.status === 'complete') {
    stopPolling();
    refreshLibrary();
    await showResult(scanId, status);
  } else if (status.status === 'failed') {
    stopPolling();
    refreshLibrary();
    const log = await fetchLog(scanId);
    showError(status.error || { title: 'Reconstruction failed', message: '', hints: [] }, log);
  } else if (status.status === 'cancelled') {
    stopPolling();
    refreshLibrary();
    resetToReady();
  } else if (document.querySelector('.processing .tech[open]')) {
    el.techLog.textContent = (await fetchLog(scanId)) || 'Waiting for output…';
    el.techLog.scrollTop = el.techLog.scrollHeight;
  }
}

async function fetchLog(scanId) {
  try {
    const payload = await api(`/api/scans/${scanId}/log`);
    return payload.log || '';
  } catch {
    return '';
  }
}

// ── result ──────────────────────────────────────────────────────────────────

/** How much of the capture COLMAP could actually use, in words. */
function verdictFor(status) {
  const placed = status.result?.registered_images ?? status.registeredImages ?? 0;
  const input = status.result?.input_images || status.acceptedFrames || placed || 1;
  const ratio = placed / input;
  if (ratio >= 0.85) return { label: 'Excellent', tone: 'good', ratio, placed, input };
  if (ratio >= 0.6) return { label: 'Good', tone: 'good', ratio, placed, input };
  if (ratio >= 0.35) return { label: 'Partial', tone: 'warn', ratio, placed, input };
  return { label: 'Weak', tone: 'bad', ratio, placed, input };
}

function warningFor(verdict, status) {
  if (verdict.ratio >= 0.6) return '';
  const timeline = status.timeline || [];
  const count = (name) => timeline.filter((entry) => entry[1] === name).length;
  let cause = timeline.length
    ? 'The views probably did not overlap enough.'
    : 'Usually the camera turned in place or moved too fast between views.';
  if (count('rotating') >= Math.max(3, count('lost'))) {
    cause = 'The camera mostly turned in place, which gives no depth. Next time, walk sideways around the subject.';
  } else if (count('lost') >= 3) {
    cause = 'Tracking was lost several times. Next time, move slower so each view overlaps the last.';
  } else if (count('low_texture') >= 3) {
    cause = 'The scene had little texture. Try a subject with more visible surface detail.';
  }
  return `Only ${verdict.placed} of ${verdict.input} frames could be placed in 3D. ${cause}`;
}

function renderTimeline(timeline) {
  el.timeline.innerHTML = '';
  for (const [score, trackingState] of timeline || []) {
    const bar = document.createElement('i');
    bar.style.height = `${Math.max(18, score)}%`;
    bar.dataset.state = trackingState;
    el.timeline.appendChild(bar);
  }
}

async function showResult(scanId, status) {
  setView('result');
  renderLibrary();
  state.result = status;
  el.viewerLoading.hidden = false;
  renderResultTitle(status);
  renderScale(status);
  el.statPoints.textContent = (status.points || 0).toLocaleString();
  const verdict = verdictFor(status);
  el.statImages.textContent = `${verdict.placed}/${verdict.input}`;
  el.statTime.textContent = `${Math.round(status.result?.duration_seconds ?? status.elapsedProcessing)}s`;
  el.resultVerdict.hidden = false;
  el.resultVerdict.dataset.tone = verdict.tone;
  el.resultCheck.dataset.tone = verdict.tone;
  el.resultVerdict.textContent = `${verdict.label} · ${presetLabel(status.preset)}`;
  const warning = warningFor(verdict, status);
  el.resultWarning.hidden = !warning;
  el.resultWarning.textContent = warning;
  renderTimeline(status.timeline);

  if (!state.viewer) {
    state.viewer = new PointCloudViewer(el.viewerCanvas);
    state.viewer.onUserInteract = () => (el.autoRotate.checked = false);
    state.viewer.onScaleBar = renderScaleBar;
    state.viewer.onMeasure = renderMeasure;
  }
  const viewer = state.viewer;
  viewer.clear();
  viewer.setScale(status.scale?.status === 'scaled' ? status.scale.mmPerUnit : null);
  setMeasuring(false);
  viewer.start();
  // The canvas only has a real size once the result view is visible.
  requestAnimationFrame(() => viewer.resize());

  try {
    const result = await api(`/api/scans/${scanId}/result`);
    let cameras = [];
    if (result.camerasUrl) {
      try {
        cameras = (await api(result.camerasUrl)).cameras || [];
      } catch {
        /* the camera overlay and upright orientation are optional */
      }
    }
    if (scanId !== state.scanId) return;

    const loaded = await viewer.load(result.plyUrl, cameras);
    el.statPoints.textContent = loaded.points.toLocaleString();
    viewer.setCamerasVisible(el.showCameras.checked);
    viewer.setAutoRotate(el.autoRotate.checked);
    viewer.setPointSize(Number(el.pointSize.value));
    viewer.resize();
    el.viewerLoading.hidden = true;
    el.viewerHint.classList.remove('fade');
    setTimeout(() => el.viewerHint.classList.add('fade'), 5000);
  } catch (error) {
    el.viewerLoading.hidden = true;
    showError({
      title: 'Could not display the point cloud',
      message: error.message,
      hints: ['The .PLY file is still on disk — open the scan folder to find it'],
    });
  }
}

function renderResultTitle(status) {
  el.resultName.textContent = status.record?.findNumber
    ? `${status.record.findNumber} · ${scanDate(status)}`
    : `Scan · ${scanDate(status)}`;
}

// ── scale ───────────────────────────────────────────────────────────────────

const SCALE_REASONS = {
  no_markers: 'no marker board in the frames.',
  insufficient: 'the board was seen, but fewer than 3 markers could be measured.',
  error: 'the scale could not be measured.',
  not_measured: 'this scan was made before scaling existed.',
};

/** The ± shown for a scaled scan: the larger of the between-marker and bootstrap spreads. */
function uncertaintyOf(status) {
  const scale = status?.scale;
  if (scale?.status !== 'scaled') return null;
  return status.scaleUncertaintyPct ?? Math.max(scale.spreadPct || 0, scale.bootstrapPct || 0);
}

/** Top-right badge: the measured scale with its uncertainty, or why there is none. */
function renderScale(status) {
  const scale = status.scale || { status: 'not_measured' };
  const badge = el.scaleBadge;
  badge.replaceChildren();
  badge.hidden = false;
  const scaled = scale.status === 'scaled';
  badge.dataset.scaled = String(scaled);
  const head = document.createElement('b');
  if (scaled) {
    head.textContent = 'Scaled';
    badge.append(
      head,
      ` ± ${sig2(uncertaintyOf(status))}% · ${scale.markersUsed} markers, ${scale.framesUsed} frames` +
        (scale.markerSizeMm !== 30 ? ` · ${scale.markerSizeMm} mm markers` : ''),
    );
    for (const warning of scale.warnings || []) {
      const line = document.createElement('span');
      line.className = 'warn';
      line.textContent = warning;
      badge.append(line);
    }
  } else {
    head.textContent = 'Unscaled';
    const link = document.createElement('a');
    link.href = '/api/marker-board.pdf?paper=a4';
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = 'print the marker board';
    badge.append(head, ': ', link, ` — ${SCALE_REASONS[scale.status] || SCALE_REASONS.error}`);
  }
  el.btnMeasure.hidden = !scaled;
  el.scaleBar.hidden = true;
  el.btnRemeasure.hidden = false;
}

function renderScaleBar(bar) {
  el.scaleBar.hidden = !bar;
  if (!bar) return;
  el.scaleBarLine.style.width = `${bar.px}px`;
  el.scaleBarLabel.textContent = `${bar.mm < 1 ? bar.mm.toFixed(1) : bar.mm} mm at the orbit centre`;
}

function setMeasuring(on) {
  el.btnMeasure.setAttribute('aria-pressed', String(on));
  el.canvasWrap.dataset.measuring = String(on);
  state.viewer?.setMeasuring(on);
  if (on) {
    el.autoRotate.checked = false;
    state.viewer?.setAutoRotate(false);
  }
  el.measureReadout.hidden = !on;
  el.measureReadout.textContent = on ? 'Click the first point' : '';
}

function renderMeasure({ picks, mm, missed }) {
  const box = el.measureReadout;
  box.hidden = false;
  box.replaceChildren();
  if (mm === null) {
    box.textContent = missed
      ? 'No point there — click closer to the cloud'
      : picks === 1 ? 'Click the second point' : 'Click the first point';
    return;
  }
  const pct = uncertaintyOf(state.result) ?? 0;
  const value = document.createElement('b');
  value.textContent = `${formatMm(mm)} mm`;
  box.append(
    value,
    ` ± ${sig2((mm * pct) / 100 || 0)} mm (scale only) · between two reconstructed points · click again to restart`,
  );
}

function resetToReady() {
  stopPolling();
  stopCapture();
  forgetFrames();
  state.scanId = null;
  state.result = null;
  setMeasuring(false);
  el.btnRemeasure.hidden = true;
  state.acceptedFrames = 0;
  if (state.viewer) {
    state.viewer.clear();
    state.viewer.stop();
  }
  el.scanFrames.textContent = '0';
  el.scanElapsed.textContent = '00:00';
  el.progressFill.style.width = '0%';
  el.progressPct.textContent = '0%';
  el.progressDetail.textContent = '';
  el.techLog.textContent = 'Waiting for output…';
  setView('capture');
  renderLibrary();
  if (!state.stream) initCamera();
}

// ── library ─────────────────────────────────────────────────────────────────

async function refreshLibrary() {
  const query = state.query;
  try {
    const payload = await api(`/api/scans?q=${encodeURIComponent(query)}`);
    if (query !== state.query) return; // a newer search is on its way
    state.scans = payload.scans || [];
  } catch {
    return; // the library is a convenience; never let it break the page
  }
  renderLibrary();
}

let searchTimer = null;
function onLibrarySearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = el.librarySearch.value.trim();
    refreshLibrary();
  }, 200);
}

function libraryStatus(scan) {
  if (scan.status === 'complete') return { key: 'complete', label: 'Ready' };
  if (scan.status === 'failed') return { key: 'failed', label: 'Failed' };
  if (scan.status === 'cancelled') return { key: 'cancelled', label: 'Cancelled' };
  if (scan.status === 'capturing') return { key: 'processing', label: 'Capturing' };
  return { key: 'processing', label: 'Processing' };
}

function renderLibrary() {
  const scans = state.scans;
  el.libraryCount.textContent = scans.length;
  el.libraryEmpty.hidden = scans.length > 0;
  el.libraryEmptyText.textContent = state.query
    ? 'No find matches that search. Find number, site, context and material are searched.'
    : 'Finished scans appear here. Click one to reopen it in the viewer.';
  el.libraryList.innerHTML = '';

  for (const scan of scans) {
    const status = libraryStatus(scan);
    const card = document.createElement('article');
    card.className = 'scan-card';
    card.tabIndex = 0;
    card.role = 'button';
    card.setAttribute('aria-current', String(scan.id === state.scanId));
    card.setAttribute('aria-label', `Open ${scanTitle(scan)}`);
    if (compare.picking) {
      const pick = compare.picked.findIndex((picked) => picked.id === scan.id);
      card.dataset.pickable = String(scan.status === 'complete');
      if (pick >= 0) card.dataset.picked = String(pick + 1);
      card.setAttribute('aria-label', `Compare ${scanTitle(scan)}`);
      card.setAttribute('aria-pressed', String(pick >= 0));
    }

    const thumb = document.createElement('div');
    thumb.className = 'scan-thumb';
    if (scan.status !== 'capturing') {
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.alt = '';
      img.src = `/api/scans/${scan.id}/thumbnail.jpg`;
      img.addEventListener('error', () => img.remove());
      thumb.appendChild(img);
    }
    const chip = document.createElement('span');
    chip.className = 'scan-status';
    chip.dataset.status = status.key;
    chip.textContent = status.label;
    thumb.appendChild(chip);
    if (scan.status === 'complete') {
      const points = document.createElement('span');
      points.className = 'scan-points';
      points.textContent = `${(scan.points || 0).toLocaleString()} pts`;
      thumb.appendChild(points);
    }

    const meta = document.createElement('div');
    meta.className = 'scan-meta';
    if (scan.record?.findNumber) {
      const find = document.createElement('span');
      find.className = 'scan-find';
      find.textContent = scan.record.findNumber;
      find.title = [scan.record.siteCode, scan.record.context].filter(Boolean).join(' · ');
      meta.append(find);
    }
    const date = document.createElement('span');
    date.className = 'scan-date';
    date.textContent = [scan.record?.siteCode, scan.record?.context, scanDate(scan)]
      .filter(Boolean)
      .join(' · ');
    const sub = document.createElement('span');
    sub.className = 'scan-sub';
    if (scan.status === 'complete') {
      const verdict = verdictFor(scan);
      sub.textContent = `${verdict.placed}/${verdict.input} placed · ${presetLabel(scan.preset)}`;
    } else if (scan.status === 'failed') {
      sub.textContent = scan.error?.title || 'Reconstruction failed';
    } else {
      sub.textContent = `${scan.acceptedFrames} frames · ${presetLabel(scan.preset)}`;
    }
    meta.append(date, sub);

    const remove = document.createElement('button');
    remove.className = 'scan-delete';
    remove.type = 'button';
    remove.title = 'Delete scan';
    remove.setAttribute('aria-label', 'Delete scan');
    remove.innerHTML =
      '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      armDelete(remove, scan);
    });

    card.append(thumb, meta, remove);
    const activate = () => (compare.picking ? togglePick(scan) : openScan(scan));
    card.addEventListener('click', activate);
    card.addEventListener('keydown', (event) => {
      if (event.target === card && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        activate();
      }
    });
    el.libraryList.appendChild(card);
  }
}

/** Two-step delete: the first click arms the button, a second within 3 s confirms. */
function armDelete(button, scan) {
  if (button.dataset.armed === 'true') {
    deleteScan(scan);
    return;
  }
  button.dataset.armed = 'true';
  button.append(' Delete?');
  setTimeout(() => {
    if (!button.isConnected) return;
    button.dataset.armed = 'false';
    button.lastChild.remove();
  }, 3000);
}

async function deleteScan(scan) {
  try {
    await api(`/api/scans/${scan.id}`, { method: 'DELETE' });
  } catch (error) {
    toast(error.message);
    return;
  }
  toast('Scan deleted.');
  if (scan.id === state.scanId) resetToReady();
  refreshLibrary();
}

async function openScan(scan) {
  if (state.capturing) return;
  if (scan.status === 'complete') {
    stopPolling();
    forgetFrames();
    state.scanId = scan.id;
    await showResult(scan.id, scan);
  } else if (scan.status === 'failed') {
    stopPolling();
    state.scanId = scan.id;
    renderLibrary();
    showError(scan.error || { title: 'Reconstruction failed', message: '', hints: [] }, await fetchLog(scan.id));
  } else if (PROCESSING.has(scan.status)) {
    state.scanId = scan.id;
    el.processingSub.textContent = `${scan.acceptedFrames} frames · ${presetLabel(scan.preset)} quality`;
    showProcessing();
    renderLibrary();
    startPolling(scan.id);
  } else {
    toast('This scan was stopped before it finished, so there is nothing to show.');
  }
}

// ── compare ─────────────────────────────────────────────────────────────────

function setComparePicking(on) {
  if (on && state.capturing) {
    toast('Finish the current scan first.');
    return;
  }
  compare.picking = on;
  compare.picked = [];
  el.btnCompare.setAttribute('aria-pressed', String(on));
  el.library.dataset.picking = String(on);
  el.libraryPick.hidden = !on;
  if (on && el.shell.dataset.library !== 'open') setLibraryOpen(true);
  renderLibrary();
}

function togglePick(scan) {
  if (scan.status !== 'complete') {
    toast('Only finished scans can be compared.');
    return;
  }
  const index = compare.picked.findIndex((picked) => picked.id === scan.id);
  if (index >= 0) compare.picked.splice(index, 1);
  else compare.picked.push(scan);
  el.libraryPick.textContent =
    compare.picked.length === 1 ? 'Now pick a second scan.' : 'Pick two finished scans to compare.';
  if (compare.picked.length === 2) {
    const [a, b] = compare.picked;
    setComparePicking(false);
    openCompare(a, b);
  } else {
    renderLibrary();
  }
}

function closeCompareViewers() {
  if (!compare.viewers.length) return;
  compare.viewers.forEach((viewer) => viewer.dispose());
  compare.viewers = [];
  el.compareCanvas = el.compareCanvas.map(freshCanvas);
}

async function openCompare(a, b) {
  stopPolling();
  compare.returnTo = !el.views.result.hidden ? state.scanId : null;
  state.viewer?.stop(); // hidden, so it need not keep drawing
  setView('compare');
  closeCompareViewers();
  compare.scans = [a, b];

  const viewers = el.compareCanvas.map((canvas) => new PointCloudViewer(canvas));
  compare.viewers = viewers;
  viewers.forEach((viewer, side) => {
    viewer.setAutoRotate(false);
    viewer.onOrbit = (orbit) => {
      if (el.compareSync.checked) viewers[1 - side].setOrbit(orbit);
    };
    viewer.start();
    const tag = document.createElement('b');
    tag.textContent = 'AB'[side];
    el.compareLabel[side].replaceChildren(tag, scanTitle(compare.scans[side]));
    el.compareHead[side].textContent = `${'AB'[side]} · ${scanTitle(compare.scans[side])}`;
  });
  requestAnimationFrame(() => viewers.forEach((viewer) => viewer.resize()));
  renderCompareTable(null);

  await Promise.all([loadCompareSide(0), loadCompareSide(1), loadCompareMetrics()]);
  // Start both from the same framing-relative view.
  if (compare.viewers === viewers && el.compareSync.checked) viewers[1].setOrbit(viewers[0].getOrbit());
}

async function loadCompareSide(side) {
  const viewer = compare.viewers[side];
  const scan = compare.scans[side];
  el.compareLoading[side].hidden = false;
  try {
    const result = await api(`/api/scans/${scan.id}/result`);
    let cameras = [];
    if (result.camerasUrl) {
      try {
        cameras = (await api(result.camerasUrl)).cameras || [];
      } catch {
        /* orientation only */
      }
    }
    if (compare.viewers[side] !== viewer) return; // compare mode was left meanwhile
    await viewer.load(result.plyUrl, cameras);
    viewer.setAutoRotate(false);
  } catch (error) {
    toast(`Could not load scan ${'AB'[side]}: ${error.message}`);
  } finally {
    el.compareLoading[side].hidden = true;
  }
}

async function loadCompareMetrics() {
  const scans = compare.scans;
  try {
    const metrics = await Promise.all(scans.map((scan) => api(`/api/scans/${scan.id}/metrics`)));
    if (compare.scans === scans) renderCompareTable(metrics);
  } catch (error) {
    toast(error.message);
  }
}

const fixed = (digits, unit = '') => (value) => `${value.toFixed(digits)}${unit}`;

/** `better`: which direction wins the row; rows without one are context, not a contest. */
const COMPARE_ROWS = [
  { label: 'Points', value: (m) => m.points, better: 'high', format: (v) => v.toLocaleString() },
  {
    label: 'Frames placed',
    hint: 'of the accepted frames',
    value: (m) => (m.accepted ? m.placed / m.accepted : null),
    better: 'high',
    format: (v, m) => `${m.placed} / ${m.accepted} · ${Math.round(v * 100)}%`,
  },
  { label: 'Frames captured', value: (m) => m.captured, format: String },
  { label: 'Preset', value: (m) => m.preset, format: presetLabel },
  {
    label: 'Reconstruction time',
    value: (m) => m.durationSeconds,
    better: 'low',
    format: fixed(1, ' s'),
  },
  { label: 'Weak links', hint: 'frames where tracking was lost', value: (m) => m.weakLinks, better: 'low', format: String },
  { label: 'Rotation frames', hint: 'turning in place, no depth', value: (m) => m.rotationFrames, better: 'low', format: String },
  {
    label: 'Mean track length',
    hint: 'views that see each point',
    value: (m) => m.meanTrackLength,
    better: 'high',
    format: fixed(2),
  },
  {
    label: 'Mean reprojection error',
    hint: 'pixels, lower is a tighter fit',
    value: (m) => m.meanReprojectionError,
    better: 'low',
    format: fixed(3, ' px'),
  },
];

function renderCompareTable(metrics) {
  const body = el.compareTable.tBodies[0];
  body.innerHTML = '';
  for (const row of COMPARE_ROWS) {
    const tr = document.createElement('tr');
    const th = document.createElement('td');
    th.textContent = row.label;
    if (row.hint) {
      const hint = document.createElement('span');
      hint.className = 'hint';
      hint.textContent = row.hint;
      th.appendChild(hint);
    }
    tr.appendChild(th);

    const values = metrics ? metrics.map((m) => row.value(m)) : [null, null];
    const numeric = values.every((v) => typeof v === 'number' && Number.isFinite(v));
    let best = -1;
    if (row.better && numeric && Math.abs(values[0] - values[1]) > 1e-9) {
      const firstWins = row.better === 'high' ? values[0] > values[1] : values[0] < values[1];
      best = firstWins ? 0 : 1;
    }
    values.forEach((value, side) => {
      const td = document.createElement('td');
      if (!metrics) {
        td.textContent = '…';
        td.className = 'na';
      } else if (value === null || value === undefined) {
        td.textContent = '—';
        td.className = 'na';
        td.title = 'Not recorded for this scan';
      } else {
        td.textContent = row.format(value, metrics[side]);
        if (side === best) td.className = 'best';
      }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}

function closeCompare() {
  const returnTo = state.scans.find((scan) => scan.id === compare.returnTo);
  closeCompareViewers();
  compare.scans = [];
  if (returnTo) openScan(returnTo);
  else resetToReady();
}

// ── error ───────────────────────────────────────────────────────────────────

function showError(error, log = '') {
  setView('error');
  el.errorTitle.textContent = error.title || 'Something went wrong';
  el.errorMessage.textContent = error.message || '';
  el.errorHints.innerHTML = '';
  for (const hint of error.hints || []) {
    const item = document.createElement('li');
    item.textContent = hint;
    el.errorHints.appendChild(item);
  }
  el.errorTech.hidden = !log;
  el.errorLog.textContent = log;
  el.btnErrorSettings.hidden = error.code !== 'colmap_missing';
}

// ── find record ─────────────────────────────────────────────────────────────

const RECORD_FIELDS = ['findNumber', 'siteCode', 'context', 'material', 'materialOther', 'dateFound', 'recorder', 'notes'];

/** Materials and length limits come from the server, which enforces the same ones. */
function setupRecordForm(config) {
  const select = el.recordMaterial;
  select.replaceChildren(new Option('—', ''));
  for (const { value, label } of config.materials) select.append(new Option(label, value));
  for (const name of RECORD_FIELDS) {
    const input = el.recordForm.elements[name];
    if (input && config.limits[name] && input.tagName !== 'SELECT') input.maxLength = config.limits[name];
  }
}

function recordFromForm() {
  const record = {};
  for (const name of RECORD_FIELDS) record[name] = el.recordForm.elements[name].value;
  if (record.material !== 'other') record.materialOther = '';
  return record;
}

function fillRecordForm(record) {
  for (const name of RECORD_FIELDS) el.recordForm.elements[name].value = record?.[name] || '';
  el.recordOther.hidden = el.recordMaterial.value !== 'other';
}

function showRecordError(message, field) {
  el.recordError.hidden = !message;
  el.recordError.textContent = message || '';
  for (const name of RECORD_FIELDS) {
    el.recordForm.elements[name].setAttribute('aria-invalid', String(name === field));
  }
  if (field) el.recordForm.elements[field]?.focus();
}

async function openRecordForm(target) {
  recordForm.target = target;
  showRecordError('');
  let record = null;
  if (target.kind === 'pending') {
    record = state.pendingRecord;
    el.recordContext.textContent = 'Attached to the next scan when you press Start Scan. You can edit it afterwards.';
  } else {
    el.recordContext.textContent = 'Stored with this scan and printed in its find report.';
    try {
      record = (await api(`/api/scans/${target.id}/record`)).record;
    } catch (error) {
      toast(error.message);
      return;
    }
  }
  fillRecordForm(record);
  el.recordModal.hidden = false;
  el.recordForm.elements.findNumber.focus();
}

function closeRecordForm() {
  el.recordModal.hidden = true;
  recordForm.target = null;
}

async function saveRecord(event) {
  event.preventDefault();
  const record = recordFromForm();
  if (!record.findNumber.trim()) {
    showRecordError('A find number is required.', 'findNumber');
    return;
  }
  const target = recordForm.target;
  if (target?.kind === 'pending') {
    state.pendingRecord = record;
    renderPendingRecord();
    closeRecordForm();
    return;
  }
  try {
    const saved = (await putJSON(`/api/scans/${target.id}/record`, record)).record;
    closeRecordForm();
    toast('Find record saved.');
    if (state.result?.id === target.id) {
      state.result.record = saved;
      renderResultTitle(state.result);
    }
    refreshLibrary();
  } catch (error) {
    showRecordError(error.message, error.field);
  }
}

function renderPendingRecord() {
  const record = state.pendingRecord;
  el.btnRecordPending.dataset.set = String(Boolean(record));
  el.pendingRecordLabel.textContent = record ? `Find ${record.findNumber}` : 'Add find record';
}

// ── marker size ─────────────────────────────────────────────────────────────

/** The measured size of a printed marker, if the user entered one; null means the design size. */
function markerSizeSetting() {
  const value = Number(storageGet('markerSizeMm'));
  return value > 0 ? value : null;
}

function saveMarkerSize() {
  const raw = el.markerSize.value.trim();
  const [low, high] = state.system.markerBoard.sizeRangeMm;
  if (raw && !(Number(raw) >= low && Number(raw) <= high)) {
    el.markerNote.textContent = `Enter a size between ${low} and ${high} mm, or leave it empty for the printed 30 mm.`;
    return false;
  }
  storageSet('markerSizeMm', raw);
  el.markerNote.textContent = raw
    ? `New scans use ${raw} mm markers.`
    : 'New scans use the designed 30 mm markers.';
  return true;
}

async function remeasureOpenScan() {
  if (!state.result || !saveMarkerSize()) return;
  el.markerNote.textContent = 'Measuring the board again…';
  try {
    const scale = await postJSON(`/api/scans/${state.result.id}/scale`, { markerSizeMm: markerSizeSetting() });
    state.result.scale = scale;
    state.result.scaleUncertaintyPct = scale.uncertaintyPct;
    renderScale(state.result);
    state.viewer?.setScale(scale.status === 'scaled' ? scale.mmPerUnit : null);
    setMeasuring(false);
    el.markerNote.textContent =
      scale.status === 'scaled'
        ? `Re-measured with ${scale.markerSizeMm} mm markers.`
        : `Re-measured: ${SCALE_REASONS[scale.status] || scale.status}`;
  } catch (error) {
    el.markerNote.textContent = error.message;
  }
}

// ── events ──────────────────────────────────────────────────────────────────

el.btnStart.addEventListener('click', () => {
  if (!state.system?.colmap.available) {
    showError({
      code: 'colmap_missing',
      title: '3D reconstruction engine not found',
      message:
        'COLMAP must be installed to build a map. You can still capture frames, but ' +
        'reconstruction will fail until COLMAP is available.',
      hints: [
        'Install COLMAP, then open Settings and press Re-detect',
        'Or enter the full path to COLMAP.bat in Settings',
      ],
    });
    return;
  }
  startScan();
});

el.btnFinish.addEventListener('click', () => finishScan());
el.btnRetryCamera.addEventListener('click', () => initCamera());
el.cameraSelect.addEventListener('change', async (event) => {
  try {
    await startCamera(event.target.value);
  } catch (error) {
    cameraFailure(error);
  }
});

el.btnCancel.addEventListener('click', async () => {
  if (state.scanId) {
    try {
      await postJSON(`/api/scans/${state.scanId}/cancel`);
    } catch {
      /* cancelling is best-effort */
    }
  }
  resetToReady();
  refreshLibrary();
});

el.btnNewScan.addEventListener('click', () => resetToReady());
el.btnRecordPending.addEventListener('click', () => openRecordForm({ kind: 'pending' }));
el.btnRecord.addEventListener('click', () => state.scanId && openRecordForm({ kind: 'scan', id: state.scanId }));
el.btnReport.addEventListener('click', () => {
  if (state.scanId) window.open(`/api/scans/${state.scanId}/report.pdf`, '_blank', 'noopener');
});
el.btnMeasure.addEventListener('click', () =>
  setMeasuring(el.btnMeasure.getAttribute('aria-pressed') !== 'true'),
);
el.recordForm.addEventListener('submit', saveRecord);
el.recordForm.addEventListener('input', (event) => {
  // Whatever was wrong is being fixed: stop shouting about it.
  if (event.target.getAttribute('aria-invalid') === 'true') showRecordError('');
});
el.recordMaterial.addEventListener('change', () => {
  el.recordOther.hidden = el.recordMaterial.value !== 'other';
});
el.btnCloseRecord.addEventListener('click', closeRecordForm);
el.btnRecordClear.addEventListener('click', () => {
  fillRecordForm(null);
  if (recordForm.target?.kind === 'pending') {
    state.pendingRecord = null;
    renderPendingRecord();
  }
});
el.recordModal.addEventListener('click', (event) => {
  if (event.target === el.recordModal) closeRecordForm();
});
el.librarySearch.addEventListener('input', onLibrarySearch);
el.btnMarkerSize.addEventListener('click', saveMarkerSize);
el.btnRemeasure.addEventListener('click', remeasureOpenScan);
el.btnErrorRetry.addEventListener('click', () => resetToReady());
el.btnErrorSettings.addEventListener('click', () => openSettings());
el.btnResetView.addEventListener('click', () => state.viewer?.resetView());

el.btnExport.addEventListener('click', () => {
  if (!state.scanId) return;
  const link = document.createElement('a');
  link.href = `/api/scans/${state.scanId}/map.ply`;
  link.download = `${state.scanId}.ply`;
  link.click();
});

el.btnOpenFolder.addEventListener('click', async () => {
  if (!state.scanId) return;
  try {
    await postJSON(`/api/scans/${state.scanId}/reveal`);
  } catch (error) {
    toast(error.message);
  }
});

el.pointSize.addEventListener('input', (event) =>
  state.viewer?.setPointSize(Number(event.target.value)),
);
el.showCameras.addEventListener('change', (event) =>
  state.viewer?.setCamerasVisible(event.target.checked),
);
el.autoRotate.addEventListener('change', (event) =>
  state.viewer?.setAutoRotate(event.target.checked),
);

el.btnLibrary.addEventListener('click', () =>
  setLibraryOpen(el.shell.dataset.library !== 'open'),
);
el.btnCompare.addEventListener('click', () => setComparePicking(!compare.picking));
el.btnCompareClose.addEventListener('click', () => closeCompare());
el.btnCompareReset.addEventListener('click', () => {
  compare.viewers.forEach((viewer) => viewer.resetView());
});
el.compareSync.addEventListener('change', (event) => {
  const [a, b] = compare.viewers;
  if (event.target.checked && a && b) b.setOrbit(a.getOrbit());
});
el.btnSettings.addEventListener('click', openSettings);
el.btnCloseSettings.addEventListener('click', closeSettings);
el.settingsModal.addEventListener('click', (event) => {
  if (event.target === el.settingsModal) closeSettings();
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!el.recordModal.hidden) closeRecordForm();
  else if (!el.settingsModal.hidden) closeSettings();
  else if (compare.picking) setComparePicking(false);
});

el.btnSaveColmap.addEventListener('click', async () => {
  try {
    const payload = await postJSON('/api/system/colmap', { path: el.colmapPath.value });
    state.system.colmap = payload.colmap;
    renderEngine();
    toast('COLMAP configured.');
  } catch (error) {
    el.engineNote.textContent = error.message;
  }
});

el.btnDetectColmap.addEventListener('click', async () => {
  const payload = await postJSON('/api/system/colmap/detect');
  state.system.colmap = payload.colmap;
  renderEngine();
  toast(payload.colmap.available ? 'COLMAP detected.' : 'COLMAP still not found.');
});

el.btnImport.addEventListener('click', async () => {
  el.importNote.textContent = 'Copying images…';
  try {
    const scan = await postJSON('/api/dev/import', {
      folder: el.importFolder.value,
      preset: state.preset,
      markerSizeMm: markerSizeSetting(),
    });
    forgetFrames();
    state.scanId = scan.id;
    state.acceptedFrames = scan.acceptedFrames;
    el.importNote.textContent = `Reconstructing ${scan.acceptedFrames} images…`;
    closeSettings();
    el.processingSub.textContent =
      `${scan.acceptedFrames} imported images · ${presetLabel(state.preset)} quality`;
    showProcessing();
    startPolling(scan.id);
    refreshLibrary();
  } catch (error) {
    el.importNote.textContent = error.message;
  }
});

window.addEventListener('beforeunload', () => stopStream());

// ── boot ────────────────────────────────────────────────────────────────────

(async function boot() {
  buildMeter();
  setLibraryOpen(storageGet('library') !== 'closed');
  try {
    await loadSystem();
  } catch (error) {
    showError({ title: 'Could not reach the local server', message: error.message, hints: [] });
    return;
  }
  setView('capture');
  renderStages(null);
  refreshLibrary();
  await initCamera();
})();
