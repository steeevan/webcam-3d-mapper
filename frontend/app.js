/**
 * Application shell: camera, capture loop, state machine, settings.
 *
 * One page, five states (ready / scanning / processing / result / error). The browser owns the
 * webcam and the WebGL viewer; the Python backend owns frames, COLMAP and the filesystem.
 */

import { PointCloudViewer } from './viewer.js';

const $ = (id) => document.getElementById(id);

const el = {
  pillCamera: $('pill-camera'),
  pillEngine: $('pill-engine'),
  btnSettings: $('btn-settings'),

  views: Object.fromEntries(
    [...document.querySelectorAll('[data-view]')].map((node) => [node.dataset.view, node]),
  ),
  rowReady: document.querySelector('[data-when="ready"]'),
  rowScanning: document.querySelector('[data-when="scanning"]'),

  preview: $('preview'),
  grabber: $('grabber'),
  cameraEmpty: $('camera-empty'),
  cameraEmptyTitle: $('camera-empty-title'),
  cameraEmptyBody: $('camera-empty-body'),
  btnRetryCamera: $('btn-retry-camera'),
  scanOverlay: $('scan-overlay'),
  viewportBadge: $('viewport-badge'),
  scanTimer: $('scan-timer'),

  cameraSelect: $('camera-select'),
  presetToggle: $('preset-toggle'),
  btnStart: $('btn-start'),
  btnFinish: $('btn-finish'),
  scanElapsed: $('scan-elapsed'),
  scanFrames: $('scan-frames'),
  scanGuidance: $('scan-guidance'),

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
  pointSize: $('point-size'),
  showCameras: $('show-cameras'),
  statPoints: $('stat-points'),
  statImages: $('stat-images'),
  statTime: $('stat-time'),
  btnResetView: $('btn-reset-view'),
  btnExport: $('btn-export'),
  btnOpenFolder: $('btn-open-folder'),
  btnNewScan: $('btn-new-scan'),

  errorTitle: $('error-title'),
  errorMessage: $('error-message'),
  errorHints: $('error-hints'),
  errorTech: $('error-tech'),
  errorLog: $('error-log'),
  btnErrorRetry: $('btn-error-retry'),
  btnErrorSettings: $('btn-error-settings'),

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
};

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
    throw new Error(body?.detail || `Request failed (${response.status})`);
  }
  return body;
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
  for (const [key, node] of Object.entries(el.views)) node.hidden = key !== name;
  document.documentElement.dataset.state = name;
}

const pad = (n) => String(n).padStart(2, '0');
const clock = (seconds) => `${pad(Math.floor(seconds / 60))}:${pad(Math.floor(seconds % 60))}`;

// ── system / settings ───────────────────────────────────────────────────────

async function loadSystem() {
  state.system = await api('/api/system');
  state.preset = state.system.defaultPreset;
  renderPresets();
  renderEngine();
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
    el.engineStatus.innerHTML = '<span class="bad">Not found</span>';
    el.engineNote.textContent =
      'COLMAP must be installed to build a map. Install it, then enter the full path to ' +
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

    state.acceptedFrames = result.acceptedFrames;
    el.scanFrames.textContent = state.acceptedFrames;

    if (result.limitReached) {
      toast('Frame limit reached — finishing the scan.');
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
    const scan = await postJSON('/api/scans', { preset: state.preset });
    state.scanId = scan.id;
    state.scanStartedAt = Date.now();
    state.acceptedFrames = 0;
    state.capturing = true;

    el.scanFrames.textContent = '0';
    el.scanElapsed.textContent = '00:00';
    el.scanTimer.textContent = '00:00';
    el.rowReady.hidden = true;
    el.rowScanning.hidden = false;
    el.scanOverlay.hidden = false;
    el.viewportBadge.hidden = false;
    el.cameraSelect.disabled = true;
    setView('capture');

    state.captureTimer = setInterval(sendFrame, state.system.capture.intervalMs);
    state.tickTimer = setInterval(() => {
      const seconds = (Date.now() - state.scanStartedAt) / 1000;
      el.scanElapsed.textContent = clock(seconds);
      el.scanTimer.textContent = clock(seconds);
      if (seconds > 8 && state.acceptedFrames < 3) {
        el.scanGuidance.textContent =
          'Few frames are being kept — try more light, or move a little more slowly.';
      }
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
  el.rowScanning.hidden = true;
  el.rowReady.hidden = false;
  el.scanOverlay.hidden = true;
  el.viewportBadge.hidden = true;
  el.cameraSelect.disabled = false;
  el.scanGuidance.textContent = 'Move slowly around the subject and keep it visible.';
}

async function finishScan() {
  if (!state.scanId) return;
  const scanId = state.scanId;
  stopCapture();

  const seconds = (Date.now() - state.scanStartedAt) / 1000;
  el.processingSub.textContent =
    `${state.acceptedFrames} frames · ${clock(seconds)} of scanning · ` +
    `${state.preset === 'fast' ? 'Fast' : 'Balanced'} quality`;

  setView('processing');
  renderStages(null);
  el.techLog.textContent = 'Waiting for output…';

  try {
    await postJSON(`/api/scans/${scanId}/finish`);
    startPolling(scanId);
  } catch (error) {
    showError({ title: 'Could not start reconstruction', message: error.message, hints: [] });
  }
}

// ── processing ──────────────────────────────────────────────────────────────

const STAGE_ORDER = ['preparing', 'extracting', 'matching', 'mapping', 'exporting'];

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

  renderStages(status.status);
  el.progressFill.style.width = `${status.progress}%`;
  el.progressPct.textContent = `${status.progress}%`;
  el.progressDetail.textContent = status.stageDetail || '';

  if (status.status === 'complete') {
    stopPolling();
    await showResult(scanId, status);
  } else if (status.status === 'failed') {
    stopPolling();
    const log = await fetchLog(scanId);
    showError(status.error || { title: 'Reconstruction failed', message: '', hints: [] }, log);
  } else if (status.status === 'cancelled') {
    stopPolling();
    resetToReady();
  } else if (document.querySelector('.tech[open]')) {
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

async function showResult(scanId, status) {
  setView('result');
  el.viewerLoading.hidden = false;
  el.statPoints.textContent = (status.points || 0).toLocaleString();
  el.statImages.textContent = status.registeredImages || 0;
  el.statTime.textContent = `${Math.round(status.result?.duration_seconds ?? status.elapsedProcessing)}s`;

  if (!state.viewer) state.viewer = new PointCloudViewer(el.viewerCanvas);
  const viewer = state.viewer;
  viewer.clear();
  viewer.start();
  // The canvas only has a real size once the result view is visible.
  requestAnimationFrame(() => viewer.resize());

  try {
    const result = await api(`/api/scans/${scanId}/result`);
    const loaded = await viewer.load(result.plyUrl);
    el.statPoints.textContent = loaded.points.toLocaleString();

    if (result.camerasUrl) {
      try {
        const cameras = await api(result.camerasUrl);
        viewer.setCameras(cameras.cameras || []);
        viewer.setCamerasVisible(el.showCameras.checked);
      } catch {
        /* the trajectory overlay is optional */
      }
    }

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
      hints: ['The .PLY file is still on disk — try Export from a previous scan'],
    });
  }
}

function resetToReady() {
  stopPolling();
  stopCapture();
  state.scanId = null;
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
  if (!state.stream) initCamera();
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
});

el.btnNewScan.addEventListener('click', () => resetToReady());
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

el.btnSettings.addEventListener('click', openSettings);
el.btnCloseSettings.addEventListener('click', closeSettings);
el.settingsModal.addEventListener('click', (event) => {
  if (event.target === el.settingsModal) closeSettings();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !el.settingsModal.hidden) closeSettings();
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
    });
    state.scanId = scan.id;
    state.acceptedFrames = scan.acceptedFrames;
    el.importNote.textContent = `Reconstructing ${scan.acceptedFrames} images…`;
    closeSettings();
    el.processingSub.textContent = `${scan.acceptedFrames} imported images`;
    setView('processing');
    renderStages(null);
    startPolling(scan.id);
  } catch (error) {
    el.importNote.textContent = error.message;
  }
});

window.addEventListener('beforeunload', () => stopStream());

// ── boot ────────────────────────────────────────────────────────────────────

(async function boot() {
  try {
    await loadSystem();
  } catch (error) {
    showError({ title: 'Could not reach the local server', message: error.message, hints: [] });
    return;
  }
  setView('capture');
  renderStages(null);
  await initCamera();
})();
