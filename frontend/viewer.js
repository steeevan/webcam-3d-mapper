/**
 * Three.js point-cloud viewer.
 *
 * Renders a COLMAP sparse model (`map.ply`) plus the reconstructed cameras as small frustums.
 * The whole cloud lives in two typed arrays inside one `THREE.Points` object — there is no
 * per-point JavaScript object anywhere, so a few hundred thousand points stay cheap.
 *
 * COLMAP's world frame is arbitrary (y points *down* in the first camera's frame), so the cloud
 * and cameras sit inside a `world` group rotated so the average camera "up" becomes +Y. Models
 * therefore load upright, standing on the grid, instead of upside down.
 *
 * With a measured scale (`setScale`), the viewer reports a scale bar for the current zoom and
 * offers a two-click distance tool. The `world` group only rotates, so distances between points
 * in its local frame are model units, and model units times `mmPerUnit` are millimetres.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { PLYLoader } from 'three/addons/PLYLoader.js';

/** Above this, the cloud is uniformly subsampled before it reaches the GPU. */
const MAX_RENDERED_POINTS = 1_500_000;

const ACCENT = new THREE.Color(0x3be8b0);
const ACCENT_2 = new THREE.Color(0x3cc8f0);
const INTRO_MS = 1100;

/** A crisp round sprite so points render as dots rather than squares. */
function discTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const easeOut = (t) => 1 - Math.pow(1 - t, 3);

/** The largest 1, 2 or 5 x 10^n not above `max`: a length people can read off a bar. */
function niceLength(max) {
  if (!(max > 0)) return 0;
  const base = 10 ** Math.floor(Math.log10(max));
  for (const step of [5, 2, 1]) if (step * base <= max) return step * base;
  return base / 2;
}

/** Pixels a click may travel and still count as a click, not an orbit drag. */
const CLICK_SLOP_PX = 5;

export class PointCloudViewer {
  /**
   * `gizmo: false` drops the orientation gizmo (small insets); `intro: false` skips the fly-in,
   * which would replay on every live-preview update.
   */
  constructor(canvas, { gizmo = true, intro = true } = {}) {
    this.canvas = canvas;
    this.pointScale = 1.4;
    this.points = null;
    this.cameraGroup = null;
    this.grid = null;
    this.frame = { center: new THREE.Vector3(), radius: 1, floor: -1 };
    this.onUserInteract = null;
    /** Called while the *user* moves the view (not auto-rotate, not setOrbit). */
    this.onOrbit = null;
    this.showGizmo = gizmo;
    this.playIntro = intro;
    this._running = false;
    this._intro = null;
    this._dragging = false;
    this._settling = 0;
    this._applying = false;
    this._disc = discTexture();
    /** Millimetres per model unit, or null when the scan has no measured scale. */
    this.mmPerUnit = null;
    /** Called with `{ mm, px }` when the scale bar changes, or `null` when there is none. */
    this.onScaleBar = null;
    /** Called after each measuring click: `{ picks, mm }` (mm once two points are picked). */
    this.onMeasure = null;
    this._bar = undefined;
    this._measuring = false;
    this._picks = [];
    this._measureGroup = null;
    this._down = null;
    this._raycaster = new THREE.Raycaster();

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Transparent: the CSS gradient behind the canvas is the backdrop.
    this.renderer.setClearColor(0x000000, 0);

    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.scene.add(this.world);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
    this.camera.position.set(0, 0, 5);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.7;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.9;
    this.controls.addEventListener('start', () => {
      this._intro = null;
      this._dragging = true;
      if (this.controls.autoRotate) {
        this.controls.autoRotate = false;
        this.onUserInteract?.();
      }
    });
    this.controls.addEventListener('end', () => {
      this._dragging = false;
      // Damping keeps the view gliding briefly after release; keep reporting until it settles.
      this._settling = performance.now() + 1500;
    });
    this.controls.addEventListener('change', () => {
      if (this._applying || !this.onOrbit) return;
      if (this._dragging || performance.now() < this._settling) this.onOrbit(this.getOrbit());
    });

    // A separate tiny scene drawn into the corner of the same canvas via scissor test,
    // so the orientation gizmo costs one extra draw call rather than a second WebGL context.
    this.gizmoScene = new THREE.Scene();
    this.gizmoScene.add(new THREE.AxesHelper(1));
    this.gizmoCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10);

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this._onPointerDown = (event) => (this._down = { x: event.clientX, y: event.clientY });
    this._onPointerUp = (event) => this._maybePick(event);
    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup', this._onPointerUp);
    this.resize();
  }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = () => {
      if (!this._running) return;
      this._raf = requestAnimationFrame(tick);
      this._animateIntro();
      this.controls.update();
      this.render();
    };
    tick();
  }

  stop() {
    this._running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  render() {
    const { renderer } = this;
    const { width, height } = this._size();

    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.render(this.scene, this.camera);
    this._updateScaleBar(height);
    if (!this.showGizmo) return;

    // Orientation gizmo, bottom-right.
    const size = 62;
    const pad = 14;
    renderer.setScissorTest(true);
    renderer.setViewport(width - size - pad, pad, size, size);
    renderer.setScissor(width - size - pad, pad, size, size);
    this.gizmoCamera.position
      .subVectors(this.camera.position, this.controls.target)
      .normalize()
      .multiplyScalar(3);
    this.gizmoCamera.up.copy(this.camera.up);
    this.gizmoCamera.lookAt(0, 0, 0);
    renderer.render(this.gizmoScene, this.gizmoCamera);
    renderer.setScissorTest(false);
  }

  _size() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    return { width: Math.max(1, Math.floor(rect.width)), height: Math.max(1, Math.floor(rect.height)) };
  }

  resize() {
    const { width, height } = this._size();
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  /**
   * Load a PLY point cloud, oriented by the reconstructed cameras when available.
   * Resolves with `{ points }`.
   */
  load(url, cameras = []) {
    return new Promise((resolve, reject) => {
      new PLYLoader().load(
        url,
        (geometry) => {
          try {
            resolve(this._install(geometry, cameras));
          } catch (err) {
            reject(err);
          }
        },
        undefined,
        () => reject(new Error('Could not load the point cloud file.')),
      );
    });
  }

  _install(geometry, cameras) {
    this.clear();
    this.clearMeasure();

    let positions = geometry.getAttribute('position');
    let colors = geometry.getAttribute('color');
    if (!positions) throw new Error('The point cloud file contains no vertices.');

    // Guard the GPU against pathological clouds by keeping every Nth point.
    if (positions.count > MAX_RENDERED_POINTS) {
      const stride = Math.ceil(positions.count / MAX_RENDERED_POINTS);
      const kept = Math.floor(positions.count / stride);
      const pos = new Float32Array(kept * 3);
      const col = colors ? new Float32Array(kept * 3) : null;
      for (let i = 0; i < kept; i++) {
        const src = i * stride;
        pos[i * 3] = positions.getX(src);
        pos[i * 3 + 1] = positions.getY(src);
        pos[i * 3 + 2] = positions.getZ(src);
        if (col) {
          col[i * 3] = colors.getX(src);
          col[i * 3 + 1] = colors.getY(src);
          col[i * 3 + 2] = colors.getZ(src);
        }
      }
      const trimmed = new THREE.BufferGeometry();
      trimmed.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      if (col) trimmed.setAttribute('color', new THREE.BufferAttribute(col, 3));
      geometry.dispose();
      geometry = trimmed;
      positions = trimmed.getAttribute('position');
      colors = trimmed.getAttribute('color');
    }

    this.world.quaternion.copy(uprightRotation(cameras));
    this.world.updateMatrixWorld(true);
    this.frame = robustBounds(positions, this.world.quaternion);

    const material = new THREE.PointsMaterial({
      size: this._pointWorldSize(),
      sizeAttenuation: true,
      vertexColors: Boolean(colors),
      color: colors ? 0xffffff : ACCENT,
      map: this._disc,
      alphaTest: 0.5,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.world.add(this.points);

    this._buildCameras(cameras);
    this._buildGrid();
    this.resetView();
    this._intro = this.playIntro ? { start: performance.now() } : null;
    return { points: positions.count };
  }

  /**
   * The view relative to this cloud's own framing: orbit angles, distance in cloud radii and
   * pan offset in cloud radii. Two scans have unrelated scales and origins, so syncing these
   * (not absolute camera positions) makes both show "the same view" of their own model.
   */
  getOrbit() {
    const { center, radius } = this.frame;
    const offset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    return {
      theta: spherical.theta,
      phi: spherical.phi,
      distance: spherical.radius / radius,
      pan: new THREE.Vector3().subVectors(this.controls.target, center).divideScalar(radius).toArray(),
    };
  }

  setOrbit({ theta, phi, distance, pan }) {
    const { center, radius } = this.frame;
    this._intro = null;
    this._applying = true;
    this.controls.target.copy(center).addScaledVector(new THREE.Vector3().fromArray(pan), radius);
    this.camera.position
      .setFromSpherical(new THREE.Spherical(distance * radius, phi, theta))
      .add(this.controls.target);
    this.controls.update();
    this._applying = false;
  }

  /** Each reconstructed view as a small frustum, joined by a path that shifts mint → cyan. */
  _buildCameras(cameras) {
    if (!cameras || cameras.length === 0) return;

    const depth = this.frame.radius * 0.045;
    const halfW = depth * 0.5;
    const halfH = depth * 0.28;
    const withFrustum = cameras.filter((cam) => cam.forward && cam.up);

    const segments = new Float32Array(withFrustum.length * 16 * 3);
    const segmentColors = new Float32Array(withFrustum.length * 16 * 3);
    const pathPositions = new Float32Array(cameras.length * 3);
    const pathColors = new Float32Array(cameras.length * 3);
    const colour = new THREE.Color();

    cameras.forEach((cam, i) => {
      pathPositions.set(cam.position, i * 3);
      colour.lerpColors(ACCENT, ACCENT_2, cameras.length > 1 ? i / (cameras.length - 1) : 0);
      pathColors.set([colour.r, colour.g, colour.b], i * 3);
    });

    const apex = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const up = new THREE.Vector3();
    const right = new THREE.Vector3();
    const base = new THREE.Vector3();
    withFrustum.forEach((cam, i) => {
      apex.fromArray(cam.position);
      forward.fromArray(cam.forward).normalize();
      up.fromArray(cam.up).normalize();
      right.crossVectors(forward, up).normalize();
      base.copy(apex).addScaledVector(forward, depth);

      const corner = (sx, sy) =>
        base.clone().addScaledVector(right, sx * halfW).addScaledVector(up, sy * halfH);
      const c = [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)];
      const lines = [
        apex, c[0], apex, c[1], apex, c[2], apex, c[3],
        c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0],
      ];
      colour.lerpColors(ACCENT, ACCENT_2, withFrustum.length > 1 ? i / (withFrustum.length - 1) : 0);
      lines.forEach((v, k) => {
        segments.set([v.x, v.y, v.z], (i * 16 + k) * 3);
        segmentColors.set([colour.r, colour.g, colour.b], (i * 16 + k) * 3);
      });
    });

    const group = new THREE.Group();

    const pathGeometry = new THREE.BufferGeometry();
    pathGeometry.setAttribute('position', new THREE.BufferAttribute(pathPositions, 3));
    pathGeometry.setAttribute('color', new THREE.BufferAttribute(pathColors, 3));
    group.add(
      new THREE.Line(
        pathGeometry,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 }),
      ),
    );

    if (withFrustum.length) {
      const frustumGeometry = new THREE.BufferGeometry();
      frustumGeometry.setAttribute('position', new THREE.BufferAttribute(segments, 3));
      frustumGeometry.setAttribute('color', new THREE.BufferAttribute(segmentColors, 3));
      group.add(
        new THREE.LineSegments(
          frustumGeometry,
          new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }),
        ),
      );
    } else {
      // Older scans without orientation: fall back to a marker per view.
      const markerGeometry = new THREE.BufferGeometry();
      markerGeometry.setAttribute('position', new THREE.BufferAttribute(pathPositions.slice(), 3));
      group.add(
        new THREE.Points(
          markerGeometry,
          new THREE.PointsMaterial({ color: ACCENT, size: this.frame.radius * 0.014 }),
        ),
      );
    }

    this.cameraGroup = group;
    this.world.add(group);
  }

  // -- scale -------------------------------------------------------------------

  /** Millimetres per model unit (null: unscaled, which hides the bar and the tool). */
  setScale(mmPerUnit) {
    this.mmPerUnit = mmPerUnit > 0 ? mmPerUnit : null;
    if (!this.mmPerUnit) this.setMeasuring(false);
    this._bar = undefined;
  }

  /**
   * A perspective view has no single scale: the bar is right for things at the orbit centre
   * (the point the view turns around) and wrong for nearer or further ones.
   */
  _updateScaleBar(height) {
    if (!this.onScaleBar) return;
    if (!this.mmPerUnit || !this.points) {
      if (this._bar !== null) this.onScaleBar((this._bar = null));
      return;
    }
    const distance = this.camera.position.distanceTo(this.controls.target);
    const unitsPerPx = (2 * distance * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) / height;
    const mmPerPx = unitsPerPx * this.mmPerUnit;
    const mm = niceLength(mmPerPx * 150);
    const px = Math.round(mm / mmPerPx);
    if (this._bar && this._bar.mm === mm && this._bar.px === px) return;
    this._bar = { mm, px };
    this.onScaleBar(this._bar);
  }

  // -- two-click measuring ------------------------------------------------------

  setMeasuring(on) {
    this._measuring = Boolean(on && this.mmPerUnit);
    this.clearMeasure();
  }

  clearMeasure() {
    this._picks = [];
    if (this._measureGroup) {
      this.world.remove(this._measureGroup);
      this._measureGroup.traverse((child) => {
        child.geometry?.dispose();
        child.material?.dispose();
      });
      this._measureGroup = null;
    }
  }

  _maybePick(event) {
    const down = this._down;
    this._down = null;
    if (!this._measuring || !this.points || !down) return;
    if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > CLICK_SLOP_PX) return;

    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this._raycaster.setFromCamera(ndc, this.camera);
    this._raycaster.params.Points.threshold = this.frame.radius * 0.012;
    const hits = this._raycaster.intersectObject(this.points);
    if (!hits.length) {
      this.onMeasure?.({ picks: this._picks.length, mm: null, missed: true });
      return;
    }
    // Of the points near the ray, the one the ray passes closest to - not the nearest one in
    // depth, which would favour stray points floating in front of the surface.
    const hit = hits.reduce((best, h) => (h.distanceToRay < best.distanceToRay ? h : best));
    const position = this.points.geometry.getAttribute('position');
    const point = new THREE.Vector3().fromBufferAttribute(position, hit.index);
    const picks = this._picks.length >= 2 ? [point] : [...this._picks, point];
    this._drawMeasure(picks);
    const mm = picks.length === 2 ? picks[0].distanceTo(picks[1]) * this.mmPerUnit : null;
    this.onMeasure?.({ picks: picks.length, mm });
  }

  _drawMeasure(picks) {
    this.clearMeasure();
    this._picks = picks;
    const group = new THREE.Group();
    const size = this.frame.radius * 0.012;
    for (const point of picks) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(size, 12, 8),
        new THREE.MeshBasicMaterial({ color: ACCENT_2, depthTest: false }),
      );
      marker.position.copy(point);
      marker.renderOrder = 10;
      group.add(marker);
    }
    if (picks.length === 2) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(picks),
        new THREE.LineBasicMaterial({ color: ACCENT_2, depthTest: false }),
      );
      line.renderOrder = 10;
      group.add(line);
    }
    this._measureGroup = group;
    this.world.add(group);
  }

  setCamerasVisible(visible) {
    if (this.cameraGroup) this.cameraGroup.visible = visible;
  }

  setAutoRotate(enabled) {
    this.controls.autoRotate = enabled;
  }

  setPointSize(scale) {
    this.pointScale = scale;
    if (this.points && !this._intro) this.points.material.size = this._pointWorldSize();
  }

  _pointWorldSize() {
    // Sized in world units so `sizeAttenuation` keeps perspective. The constant is chosen so
    // the default scale lands around 3 device pixels at the reset-view distance - anything
    // smaller renders sub-pixel and the cloud looks empty.
    return Math.max(this.frame.radius * 0.0075 * this.pointScale, 1e-5);
  }

  /** Points swell into place while the camera glides in from further out. */
  _animateIntro() {
    if (!this._intro || !this.points) return;
    const t = Math.min(1, (performance.now() - this._intro.start) / INTRO_MS);
    const k = easeOut(t);
    this.points.material.size = this._pointWorldSize() * k;
    const { center } = this.frame;
    const offset = this._viewOffset().multiplyScalar(1 + 0.8 * (1 - k));
    this.camera.position.copy(center).add(offset);
    if (t >= 1) this._intro = null;
  }

  _viewOffset() {
    const distance = this.frame.radius * 1.7;
    return new THREE.Vector3(distance * 0.62, distance * 0.42, distance * 0.78);
  }

  /** Frame the cloud so the user never has to hunt for the model. */
  resetView() {
    const { center, radius } = this.frame;
    this.camera.near = Math.max(radius / 800, 0.001);
    this.camera.far = radius * 60;
    this.camera.position.copy(center).add(this._viewOffset());
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.minDistance = radius * 0.02;
    this.controls.maxDistance = radius * 30;
    this.controls.update();
  }

  _buildGrid() {
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
    }
    const { center, radius, floor } = this.frame;
    const grid = new THREE.GridHelper(radius * 6, 30, 0x2c5a4c, 0x1a2229);
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    grid.position.set(center.x, floor, center.z);
    this.grid = grid;
    this.scene.add(grid);
  }

  _disposePoints() {
    if (!this.points) return;
    this.world.remove(this.points);
    this.points.geometry.dispose();
    this.points.material.dispose();
    this.points = null;
  }

  _disposeCameras() {
    if (!this.cameraGroup) return;
    this.world.remove(this.cameraGroup);
    this.cameraGroup.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    this.cameraGroup = null;
  }

  clear() {
    this._intro = null;
    this._disposePoints();
    this._disposeCameras();
  }

  dispose() {
    this.stop();
    this.clear();
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
      this.grid = null;
    }
    window.removeEventListener('resize', this._onResize);
    this.canvas.removeEventListener('pointerdown', this._onPointerDown);
    this.canvas.removeEventListener('pointerup', this._onPointerUp);
    this.clearMeasure();
    this.controls.dispose();
    this.renderer.dispose();
    // renderer.dispose() frees GPU resources but keeps the WebGL context; browsers cap live
    // contexts (Chrome: 16) and silently kill the oldest, so give it back explicitly.
    this.renderer.forceContextLoss();
    this._disc.dispose();
  }
}

/**
 * Rotation that maps the scan's average camera "up" onto +Y.
 *
 * People hold a webcam roughly upright, so the mean of the per-view up vectors is a good
 * gravity estimate. Without cameras we fall back to COLMAP's convention (y down) and flip.
 */
function uprightRotation(cameras) {
  const up = new THREE.Vector3();
  for (const cam of cameras || []) {
    if (cam.up) up.add(new THREE.Vector3().fromArray(cam.up));
  }
  if (up.lengthSq() < 1e-8) up.set(0, -1, 0);
  return new THREE.Quaternion().setFromUnitVectors(up.normalize(), new THREE.Vector3(0, 1, 0));
}

/**
 * Centre, radius and floor height that ignore stray outliers, in the rotated world frame.
 *
 * Sparse SfM clouds routinely contain a handful of points thousands of units away; framing on
 * the raw bounding box would leave the real model a speck in the middle of the screen. We take
 * the per-axis median and the 95th-percentile distance from a sample of at most 20k points.
 */
function robustBounds(positions, rotation) {
  const count = positions.count;
  const center = new THREE.Vector3();
  if (count === 0) return { center, radius: 1, floor: -1 };

  const stride = Math.max(1, Math.ceil(count / 20000));
  const sample = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i += stride) {
    v.set(positions.getX(i), positions.getY(i), positions.getZ(i)).applyQuaternion(rotation);
    sample.push(v.clone());
  }

  const median = (values) => {
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  center.set(
    median(sample.map((p) => p.x)),
    median(sample.map((p) => p.y)),
    median(sample.map((p) => p.z)),
  );

  const distances = sample.map((p) => p.distanceTo(center)).sort((a, b) => a - b);
  const radius = Math.max(distances[Math.floor(distances.length * 0.95)] || 1, 1e-4);

  // Rest the grid just under the bulk of the cloud rather than under its lowest outlier.
  const heights = sample.map((p) => p.y).sort((a, b) => a - b);
  const floor = Math.max(heights[Math.floor(heights.length * 0.03)], center.y - radius * 1.2);

  return { center, radius, floor: floor - radius * 0.02 };
}
