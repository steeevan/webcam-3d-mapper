/**
 * Three.js point-cloud viewer.
 *
 * Renders a COLMAP sparse model (`map.ply`) plus the reconstructed camera trajectory.
 * The whole cloud lives in two typed arrays inside one `THREE.Points` object — there is no
 * per-point JavaScript object anywhere, so a few hundred thousand points stay cheap.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { PLYLoader } from 'three/addons/PLYLoader.js';

/** Above this, the cloud is uniformly subsampled before it reaches the GPU. */
const MAX_RENDERED_POINTS = 1_500_000;

const ACCENT = 0x3be8b0;

export class PointCloudViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.pointScale = 1.4;
    this.points = null;
    this.cameraGroup = null;
    this.grid = null;
    this.frame = { center: new THREE.Vector3(), radius: 1 };
    this._running = false;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0c0e, 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
    this.camera.position.set(0, 0, 5);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.7;

    // A separate tiny scene drawn into the corner of the same canvas via scissor test,
    // so the orientation gizmo costs one extra draw call rather than a second WebGL context.
    this.gizmoScene = new THREE.Scene();
    this.gizmoScene.add(new THREE.AxesHelper(1));
    this.gizmoCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 10);

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this.resize();
  }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = () => {
      if (!this._running) return;
      this._raf = requestAnimationFrame(tick);
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

  /** Load a PLY point cloud. Resolves with `{ points }`. */
  load(url) {
    return new Promise((resolve, reject) => {
      new PLYLoader().load(
        url,
        (geometry) => {
          try {
            resolve(this._install(geometry));
          } catch (err) {
            reject(err);
          }
        },
        undefined,
        () => reject(new Error('Could not load the point cloud file.')),
      );
    });
  }

  _install(geometry) {
    this._disposePoints();

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

    this.frame = robustBounds(positions);

    const material = new THREE.PointsMaterial({
      size: this._pointWorldSize(),
      sizeAttenuation: true,
      vertexColors: Boolean(colors),
      color: colors ? 0xffffff : ACCENT,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    this._buildGrid();
    this.resetView();
    return { points: positions.count };
  }

  /** Draw the reconstructed camera trajectory as a polyline with a marker per view. */
  setCameras(cameras) {
    this._disposeCameras();
    if (!cameras || cameras.length === 0) return;

    const group = new THREE.Group();
    const positions = new Float32Array(cameras.length * 3);
    cameras.forEach((cam, i) => {
      positions[i * 3] = cam.position[0];
      positions[i * 3 + 1] = cam.position[1];
      positions[i * 3 + 2] = cam.position[2];
    });

    const pathGeometry = new THREE.BufferGeometry();
    pathGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    group.add(
      new THREE.Line(
        pathGeometry,
        new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.55 }),
      ),
    );

    const markerGeometry = new THREE.BufferGeometry();
    markerGeometry.setAttribute('position', new THREE.BufferAttribute(positions.slice(), 3));
    group.add(
      new THREE.Points(
        markerGeometry,
        new THREE.PointsMaterial({
          color: ACCENT,
          size: this.frame.radius * 0.014,
          sizeAttenuation: true,
        }),
      ),
    );

    this.cameraGroup = group;
    this.scene.add(group);
  }

  setCamerasVisible(visible) {
    if (this.cameraGroup) this.cameraGroup.visible = visible;
  }

  setPointSize(scale) {
    this.pointScale = scale;
    if (this.points) this.points.material.size = this._pointWorldSize();
  }

  _pointWorldSize() {
    // Sized in world units so `sizeAttenuation` keeps perspective. The constant is chosen so
    // the default scale lands around 3 device pixels at the reset-view distance - anything
    // smaller renders sub-pixel and the cloud looks empty.
    return Math.max(this.frame.radius * 0.0075 * this.pointScale, 1e-5);
  }

  /** Frame the cloud so the user never has to hunt for the model. */
  resetView() {
    const { center, radius } = this.frame;
    const distance = radius * 2.2;
    this.camera.near = Math.max(radius / 800, 0.001);
    this.camera.far = radius * 60;
    this.camera.position.set(
      center.x + distance * 0.62,
      center.y + distance * 0.42,
      center.z + distance * 0.78,
    );
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
    const { center, radius } = this.frame;
    const grid = new THREE.GridHelper(radius * 6, 24, 0x2a3138, 0x1b2126);
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    grid.position.set(center.x, center.y - radius * 1.05, center.z);
    this.grid = grid;
    this.scene.add(grid);
  }

  _disposePoints() {
    if (!this.points) return;
    this.scene.remove(this.points);
    this.points.geometry.dispose();
    this.points.material.dispose();
    this.points = null;
  }

  _disposeCameras() {
    if (!this.cameraGroup) return;
    this.scene.remove(this.cameraGroup);
    this.cameraGroup.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    this.cameraGroup = null;
  }

  clear() {
    this._disposePoints();
    this._disposeCameras();
  }

  dispose() {
    this.stop();
    this.clear();
    window.removeEventListener('resize', this._onResize);
    this.controls.dispose();
    this.renderer.dispose();
  }
}

/**
 * Centre and radius that ignore stray outliers.
 *
 * Sparse SfM clouds routinely contain a handful of points thousands of units away; framing on
 * the raw bounding box would leave the real model a speck in the middle of the screen. We take
 * the centroid and the 95th-percentile distance from a sample of at most 20k points instead.
 */
function robustBounds(positions) {
  const count = positions.count;
  const center = new THREE.Vector3();
  if (count === 0) return { center, radius: 1 };

  for (let i = 0; i < count; i++) {
    center.x += positions.getX(i);
    center.y += positions.getY(i);
    center.z += positions.getZ(i);
  }
  center.divideScalar(count);

  const stride = Math.max(1, Math.ceil(count / 20000));
  const distances = [];
  for (let i = 0; i < count; i += stride) {
    const dx = positions.getX(i) - center.x;
    const dy = positions.getY(i) - center.y;
    const dz = positions.getZ(i) - center.z;
    distances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  distances.sort((a, b) => a - b);
  const radius = distances[Math.floor(distances.length * 0.95)] || distances[distances.length - 1] || 1;
  return { center, radius: Math.max(radius, 1e-4) };
}
