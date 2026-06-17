# Headset FOV Designer

A single-file, browser-based tool for designing camera placement and field of view on a head-mounted device. Position any number of cameras in 3D, set their optics, see each one's field of view as a translucent frustum, highlight the volume that several cameras can see at once, and drop in a reference model to design against.

No install and no build step — it's one self-contained HTML file.

---

## Quick start

Open `headset-fov-designer.html` in any modern browser (double-click it, or drag it into a browser tab). The page loads the three.js library from a CDN on first open, so an internet connection is needed the first time.

It starts seeded with a stereo camera pair so there's something to look at immediately.

**Navigation**
- **Orbit:** click-drag in the viewport
- **Zoom:** scroll
- **Select a camera:** click its frustum or its apex marker

---

## Features

### Cameras
- Add, duplicate, hide, and delete any number of cameras from the list.
- Per-camera controls: name, color, position (x/y/z), orientation (yaw/pitch/roll), and optics.
- Optics are defined by **focal length** and **sensor width/height** (in millimetres). The field of view is derived from these.
- Optics presets: *Wide 90°*, *Standard*, *Tele*, *Full-frame 24mm*.
- A live readout shows horizontal, vertical, and diagonal FOV plus the sensor aspect ratio.
- Each camera's frustum is drawn as a translucent colored pyramid with its apex pinned to the camera position. A range slider sets how far the frustum extends; an opacity slider controls fill transparency.

### FOV intersection
- Tick the checkbox next to any cameras in the list to add them to the **intersection set**, then turn on **Intersection mode**.
- With two or more cameras selected, the app computes the exact volume seen by *all* of them, draws it as a bright highlighted solid, and hides the individual frustum fills so only the shared region is visible.
- The overlap recomputes live as you move or re-aim cameras or change their optics.
- Status feedback: *Off* / *Pick ≥2* / *No overlap* / *Overlap* (with the overlap volume in cubic scene units).
- A highlight-opacity slider and a *Frame intersection* button are provided.

### Reference model
- Import a model to design against (a head, a headset shell, a rig).
- Supported formats: `.glb`, `.gltf`, `.obj`, `.stl`. You can use the Import button or drag a file onto the viewport.
- Imported models are auto-fit (largest dimension scaled to about 1 scene unit) and centered on the origin so they appear alongside the camera rig rather than dwarfing it.
- Full pose controls: **Model position** (x/y/z) and **Model orientation** (yaw/pitch/roll), plus a **scale** multiplier on top of the auto-fit.
- *Reset pose & scale* returns the model to centered/fitted and reframes the view.

### Scene
- Toggle the ground grid and origin axes.
- *Frame all* fits the whole scene in view.

### Save / load
- **Save** writes the full working state to a `headset-fov-project.json` file you can reload later.
- **Load** restores that state.
- **Export JSON** writes a separate, human-readable specification (including computed FOV angles) for handing off to another tool or pipeline.

See [Data formats](#data-formats) for the structure of both files.

---

## Units and conventions

- **Scene units are arbitrary.** They are treated loosely as metres in the defaults — the seeded cameras sit about 0.06 units (≈ 6 cm) apart with a frustum range of 1.2 units.
- **Focal length and sensor size are in millimetres**, but only their *ratio* affects the FOV angle, so there is no unit clash with the scene: a camera's frustum angle is correct regardless of the scene's unit interpretation.
- **Field of view** is computed as:
  - `HFOV = 2 · atan( (sensor_width  / 2) / focal_length )`
  - `VFOV = 2 · atan( (sensor_height / 2) / focal_length )`
  - The diagonal FOV uses the sensor's diagonal in place of width/height.
- **Orientation** uses yaw/pitch/roll in degrees with Euler order `YXZ`, the same convention for cameras and for the reference model. A camera with zero orientation looks down its local **−Z** axis (matching a real perspective camera).

---

## Data formats

### Project file (Save / Load)

A reloadable snapshot of the whole session.

```json
{
  "version": 1,
  "cameras": [
    {
      "name": "Left",
      "color": "#4cc9f0",
      "pos": { "x": -0.06, "y": 0, "z": 0 },
      "rot": { "yaw": 20, "pitch": 0, "roll": 0 },
      "focal": 2.0,
      "sensorW": 4.0,
      "sensorH": 3.0,
      "far": 1.2,
      "opacity": 0.16,
      "visible": true,
      "inSet": true
    }
  ],
  "selectedIndex": 0,
  "intersect": { "mode": false, "opacity": 0.5 },
  "scene": { "showGrid": true, "showAxes": true },
  "model": {
    "name": "head.stl",
    "pose": { "x": 0, "y": 0, "z": 0, "yaw": 0, "pitch": 0, "roll": 0 },
    "scale": 1
  }
}
```

The reference model's **geometry is not stored** (a mesh can't live in a JSON file). Only its pose and scale are saved. After loading a project, re-import the model file — its saved pose and scale reapply automatically instead of resetting to the centered default.

### Export JSON (specification)

A read-only spec intended for downstream use, with computed FOV angles included.

```json
{
  "cameras": [
    {
      "name": "Left",
      "color": "#4cc9f0",
      "position": { "x": -0.06, "y": 0, "z": 0 },
      "rotation_deg": { "yaw": 20, "pitch": 0, "roll": 0 },
      "focal_mm": 2.0,
      "sensor_mm": { "width": 4.0, "height": 3.0 },
      "range": 1.2,
      "hfov_deg": 90.0,
      "vfov_deg": 73.74
    }
  ],
  "model": {
    "position": { "x": 0, "y": 0, "z": 0 },
    "rotation_deg": { "yaw": 0, "pitch": 0, "roll": 0 },
    "scale_x_autofit": 1,
    "autofit": 0.004
  }
}
```

---

## How it works

**Stack.** Plain HTML/JS with [three.js](https://threejs.org/) (r0.160). The library and its addons — `OrbitControls`, `GLTFLoader`, `OBJLoader`, `STLLoader`, and `ConvexGeometry` — are pulled from a CDN via an import map. There is no bundler and no framework; everything lives in one file.

**Frustum geometry.** Each camera is a `PerspectiveCamera`-style frustum built from its focal length, sensor size, and range. The translucent pyramid and its wireframe outline are generated directly from the four far-plane corners and the apex.

**Intersection.** Each frustum is a convex solid bounded by five planes (four sides plus the far plane). To find the region all selected cameras share, the tool:
1. collects every bounding plane from the selected frustums, in world space;
2. enumerates candidate vertices as the intersection points of every triple of planes;
3. keeps only the points that lie inside *all* of the planes;
4. builds the convex hull of those points with `ConvexGeometry` and renders it.

This half-space-enumeration approach handles an arbitrary number of cameras, reports an empty result cleanly when the frustums don't overlap, and is robust to coplanar faces (e.g. near-identical cameras). The overlap volume is computed from the hull via the divergence theorem.

**Model loading.** Imported files are read into memory with `FileReader` and handed to each loader's `parse()` method, rather than fetching a blob URL. This avoids the loaders' internal `fetch()`, which fails inside sandboxed preview iframes that proxy network requests.

---

## Known limitations

- **Multi-file `.gltf`** (a `.gltf` that references separate `.bin` or texture files) won't fully load from a single dropped file, because there's no folder to resolve the companions from. Use a self-contained `.glb` instead.
- **The model mesh isn't saved** in the project JSON — only its pose and scale. Re-import the model after loading a project to bring the geometry back.
- **Auto-fit normalizes the model** to roughly the rig's scale, so the displayed coverage is qualitative unless you deliberately set a real-world scale. If you need true-to-life coverage, scale the model to its real dimensions.
- **Sharp-apex degeneracy.** Frustums are modelled as sharp pyramids (no near clip), so an overlap that collapses to an extremely thin sliver can read as *No overlap*. Truncating the apex with a small near cap removes this.

---

## Possible next steps

- Near-cap apex truncation for more robust intersections.
- Pairwise overlap (highlight where any two cameras agree, not just all of them).
- A coverage heatmap, or a top-down 2D coverage plot.
- A true-scale model mode that preserves real dimensions and only reframes the view.

---

## File layout

```
headset-fov-designer.html   # the entire application
README.md                   # this file
```
