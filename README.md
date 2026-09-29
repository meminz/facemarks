# 3D Facial Landmarks Detection
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)[![Framework](https://img.shields.io/badge/Framework-Python_3.11-yellow)](https://www.python.org/downloads/release/python-3110/)

A Python package for detecting and analyzing 3D facial landmarks (facemarks) from mesh data. This package provides a complete pipeline for importing 3D facial meshes, predicting facial landmark positions in 3D space, and visualizing results.

## Features

- **Mesh Import**: Import 3D facial meshes with texture support via Open3D
- **3D Landmark Prediction**: Detect up to 468 facial landmarks in 3D space using multi-view projection and raycasting
- **JSON Export**: Save predicted facemarks with normalized coordinates and closest vertex indices
- **Visualization**: Render meshes with detected landmarks overlaid
- **Robust Detection**: Uses multiple camera projections for accurate 3D reconstruction

## Requirements

- Python 3.11
- MediaPipe Face Landmarker model, in the working directory from which you run `predict()`:
    ```bash
    wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    ```
  If the file is missing, `predict()` raises an error with this command.

## Installation

Install the package via pip:

```bash
pip install facemarks
```

## Quick Start

```python
from facemarks import (
    import_mesh_and_setup,
    predict,
    save_facemarks_json,
    render_result
)

# Import and setup your 3D mesh
meshes = import_mesh_and_setup("path/to/mesh.obj")

# Predict 3D facial landmarks
prediction_result = predict(meshes, projections=100)

if prediction_result is None:
    print("No face detected; nothing to export or render.")
else:
    # Extract results
    facemarks_3d = prediction_result["facemarks_3d"]
    closest_vertex_ids = prediction_result["closest_vertex_ids"]

    # Save results directly to JSON (unreconstructed landmarks become null)
    save_facemarks_json(
        "path/to/mesh.obj",
        prediction_result,
        "output/facemarks.json"
    )

    # Visualize only the reconstructed landmarks
    detected = [coords for coords in facemarks_3d if coords is not None]
    render_result(meshes["textured"], detected)
```

## API Reference

### `import_mesh_and_setup(filename)`

Imports a 3D facial mesh and prepares it for landmark detection. Supports textured OBJ files.

**Parameters:**
- `filename` (str): Path to the mesh file (`.obj` format)

**Returns:**
- `dict`: Dictionary containing mesh data with keys:
  - `"original"`: The base mesh geometry
  - `"textured"`: Textured mesh for rendering
  - `"tensor"`:   Tensor representation for raycasting

**Example:**
```python
meshes = import_mesh_and_setup("face_model.obj")
```

---

### `predict(meshes, projections=100, landmarks=tuple(range(468)))`

Predicts 3D facial landmarks from the provided mesh data using multi-view projection and raycasting. By default it computes all 468 face landmarks; landmarks that could not be reconstructed are kept as `None` so every result is a fixed-length list.

**Parameters:**
- `meshes` (dict): Mesh data object returned from `import_mesh_and_setup()`
- `projections` (int, optional): Number of camera projections to use for landmark detection. Default is 100. More projections increase accuracy but take longer.
- `landmarks` (iterable of int, optional): Indices of the landmarks to reconstruct, anywhere in `0..477`. Defaults to `tuple(range(468))` - all 468 face landmarks (MediaPipe's 10 iris landmarks, indices 468-477, are excluded unless you list them explicitly). Only the requested landmarks are computed, so passing a small subset is faster.

**Returns:**
- `dict` (or `None`): Dictionary containing:
  - `"facemarks_3d"` (list): 3D coordinates `[x, y, z]` for each landmark, or `None` where a landmark could not be reconstructed
  - `"closest_vertex_ids"` (list): Index of the closest mesh vertex to each landmark (same length, `None` in the same slots). Position `n` always refers to facemark `n`. The list length is `max(index) + 1` - 468 for the default full face.

**Example:**
```python
prediction_result = predict(meshes, projections=150)
if prediction_result is not None:
    landmarks = prediction_result["facemarks_3d"]
    vertex_ids = prediction_result["closest_vertex_ids"]

# Only compute the landmarks you need
prediction_result = predict(meshes, projections=150, landmarks=(33, 133, 152))
```

**Note:** Camera angles are chosen at random, so results vary slightly between runs, and different runs may resolve different subsets of the 468 facemarks. If no projection detects a face, `predict()` prints an error and returns `None`.

---

### `save_facemarks_json(input_path, prediction_result, json_path)`

Exports predicted facial landmarks to a JSON file with normalized coordinates and vertex mapping.

**Parameters:**
- `input_path` (str): Path to the original input mesh file
- `prediction_result` (dict): Prediction data object returned from `predict()`
- `json_path` (str): Destination path for the output JSON file

**Output JSON Structure:**
```json
{
    "model": "path/to/mesh.obj",
    "normalized coordinates": [[x, y, z], null, ...],
    "closest vertex indexes": [idx, null, ...]
}
```

Both coordinate arrays have one entry per landmark - 468 by default, up to 478 if iris landmarks (468-477) are explicitly requested via `predict()`. Landmarks that could not be reconstructed are written as `null`, so position `n` always refers to facemark `n`.

**Example:**
```python
save_facemarks_json(
    "input/face.obj",
    prediction_result,
    "output/landmarks.json"
)
```

---

### `render_result(mesh, facemarks)`

Visualizes the mesh with predicted facial landmarks overlaid as magenta points.

**Parameters:**
- `mesh`: The mesh geometry to be displayed (either from `meshes["original"]` or `meshes["textured"]`)
- `facemarks` (list): List of 3D landmark coordinates to visualize

**Example:**
```python
render_result(
    meshes["original"],
    [coords for coords in prediction_result["facemarks_3d"] if coords is not None]
)
```

**Note:** Requires a graphical display. In headless environments (unset `DISPLAY` or a virtual `:99` display) it prints a message and returns without rendering. Pass only concrete landmark coordinates - filter out the `None` entries from `facemarks_3d` first.

## How It Works

The package uses a sophisticated multi-view approach to detect 3D facial landmarks:

1. **Multi-View Projection**: The mesh is rendered from multiple random camera angles (default: 100 views)
2. **2D Landmark Detection**: MediaPipe Face Landmark detection is applied to each rendered view
3. **Ray Casting**: Rays are cast from camera positions through detected 2D landmarks onto the 3D mesh
4. **3D Reconstruction**: The intersection points are aggregated across all views to compute robust 3D landmark positions
5. **Vertex Mapping**: Each landmark is mapped to the closest vertex on the original mesh

## Use Cases

- Facial animation and rigging
- 3D face model analysis
- Biometric applications
- Character modeling pipelines
- Medical and dental visualization
- Motion capture alignment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/meminz/facemarks).
