# 3D Facial Landmarks Detection
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)[![Framework](https://img.shields.io/badge/Framework-Python_3.11-yellow)](https://www.python.org/downloads/release/python-3110/)

A Python package for detecting and analyzing 3D facial landmarks (facemarks) from mesh data. This package provides a complete pipeline for importing 3D facial meshes, predicting facial landmark positions in 3D space, and visualizing results.

## Features

- **Mesh Import**: Import 3D facial meshes with texture support via Open3D
- **3D Landmark Prediction**: Detect 478 facial landmarks in 3D space using multi-view projection and raycasting
- **JSON Export**: Save predicted facemarks with normalized coordinates and closest vertex indices
- **Visualization**: Render meshes with detected landmarks overlaid
- **Robust Detection**: Uses multiple camera projections for accurate 3D reconstruction

## Requirements

- Python 3.11
- MediaPipe Face Landmarker model (in working directory)
    ```bash
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    ```

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

# Extract results
facemarks_3d = result["facemarks_3d"]
closest_vertex_ids = result["closest_vertex_ids"]

# or Save results directly to JSON
save_facemarks_json(
    "path/to/mesh.obj",
    prediction_result,
    "output/facemarks.json"
)

# Visualize the results
render_result(meshes["textured"], facemarks_3d)
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

### `predict(meshes, projections=100)`

Predicts 3D facial landmarks from the provided mesh data using multi-view projection and raycasting. Detects 478 facial landmarks in 3D space.

**Parameters:**
- `meshes` (dict): Mesh data object returned from `import_mesh_and_setup()`
- `projections` (int, optional): Number of camera projections to use for landmark detection. Default is 100. More projections increase accuracy but take longer.

**Returns:**
- `dict`: Dictionary containing:
  - `"facemarks_3d"` (list): List of 3D coordinates for detected facial landmarks
  - `"closest_vertex_ids"` (list): Indices of the closest mesh vertices to each landmark

**Example:**
```python
result = predict(meshes, projections=150)
landmarks = result["facemarks_3d"]
vertex_ids = result["closest_vertex_ids"]
```

**Note:** The function uses random camera rotations for robust multi-view detection. If fewer than half of the requested projections successfully detect a face, an error message is printed.

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
    "normalized coordinates": [[x, y, z], ...],
    "closest vertex indexes": [idx1, idx2, ...]
}
```

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
render_result(meshes["original"], result["facemarks_3d"])
```

**Note:** This function requires a display environment. It will not work in headless environments with virtual displays.

## How It Works

The package uses a sophisticated multi-view approach to detect 3D facial landmarks:

1. **Multi-View Projection**: The mesh is rendered from multiple random camera angles (default: 100 views)
2. **2D Landmark Detection**: MediaPipe Face Landmark detection is applied to each rendered view
4. **Ray Casting**: Rays are cast from camera positions through detected 2D landmarks onto the 3D mesh
5. **3D Reconstruction**: The intersection points are aggregated across all views to compute robust 3D landmark positions
6. **Vertex Mapping**: Each landmark is mapped to the closest vertex on the original mesh

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

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/[username]/[repo-name]).
