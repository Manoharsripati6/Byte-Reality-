"""
Flask backend for 3D SVM Decision Boundary VR Visualization.
Handles dataset upload, SVM training, normalization, and JSON generation.
"""

import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.svm import SVC
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["STATIC_FOLDER"] = os.path.join(os.path.dirname(__file__), "static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["STATIC_FOLDER"], exist_ok=True)


def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Center data at origin and uniformly scale so all points lie within [-2.5, +2.5].
    """
    # Center at origin
    centroid = X.mean(axis=0)
    X_centered = X - centroid

    # Uniform scaling: find the max absolute value across ALL axes
    max_abs = np.max(np.abs(X_centered))
    if max_abs < 1e-12:
        return X_centered  # all points are the same

    # Scale to fit within [-2.5, 2.5]
    scale = 2.5 / max_abs
    X_normalized = X_centered * scale

    return X_normalized


def generate_plane_points(w, b, grid_range=2.5, grid_steps=30):
    """
    Generate mesh grid points for the SVM decision boundary plane.
    Equation: w[0]*x + w[1]*y + w[2]*z + b = 0

    We solve for the axis with the largest absolute weight to avoid
    division by near-zero values.
    """
    w = np.array(w, dtype=float)
    abs_w = np.abs(w)
    dominant_axis = np.argmax(abs_w)

    # If all weights are near zero, no valid plane
    if abs_w[dominant_axis] < 1e-10:
        return []

    # Create grid over the two non-dominant axes
    axes = [0, 1, 2]
    axes.remove(dominant_axis)
    ax1, ax2 = axes

    vals = np.linspace(-grid_range, grid_range, grid_steps)
    grid1, grid2 = np.meshgrid(vals, vals)

    # Solve for the dominant axis: w[dom]*x_dom = -b - w[ax1]*x_ax1 - w[ax2]*x_ax2
    grid_dom = (-b - w[ax1] * grid1 - w[ax2] * grid2) / w[dominant_axis]

    # Filter points outside the bounding cube
    mask = np.abs(grid_dom) <= grid_range

    plane_points = []
    for i in range(grid_steps):
        for j in range(grid_steps):
            if mask[i, j]:
                pt = [0.0, 0.0, 0.0]
                pt[dominant_axis] = float(grid_dom[i, j])
                pt[ax1] = float(grid1[i, j])
                pt[ax2] = float(grid2[i, j])
                # Apply axis mapping: x=X[0], y=X[2](height), z=X[1]
                plane_points.append({
                    "x": pt[0],
                    "y": pt[2],
                    "z": pt[1],
                })

    return plane_points


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """Handle dataset upload, train SVM, generate visualization JSONs."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".json"):
        return jsonify({"error": "Only JSON files are accepted"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        with open(filepath, "r") as f:
            dataset = json.load(f)

        # Validate dataset structure
        if "X" not in dataset or "y" not in dataset:
            return jsonify({"error": "Dataset must contain 'X' and 'y' keys"}), 400

        X = np.array(dataset["X"], dtype=float)
        y = np.array(dataset["y"], dtype=int)

        if X.ndim != 2 or X.shape[1] != 3:
            return jsonify({"error": "X must be an array of 3D points"}), 400
        if len(X) != len(y):
            return jsonify({"error": "X and y must have the same length"}), 400
        if len(np.unique(y)) < 2:
            return jsonify({"error": "Dataset must contain at least 2 classes"}), 400

        # Normalize data
        X_norm = normalize_data(X)

        # Train linear SVM
        svm = SVC(kernel="linear")
        svm.fit(X_norm, y)

        w = svm.coef_[0]  # shape (3,)
        b = svm.intercept_[0]  # scalar

        # Generate points.json with axis mapping: x=X[0], y=X[2](height), z=X[1]
        points_data = {
            "points": [
                {
                    "x": float(X_norm[i, 0]),
                    "y": float(X_norm[i, 2]),  # height
                    "z": float(X_norm[i, 1]),
                    "label": int(y[i]),
                }
                for i in range(len(X_norm))
            ]
        }

        # Generate plane.json
        plane_points = generate_plane_points(w, b)
        plane_data = {"plane": plane_points}

        # Write output JSONs to static folder
        points_path = os.path.join(app.config["STATIC_FOLDER"], "points.json")
        plane_path = os.path.join(app.config["STATIC_FOLDER"], "plane.json")

        with open(points_path, "w") as f:
            json.dump(points_data, f)
        with open(plane_path, "w") as f:
            json.dump(plane_data, f)

        return jsonify({
            "success": True,
            "num_points": len(X_norm),
            "classes": sorted(int(c) for c in np.unique(y)),
            "svm_weights": w.tolist(),
            "svm_bias": float(b),
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
