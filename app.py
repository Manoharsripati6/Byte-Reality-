"""
Flask backend for 3D ML Decision Boundary VR Visualization.
Supports SVM, Linear Regression, and Neural Network models.
"""

import json
import os
import warnings

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
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
    centroid = X.mean(axis=0)
    X_centered = X - centroid

    max_abs = np.max(np.abs(X_centered))
    if max_abs < 1e-12:
        return X_centered

    scale = 2.5 / max_abs
    return X_centered * scale


def generate_plane_points(w, b, grid_range=2.5, grid_steps=30):
    """
    Generate mesh grid points for a linear decision boundary.
    Equation: w[0]*x + w[1]*y + w[2]*z + b = 0
    Solves for the axis with the largest weight to avoid division by near-zero.
    """
    w = np.array(w, dtype=float)
    abs_w = np.abs(w)
    dominant_axis = np.argmax(abs_w)

    if abs_w[dominant_axis] < 1e-10:
        return []

    axes = [0, 1, 2]
    axes.remove(dominant_axis)
    ax1, ax2 = axes

    vals = np.linspace(-grid_range, grid_range, grid_steps)
    grid1, grid2 = np.meshgrid(vals, vals)

    grid_dom = (-b - w[ax1] * grid1 - w[ax2] * grid2) / w[dominant_axis]
    mask = np.abs(grid_dom) <= grid_range

    plane_points = []
    for i in range(grid_steps):
        for j in range(grid_steps):
            if mask[i, j]:
                pt = [0.0, 0.0, 0.0]
                pt[dominant_axis] = float(grid_dom[i, j])
                pt[ax1] = float(grid1[i, j])
                pt[ax2] = float(grid2[i, j])
                # Axis mapping: x=X[0], y=X[2](height), z=X[1]
                plane_points.append({
                    "x": pt[0],
                    "y": pt[2],
                    "z": pt[1],
                })

    return plane_points


def generate_neural_net_boundary(model, grid_range=2.5, grid_steps=25):
    """
    Generate decision boundary points for a neural network by sampling a 3D grid
    and finding points near the decision boundary (where prediction changes).
    """
    vals = np.linspace(-grid_range, grid_range, grid_steps)
    xx, yy, zz = np.meshgrid(vals, vals, vals)
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Get decision function or probabilities
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(grid_points)
    else:
        proba = model.predict_proba(grid_points)
        scores = proba[:, 1] - proba[:, 0]

    # Find points near the boundary (score close to 0)
    threshold = np.max(np.abs(scores)) * 0.08
    if threshold < 1e-6:
        threshold = 0.5
    boundary_mask = np.abs(scores) < threshold

    boundary_pts = grid_points[boundary_mask]

    # Axis mapping: x=X[0], y=X[2](height), z=X[1]
    return [
        {"x": float(pt[0]), "y": float(pt[2]), "z": float(pt[1])}
        for pt in boundary_pts
    ]


def generate_regression_plane(model, grid_range=2.5, grid_steps=30):
    """
    Generate the regression surface for a linear regression model.
    For 3D input predicting y, the surface is z_predicted = w[0]*x + w[1]*y + w[2]*z + b.
    We use features X[0] and X[1] as the grid, and the predicted value as the third axis.
    """
    w = model.coef_
    b = model.intercept_

    vals = np.linspace(-grid_range, grid_range, grid_steps)
    grid_x0, grid_x1 = np.meshgrid(vals, vals)

    # For each (x0, x1) pair, we need to find x2 such that the prediction
    # equals the boundary (0.5 for binary classification as regression)
    # prediction = w[0]*x0 + w[1]*x1 + w[2]*x2 + b = 0.5
    if abs(w[2]) < 1e-10:
        # Fall back: solve for x1 instead
        if abs(w[1]) < 1e-10:
            return []
        grid_x0_flat, grid_x2 = np.meshgrid(vals, vals)
        grid_x1_solved = (0.5 - b - w[0] * grid_x0_flat - w[2] * grid_x2) / w[1]
        mask = np.abs(grid_x1_solved) <= grid_range

        points = []
        for i in range(grid_steps):
            for j in range(grid_steps):
                if mask[i, j]:
                    points.append({
                        "x": float(grid_x0_flat[i, j]),
                        "y": float(grid_x2[i, j]),
                        "z": float(grid_x1_solved[i, j]),
                    })
        return points

    grid_x2 = (0.5 - b - w[0] * grid_x0 - w[1] * grid_x1) / w[2]
    mask = np.abs(grid_x2) <= grid_range

    points = []
    for i in range(grid_steps):
        for j in range(grid_steps):
            if mask[i, j]:
                # Axis mapping: x=X[0], y=X[2](height), z=X[1]
                points.append({
                    "x": float(grid_x0[i, j]),
                    "y": float(grid_x2[i, j]),
                    "z": float(grid_x1[i, j]),
                })

    return points


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """Handle dataset upload, train selected model, generate visualization JSONs."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".json"):
        return jsonify({"error": "Only JSON files are accepted"}), 400

    model_type = request.form.get("model", "svm")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        with open(filepath, "r") as f:
            dataset = json.load(f)

        if "X" not in dataset or "y" not in dataset:
            return jsonify({"error": "Dataset must contain 'X' and 'y' keys"}), 400

        X = np.array(dataset["X"], dtype=float)
        y = np.array(dataset["y"], dtype=float)

        if X.ndim != 2 or X.shape[1] != 3:
            return jsonify({"error": "X must be an array of 3D points"}), 400
        if len(X) != len(y):
            return jsonify({"error": "X and y must have the same length"}), 400

        # Normalize data
        X_norm = normalize_data(X)

        # Train model based on selection
        model_info = {"model_type": model_type}

        if model_type == "svm":
            if len(np.unique(y.astype(int))) < 2:
                return jsonify({"error": "SVM requires at least 2 classes"}), 400

            model = SVC(kernel="linear")
            model.fit(X_norm, y.astype(int))

            w = model.coef_[0]
            b = model.intercept_[0]
            accuracy = model.score(X_norm, y.astype(int))

            plane_points = generate_plane_points(w, b)
            model_info.update({
                "weights": w.tolist(),
                "bias": float(b),
                "accuracy": round(float(accuracy) * 100, 1),
            })

        elif model_type == "linear_regression":
            model = LinearRegression()
            model.fit(X_norm, y)

            w = model.coef_
            b = model.intercept_
            r2 = model.score(X_norm, y)

            plane_points = generate_regression_plane(model)
            model_info.update({
                "weights": w.tolist(),
                "bias": float(b),
                "r_squared": round(float(r2), 4),
            })

        elif model_type == "neural_network":
            if len(np.unique(y.astype(int))) < 2:
                return jsonify({"error": "Neural Network classifier requires at least 2 classes"}), 400

            model = MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                max_iter=1000,
                random_state=42,
                alpha=0.01,
            )
            model.fit(X_norm, y.astype(int))

            accuracy = model.score(X_norm, y.astype(int))
            plane_points = generate_neural_net_boundary(model)

            # Use actual output size from trained model (binary classification uses 1 output neuron)
            actual_output_size = model.coefs_[-1].shape[1]
            layer_info = [X_norm.shape[1]] + list(model.hidden_layer_sizes) + [actual_output_size]
            model_info.update({
                "layers": layer_info,
                "accuracy": round(float(accuracy) * 100, 1),
                "iterations": model.n_iter_,
            })

            # Export network structure for 3D visualization
            network_data = {
                "layers": layer_info,
                "weights": [],
                "biases": [],
                "activations": [],
            }
            for w_mat in model.coefs_:
                network_data["weights"].append(w_mat.tolist())
            for b_vec in model.intercepts_:
                network_data["biases"].append(b_vec.tolist())

            # Compute per-neuron activation magnitudes using mean of training data
            activations = [X_norm.mean(axis=0).tolist()]
            layer_input = X_norm
            for i, (w_mat, b_vec) in enumerate(zip(model.coefs_, model.intercepts_)):
                layer_output = layer_input @ w_mat + b_vec
                if i < len(model.coefs_) - 1:
                    layer_output = np.maximum(0, layer_output)  # ReLU
                activations.append(layer_output.mean(axis=0).tolist())
                layer_input = layer_output
            network_data["activations"] = activations

            network_path = os.path.join(app.config["STATIC_FOLDER"], "network.json")
            with open(network_path, "w") as f:
                json.dump(network_data, f)

        else:
            return jsonify({"error": f"Unknown model type: {model_type}"}), 400

        # Generate points.json with axis mapping: x=X[0], y=X[2](height), z=X[1]
        y_int = y.astype(int)
        points_data = {
            "points": [
                {
                    "x": float(X_norm[i, 0]),
                    "y": float(X_norm[i, 2]),
                    "z": float(X_norm[i, 1]),
                    "label": int(y_int[i]),
                }
                for i in range(len(X_norm))
            ]
        }

        plane_data = {"plane": plane_points}

        # Write output JSONs
        points_path = os.path.join(app.config["STATIC_FOLDER"], "points.json")
        plane_path = os.path.join(app.config["STATIC_FOLDER"], "plane.json")

        with open(points_path, "w") as f:
            json.dump(points_data, f)
        with open(plane_path, "w") as f:
            json.dump(plane_data, f)

        return jsonify({
            "success": True,
            "num_points": len(X_norm),
            "num_boundary_points": len(plane_points),
            "classes": sorted(int(c) for c in np.unique(y_int)),
            "model_info": model_info,
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train-animate", methods=["POST"])
def train_animate():
    """Train NN epoch-by-epoch, returning snapshots for animation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".json"):
        return jsonify({"error": "Only JSON files are accepted"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        with open(filepath, "r") as f:
            dataset = json.load(f)

        X = np.array(dataset["X"], dtype=float)
        y = np.array(dataset["y"], dtype=int)

        if X.ndim != 2 or X.shape[1] != 3:
            return jsonify({"error": "X must be an array of 3D points"}), 400
        if len(X) != len(y) or len(np.unique(y)) < 2:
            return jsonify({"error": "Invalid dataset"}), 400

        X_norm = normalize_data(X)

        num_epochs = int(request.form.get("epochs", 50))
        snapshot_interval = max(1, num_epochs // 25)  # ~25 snapshots max

        snapshots = []

        model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            max_iter=1,
            random_state=42,
            alpha=0.01,
            warm_start=True,
        )

        for epoch in range(1, num_epochs + 1):
            model.max_iter = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_norm, y)

            if epoch == 1 or epoch % snapshot_interval == 0 or epoch == num_epochs:
                accuracy = model.score(X_norm, y)
                loss = float(model.loss_)

                # Extract weights for network visualization
                net_weights = [w.tolist() for w in model.coefs_]
                net_biases = [b.tolist() for b in model.intercepts_]

                # Compute activations
                acts = [X_norm.mean(axis=0).tolist()]
                layer_in = X_norm
                for i, (w_mat, b_vec) in enumerate(
                    zip(model.coefs_, model.intercepts_)
                ):
                    layer_out = layer_in @ w_mat + b_vec
                    if i < len(model.coefs_) - 1:
                        layer_out = np.maximum(0, layer_out)
                    acts.append(layer_out.mean(axis=0).tolist())
                    layer_in = layer_out

                # Generate boundary (lower resolution for speed)
                boundary = generate_neural_net_boundary(
                    model, grid_range=2.5, grid_steps=15
                )

                actual_output_size = model.coefs_[-1].shape[1]
                layer_info = (
                    [X_norm.shape[1]]
                    + list(model.hidden_layer_sizes)
                    + [actual_output_size]
                )

                snapshots.append({
                    "epoch": epoch,
                    "accuracy": round(float(accuracy) * 100, 1),
                    "loss": round(loss, 5),
                    "layers": layer_info,
                    "weights": net_weights,
                    "biases": net_biases,
                    "activations": acts,
                    "boundary": boundary,
                })

        # Points data
        points_data = [
            {
                "x": float(X_norm[i, 0]),
                "y": float(X_norm[i, 2]),
                "z": float(X_norm[i, 1]),
                "label": int(y[i]),
            }
            for i in range(len(X_norm))
        ]

        return jsonify({
            "success": True,
            "points": points_data,
            "snapshots": snapshots,
            "total_epochs": num_epochs,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
