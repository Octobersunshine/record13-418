import io
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)


def _beeswarm_offsets(values, width=0.6):
    if len(values) == 0:
        return np.array([])

    sorted_idx = np.argsort(values)
    offsets = np.zeros(len(values))

    placed = []
    for idx in sorted_idx:
        v = values[idx]
        best_offset = 0.0
        min_overlap = float("inf")

        candidates = np.linspace(-width / 2, width / 2, 80)
        for off in candidates:
            overlap = 0.0
            for p_idx, p_off in placed:
                dy = abs(v - values[p_idx])
                dx = abs(off - p_off)
                if dx < 0.05 and dy < 0.05:
                    overlap += 1.0
            if overlap < min_overlap:
                min_overlap = overlap
                best_offset = off

        offsets[idx] = best_offset
        placed.append((idx, best_offset))

    return offsets


def _simple_beeswarm(values, width=0.6):
    if len(values) == 0:
        return np.array([])

    y = np.array(values, dtype=float)
    n = len(y)
    sorted_idx = np.argsort(y)
    x_offsets = np.zeros(n)

    placed_y = np.empty(0)
    placed_x = np.empty(0)

    y_range = y.max() - y.min()
    if y_range == 0:
        point_radius = 0.5
    else:
        point_radius = y_range * 0.025
        if point_radius < 0.1:
            point_radius = 0.1

    min_dist = point_radius * 2.0

    n_candidates = min(80, max(30, n // 2 + 20))

    for idx in sorted_idx:
        yi = y[idx]
        best_x = 0.0

        candidates = np.linspace(-width / 2, width / 2, n_candidates)
        candidates = np.concatenate([[0.0], candidates])
        candidates = np.unique(candidates)

        best_dist = -1.0
        for candidate_x in candidates:
            if len(placed_y) == 0:
                best_x = candidate_x
                break

            dx = candidate_x - placed_x
            dy = yi - placed_y
            dists = np.sqrt(dx * dx + dy * dy)
            min_dist_to_others = dists.min()

            if min_dist_to_others >= min_dist:
                center_dist = abs(candidate_x)
                if best_dist < 0 or center_dist < best_dist:
                    best_dist = center_dist
                    best_x = candidate_x

        if best_dist < 0:
            candidates_2 = np.linspace(-width / 2, width / 2, n_candidates * 2)
            best_overlap = float("inf")
            for candidate_x in candidates_2:
                dx = candidate_x - placed_x
                dy = yi - placed_y
                dists = np.sqrt(dx * dx + dy * dy)
                overlap = np.sum(np.maximum(0, min_dist - dists))
                if overlap < best_overlap:
                    best_overlap = overlap
                    best_x = candidate_x

        x_offsets[idx] = best_x
        placed_y = np.append(placed_y, yi)
        placed_x = np.append(placed_x, best_x)

    return x_offsets


def generate_beeswarm(categories, values, title="Beeswarm Plot",
                       xlabel="Category", ylabel="Value",
                       palette=None, figsize=(10, 6), dpi=120):
    df = pd.DataFrame({"category": categories, "value": values})
    groups = df.groupby("category", sort=True)

    cat_labels = list(groups.groups.keys())
    n_cats = len(cat_labels)

    if palette is None:
        cmap = plt.cm.Set2
        palette = [cmap(i / max(n_cats, 1)) for i in range(n_cats)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, cat in enumerate(cat_labels):
        grp = groups.get_group(cat)
        y_vals = grp["value"].values
        x_center = i

        x_jitter = _simple_beeswarm(y_vals, width=0.7)
        x_positions = x_center + x_jitter

        ax.scatter(x_positions, y_vals, c=[palette[i % len(palette)]],
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.5,
                   zorder=3)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(cat_labels, rotation=30, ha="right")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/plot", methods=["POST"])
def plot():
    data = request.get_json(force=True)
    categories = data.get("categories", [])
    values = data.get("values", [])
    title = data.get("title", "Beeswarm Plot")
    xlabel = data.get("xlabel", "Category")
    ylabel = data.get("ylabel", "Value")

    if not categories or not values:
        return jsonify({"error": "categories and values are required"}), 400
    if len(categories) != len(values):
        return jsonify({"error": "categories and values must have the same length"}), 400

    try:
        values = [float(v) for v in values]
    except (ValueError, TypeError):
        return jsonify({"error": "values must be numeric"}), 400

    buf = generate_beeswarm(categories, values, title=title,
                            xlabel=xlabel, ylabel=ylabel)
    return send_file(buf, mimetype="image/png")


@app.route("/api/plot/csv", methods=["POST"])
def plot_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

    if df.shape[1] < 2:
        return jsonify({"error": "CSV must have at least 2 columns (category, value)"}), 400

    title = request.form.get("title", "Beeswarm Plot")
    xlabel = request.form.get("xlabel", df.columns[0])
    ylabel = request.form.get("ylabel", df.columns[1])

    cat_col = df.columns[0]
    val_col = df.columns[1]

    df = df.dropna(subset=[cat_col, val_col])
    try:
        df[val_col] = pd.to_numeric(df[val_col])
    except Exception:
        return jsonify({"error": f"Column '{val_col}' must be numeric"}), 400

    buf = generate_beeswarm(df[cat_col].tolist(), df[val_col].tolist(),
                            title=title, xlabel=xlabel, ylabel=ylabel)
    return send_file(buf, mimetype="image/png")


@app.route("/api/demo", methods=["GET"])
def demo():
    np.random.seed(42)
    cats = ["A", "B", "C", "D", "E"]
    categories = []
    values = []
    for c in cats:
        n = np.random.randint(25, 50)
        categories.extend([c] * n)
        values.extend(np.random.normal(loc=ord(c), scale=3 + np.random.rand() * 5, size=n).tolist())

    title = request.args.get("title", "Beeswarm Demo")
    xlabel = request.args.get("xlabel", "Group")
    ylabel = request.args.get("ylabel", "Measurement")

    buf = generate_beeswarm(categories, values, title=title,
                            xlabel=xlabel, ylabel=ylabel)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
