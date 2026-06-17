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


def _compute_auto_width(values, n_points):
    if n_points <= 1:
        return 0.4
    y = np.array(values, dtype=float)
    y_range = y.max() - y.min()
    if y_range == 0:
        y_range = 1.0

    points_per_unit = n_points / y_range if y_range > 0 else n_points
    base_width = 0.4

    if n_points <= 10:
        width = base_width
    elif n_points <= 30:
        width = 0.6
    elif n_points <= 60:
        width = 0.8
    elif n_points <= 100:
        width = 1.1
    elif n_points <= 200:
        width = 1.5
    else:
        width = 2.0 + (n_points - 200) * 0.003

    if points_per_unit > 50:
        width *= 1.3
    elif points_per_unit > 20:
        width *= 1.1

    return min(width, 3.5)


def _simple_beeswarm(values, width=None):
    if len(values) == 0:
        return np.array([])

    y = np.array(values, dtype=float)
    n = len(y)
    sorted_idx = np.argsort(y)
    x_offsets = np.zeros(n)

    if width is None:
        width = _compute_auto_width(values, n)

    y_range = y.max() - y.min()
    if y_range == 0:
        point_radius_y = 0.02
        point_radius_x = width * 0.04
    else:
        density_factor = n / y_range if y_range > 0 else n
        base_radius_y = y_range * 0.012
        point_radius_y = max(base_radius_y, y_range / max(n * 1.5, 10))
        point_radius_x = width * 0.04

    min_dist_y = point_radius_y * 2.2
    min_dist_x = point_radius_x * 2.2

    placed_y = []
    placed_x = []

    for idx in sorted_idx:
        yi = y[idx]

        best_x = 0.0
        best_score = float("inf")

        n_candidates = min(120, max(40, n * 2))
        candidates = np.linspace(-width / 2, width / 2, n_candidates)

        half_n = n_candidates // 2
        order = np.concatenate([
            [0],
            np.arange(1, half_n + 1).repeat(2) * np.array([1, -1] * half_n)
        ])
        order = np.clip(order, -half_n, half_n)
        ordered_candidates = []
        seen = set()
        for o in order:
            idx_c = half_n + o
            if 0 <= idx_c < n_candidates and idx_c not in seen:
                ordered_candidates.append(candidates[idx_c])
                seen.add(idx_c)
        for c in candidates:
            if c not in seen:
                ordered_candidates.append(c)

        for candidate_x in ordered_candidates:
            if len(placed_y) == 0:
                best_x = candidate_x
                best_score = 0
                break

            py = np.array(placed_y)
            px = np.array(placed_x)

            dy = np.abs(yi - py)
            dx = np.abs(candidate_x - px)

            y_overlap = dy < min_dist_y
            x_overlap = dx < min_dist_x
            overlap_mask = y_overlap & x_overlap

            if not np.any(overlap_mask):
                dist_sq = dy * dy + dx * dx
                closest = np.sqrt(dist_sq).min()
                center_penalty = abs(candidate_x) * 0.01
                score = -closest + center_penalty
                if score < best_score:
                    best_score = score
                    best_x = candidate_x

        if best_score == float("inf"):
            fine_candidates = np.linspace(-width / 2, width / 2, n_candidates * 2)
            best_overlap = float("inf")
            for candidate_x in fine_candidates:
                py = np.array(placed_y)
                px = np.array(placed_x)
                dy = np.abs(yi - py)
                dx = np.abs(candidate_x - px)
                y_pen = np.maximum(0, min_dist_y - dy)
                x_pen = np.maximum(0, min_dist_x - dx)
                total_overlap = np.sum(y_pen * x_pen * 100 + y_pen + x_pen)
                total_overlap += abs(candidate_x) * 0.001
                if total_overlap < best_overlap:
                    best_overlap = total_overlap
                    best_x = candidate_x

        placed_y.append(yi)
        placed_x.append(best_x)
        x_offsets[idx] = best_x

    x_arr = np.array(placed_x)
    y_arr = np.array(placed_y)
    for _ in range(3):
        for i in range(len(x_arr)):
            if i == 0:
                continue
            others_x = np.delete(x_arr, i)
            others_y = np.delete(y_arr, i)
            yi = y_arr[i]

            dy = np.abs(yi - others_y)
            dx = np.abs(x_arr[i] - others_x)
            y_overlap = dy < min_dist_y
            x_overlap = dx < min_dist_x

            if np.any(y_overlap & x_overlap):
                current_x = x_arr[i]
                best_candidate = current_x
                best_penalty = float("inf")

                perturb = np.linspace(-width * 0.15, width * 0.15, 21)
                for p in perturb:
                    cx = current_x + p
                    if abs(cx) > width / 2:
                        continue
                    dx2 = np.abs(cx - others_x)
                    x_overlap2 = dx2 < min_dist_x
                    overlap_mask2 = y_overlap & x_overlap2
                    if np.any(overlap_mask2):
                        y_pen = np.maximum(0, min_dist_y - dy[overlap_mask2])
                        x_pen = np.maximum(0, min_dist_x - dx2[overlap_mask2])
                        pen = np.sum(y_pen + x_pen) + abs(cx) * 0.001
                    else:
                        pen = abs(cx) * 0.001
                    if pen < best_penalty:
                        best_penalty = pen
                        best_candidate = cx
                x_arr[i] = best_candidate

    for k, idx in enumerate(sorted_idx):
        x_offsets[idx] = x_arr[k]

    return x_offsets


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _darken_color(color, factor=0.6):
    if isinstance(color, str) and color.startswith("#"):
        rgba = _hex_to_rgba(color, 1.0)
    else:
        rgba = color if len(color) == 4 else (*color, 1.0)
    return (
        min(max(rgba[0] * factor, 0), 1),
        min(max(rgba[1] * factor, 0), 1),
        min(max(rgba[2] * factor, 0), 1),
        rgba[3]
    )


def generate_beeswarm(categories, values, title="Beeswarm Plot",
                       xlabel="Category", ylabel="Value",
                       palette=None, figsize=None, dpi=120,
                       show_boxplot=False):
    df = pd.DataFrame({"category": categories, "value": values})
    groups = df.groupby("category", sort=True)

    cat_labels = list(groups.groups.keys())
    n_cats = len(cat_labels)
    max_group_size = max([len(groups.get_group(c)) for c in cat_labels]) if cat_labels else 0

    if figsize is None:
        if n_cats <= 3:
            base_w = 8
        elif n_cats <= 6:
            base_w = 10
        else:
            base_w = 12 + (n_cats - 6) * 0.6
        if max_group_size > 150:
            base_w *= 1.25
        elif max_group_size > 80:
            base_w *= 1.1
        figsize = (base_w, 6)

    if palette is None:
        cmap = plt.cm.Set2
        palette = [cmap(i / max(n_cats, 1)) for i in range(n_cats)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if show_boxplot and n_cats > 0:
        data_by_cat = [groups.get_group(cat)["value"].values for cat in cat_labels]
        box_width = 0.35

        bp = ax.boxplot(
            data_by_cat,
            positions=range(n_cats),
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            zorder=2,
            medianprops={"linewidth": 2, "zorder": 4},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            boxprops={"linewidth": 1.5}
        )

        for i, patch in enumerate(bp["boxes"]):
            color = palette[i % len(palette)]
            patch.set_facecolor((*color[:3], 0.22))
            patch.set_edgecolor(_darken_color(color, 0.55))

        for elem_name in ["whiskers", "caps"]:
            for i, line in enumerate(bp[elem_name]):
                color = palette[(i // 2) % len(palette)]
                line.set_color(_darken_color(color, 0.55))

        for i, line in enumerate(bp["medians"]):
            color = palette[i % len(palette)]
            line.set_color(_darken_color(color, 0.35))

    for i, cat in enumerate(cat_labels):
        grp = groups.get_group(cat)
        y_vals = grp["value"].values
        x_center = i

        x_jitter = _simple_beeswarm(y_vals)
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


def _parse_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return default


@app.route("/api/plot", methods=["POST"])
def plot():
    data = request.get_json(force=True)
    categories = data.get("categories", [])
    values = data.get("values", [])
    title = data.get("title", "Beeswarm Plot")
    xlabel = data.get("xlabel", "Category")
    ylabel = data.get("ylabel", "Value")
    show_boxplot = _parse_bool(data.get("show_boxplot", False))

    if not categories or not values:
        return jsonify({"error": "categories and values are required"}), 400
    if len(categories) != len(values):
        return jsonify({"error": "categories and values must have the same length"}), 400

    try:
        values = [float(v) for v in values]
    except (ValueError, TypeError):
        return jsonify({"error": "values must be numeric"}), 400

    buf = generate_beeswarm(categories, values, title=title,
                            xlabel=xlabel, ylabel=ylabel,
                            show_boxplot=show_boxplot)
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
    show_boxplot = _parse_bool(request.form.get("show_boxplot", False))

    cat_col = df.columns[0]
    val_col = df.columns[1]

    df = df.dropna(subset=[cat_col, val_col])
    try:
        df[val_col] = pd.to_numeric(df[val_col])
    except Exception:
        return jsonify({"error": f"Column '{val_col}' must be numeric"}), 400

    buf = generate_beeswarm(df[cat_col].tolist(), df[val_col].tolist(),
                            title=title, xlabel=xlabel, ylabel=ylabel,
                            show_boxplot=show_boxplot)
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
    show_boxplot = _parse_bool(request.args.get("show_boxplot", True), True)

    buf = generate_beeswarm(categories, values, title=title,
                            xlabel=xlabel, ylabel=ylabel,
                            show_boxplot=show_boxplot)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
