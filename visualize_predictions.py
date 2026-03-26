import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR


def collect_prediction_csvs(results_root):
    """Recursively collect prediction csv paths under results root."""
    csv_paths = []
    for root, _, files in os.walk(results_root):
        if "predictions.csv" in files:
            csv_paths.append(os.path.join(root, "predictions.csv"))
    return sorted(csv_paths)


def load_meta(run_dir):
    """Load run_meta.csv if exists."""
    meta_path = os.path.join(run_dir, "run_meta.csv")
    if not os.path.exists(meta_path):
        return {}
    try:
        meta_df = pd.read_csv(meta_path)
        if len(meta_df) == 0:
            return {}
        return dict(meta_df.iloc[0])
    except Exception:
        return {}


def build_patient_stats(df):
    """Aggregate scan-level rows into patient-level rows."""
    if "patient" not in df.columns:
        raise ValueError("Missing column: patient")

    mode = "regression"
    pred_label_col = None

    # Prefer group-only outputs when available.
    if {"group_target", "group_prob"}.issubset(df.columns):
        target_col = "group_target"
        pred_col = "group_prob"
        mode = "classification"
        if "group_pred" in df.columns:
            pred_label_col = "group_pred"
    elif {"ratio_target", "ratio_pred"}.issubset(df.columns):
        target_col = "ratio_target"
        pred_col = "ratio_pred"
    elif {"target", "pred"}.issubset(df.columns):
        target_col = "target"
        pred_col = "pred"
    else:
        raise ValueError(
            "Missing target/pred columns. Need either "
            "['group_target', 'group_prob'] / ['ratio_target', 'ratio_pred'] "
            "or legacy ['target', 'pred']."
        )

    unique_patients = df["patient"].astype(str).unique()
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}

    df = df.copy()
    df["patient"] = df["patient"].astype(str)
    df["patient_idx"] = df["patient"].map(patient_to_idx)

    agg_map = {target_col: "first", pred_col: "mean", "patient_idx": "first"}
    rename_map = {target_col: "target", pred_col: "pred"}
    if pred_label_col is not None:
        agg_map[pred_label_col] = "mean"
        rename_map[pred_label_col] = "pred_label"
    patient_stats = df.groupby("patient").agg(agg_map).reset_index().rename(columns=rename_map).sort_values("patient_idx")
    patient_stats["mode"] = mode
    return patient_stats


def compute_global_y_limits(csv_paths):
    """Compute global y-axis limits from all runs.

    Rule: upper bound is fixed at 1.1, only lower bound is auto-scaled.
    """
    all_values = []
    valid_runs = []

    for csv_path in csv_paths:
        run_dir = os.path.dirname(csv_path)
        try:
            df = pd.read_csv(csv_path)
            patient_stats = build_patient_stats(df)
            values = np.concatenate(
                [
                    patient_stats["target"].to_numpy(dtype=float),
                    patient_stats["pred"].to_numpy(dtype=float),
                ]
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            all_values.append(values)
            valid_runs.append(run_dir)
        except Exception:
            continue

    if not all_values:
        raise ValueError("No valid target/pred values found in any subdirectory.")

    merged = np.concatenate(all_values)
    vmin = float(np.min(merged))
    ymax = 1.1
    if vmin >= ymax:
        return (0.9, ymax), valid_runs

    span_to_top = ymax - vmin
    pad = max(span_to_top * 0.08, 0.01)
    lower = vmin - pad
    return (lower, ymax), valid_runs


def plot_one_run(run_dir, y_limits, overwrite):
    """Generate one plot in the run directory."""
    csv_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        return None

    output_path = os.path.join(run_dir, "prediction_plot.png")
    if os.path.exists(output_path) and not overwrite:
        return output_path

    df = pd.read_csv(csv_path)
    patient_stats = build_patient_stats(df)
    meta = load_meta(run_dir)

    train_n = meta.get("train_n", "NA")
    test_n = len(patient_stats)
    prompt_template = meta.get("prompt_template", "NA")
    scan_handling = meta.get("scan_handling", "NA")
    lr = meta.get("lr", "NA")
    epochs = meta.get("epochs", "NA")

    plt.figure(figsize=(14, 7))
    x = np.arange(len(patient_stats))

    target_vals = patient_stats["target"].to_numpy(dtype=float)
    pred_vals = patient_stats["pred"].to_numpy(dtype=float)
    mode = str(patient_stats["mode"].iloc[0]) if "mode" in patient_stats.columns else "regression"

    if mode == "classification":
        target_is_one = target_vals >= 0.5
        if "pred_label" in patient_stats.columns:
            pred_is_one = patient_stats["pred_label"].to_numpy(dtype=float) >= 0.5
        else:
            pred_is_one = pred_vals >= 0.5
    else:
        eps = 1e-6
        target_is_one = target_vals >= (1.0 - eps)
        pred_is_one = pred_vals >= (1.0 - eps)
    target_lt_one = ~target_is_one
    pred_lt_one = ~pred_is_one

    # Draw target first, then prediction to keep prediction visible when coordinates overlap.
    plt.scatter(
        x[target_is_one],
        target_vals[target_is_one],
        color="#3399FF",
        s=180,
        zorder=5,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
    )
    plt.scatter(
        x[target_lt_one],
        target_vals[target_lt_one],
        color="#3399FF",
        s=130,
        zorder=5,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
    )

    plt.scatter(
        x[pred_is_one],
        pred_vals[pred_is_one],
        color="#FFCC99",
        s=95,
        zorder=6,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
    )
    plt.scatter(
        x[pred_lt_one],
        pred_vals[pred_lt_one],
        color="#FFCC99",
        s=80,
        zorder=6,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
    )

    for _, row in patient_stats.iterrows():
        plt.plot(
            [row["patient_idx"], row["patient_idx"]],
            [row["target"], row["pred"]],
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

    plt.xlabel("Patient ID", fontsize=12)
    if mode == "classification":
        plt.ylabel("Necrosis Group (target) / Probability (prediction)", fontsize=12)
        plt.title("HCC Necrosis Group: Target vs Prediction", fontsize=14)
    else:
        plt.ylabel("Necrosis Ratio", fontsize=12)
        plt.title("HCC Necrosis Ratio: Target vs Prediction", fontsize=14)
    plt.xticks(x, patient_stats["patient"].values, fontsize=8, rotation=45, ha="right")
    plt.ylim(*y_limits)
    plt.yticks(np.linspace(y_limits[0], y_limits[1], 8))
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.9)
    if mode == "classification":
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Target class=1", markerfacecolor="#3399FF", markeredgecolor="black", markersize=9),
            Line2D([0], [0], marker="^", color="w", label="Target class=0", markerfacecolor="#3399FF", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Prediction class=1", markerfacecolor="#FFCC99", markeredgecolor="black", markersize=7),
            Line2D([0], [0], marker="^", color="w", label="Prediction class=0", markerfacecolor="#FFCC99", markeredgecolor="black", markersize=7),
            Line2D([0], [0], color="gray", linestyle="--", label="y=1.0 reference"),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Target (=1)", markerfacecolor="#3399FF", markeredgecolor="black", markersize=9),
            Line2D([0], [0], marker="^", color="w", label="Target (<1)", markerfacecolor="#3399FF", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Prediction (=1)", markerfacecolor="#FFCC99", markeredgecolor="black", markersize=7),
            Line2D([0], [0], marker="^", color="w", label="Prediction (<1)", markerfacecolor="#FFCC99", markeredgecolor="black", markersize=7),
            Line2D([0], [0], color="gray", linestyle="--", label="y=1.0 reference"),
        ]
    plt.legend(handles=legend_handles, loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)

    config_text = (
        f"Train: {train_n} | Test: {test_n} | "
        f"Template: {prompt_template} | Scan: {scan_handling} | "
        f"Epochs: {epochs} | LR: {lr} | "
        f"Y-range: [{y_limits[0]:.4g}, {y_limits[1]:.4g}]"
    )
    plt.figtext(
        0.5,
        -0.03,
        config_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.23)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HCC classification/regression predictions for all subdirectories."
    )
    parser.add_argument(
        "--results-root",
        default=str(Path(ROOT_DIR) / "inference_hcc_regression"),
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction_plot.png if present.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results root not found: {results_root}")

    csv_paths = collect_prediction_csvs(results_root)
    if not csv_paths:
        print(f"未找到 predictions.csv: {results_root}")
        return

    y_limits, valid_runs = compute_global_y_limits(csv_paths)
    print(f"发现 {len(csv_paths)} 个结果目录，开始绘图...")
    print(f"全局纵轴范围: [{y_limits[0]:.6g}, 1.1] (上界固定为1.1)")
    success = 0
    failed = 0

    for run_dir in valid_runs:
        try:
            out = plot_one_run(run_dir, y_limits=y_limits, overwrite=args.overwrite)
            if out is not None:
                print(f"[OK] {out}")
                success += 1
            else:
                print(f"[SKIP] {run_dir}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {run_dir}: {exc}")

    print(f"完成: success={success}, failed={failed}")


if __name__ == "__main__":
    main()
