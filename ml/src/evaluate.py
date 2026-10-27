"""Metrics and plots. Every number is computed from real model scores on real rows;
nothing here is hard-coded."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from .config import ATTACK_SCENARIOS  # noqa: E402


def choose_threshold_f1(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    p, r, thr = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def choose_threshold_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float = 0.02) -> float:
    """Highest-recall threshold whose false-positive rate on the given (validation) rows is
    at most `max_fpr`. Avoids the degenerate 'alert on everything' optimum that maximising
    F1 can reach when many positive rows look normal."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, score)
    ok = np.where(fpr <= max_fpr)[0]
    i = ok[np.argmax(tpr[ok])] if len(ok) else 0
    t = thr[i]
    return float(np.nextafter(np.max(score), np.inf)) if not np.isfinite(t) else float(t)


def classification_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict[str, Any]:
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)), "n_positive": int(y.sum()),
        "pr_auc": None, "roc_auc": None,
    }
    if 0 < y.sum() < len(y):
        out["pr_auc"] = float(average_precision_score(y, score))
        out["roc_auc"] = float(roc_auc_score(y, score))
    return out


def per_scenario(df: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    """Row-level recall for attack scenarios, false-positive rate for benign ones."""
    out = {}
    for sc, g in df.assign(_p=pred).groupby("scenario"):
        if sc in ATTACK_SCENARIOS:
            pos = g[g["label"] == 1]
            out[sc] = {"kind": "attack", "rows": int(len(pos)), "recall": float(pos["_p"].mean()) if len(pos) else None}
        else:
            out[sc] = {"kind": "not_spoofing", "rows": int(len(g)), "false_positive_rate": float(g["_p"].mean())}
    return out


def event_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    """Trajectory-level view: was each attack noticed, how late, and how often are
    non-attack flights alarmed? `df` needs trajectory_id, timestamp, label, scenario."""
    d = df.assign(_p=pred.astype(bool))
    per_attack: dict[str, list] = {a: [] for a in ATTACK_SCENARIOS}
    delays: dict[str, list] = {a: [] for a in ATTACK_SCENARIOS}
    clean = {"n": 0, "with_alert": 0, "alerts": 0, "hours": 0.0}
    for _, g in d.groupby("trajectory_id", sort=False):
        sc = g["scenario"].iloc[0]
        if sc in ATTACK_SCENARIOS:
            hit = g[(g["label"] == 1) & g["_p"]]
            per_attack[sc].append(len(hit) > 0)
            if len(hit):
                delays[sc].append(float(hit["timestamp"].iloc[0] - g["attack_onset_ts"].iloc[0]))
        else:
            clean["n"] += 1
            clean["with_alert"] += int(g["_p"].any())
            clean["alerts"] += int(g["_p"].sum())
            clean["hours"] += len(g) / 3600.0
    return {
        "attacks": {a: {"n": len(v), "detection_rate": float(np.mean(v)) if v else None,
                        "median_delay_s": float(np.median(delays[a])) if delays[a] else None}
                    for a, v in per_attack.items()},
        "non_attack_flights": {
            "n": clean["n"],
            "share_with_any_alert": clean["with_alert"] / max(clean["n"], 1),
            "alerts_per_hour": clean["alerts"] / max(clean["hours"], 1e-9),
        },
    }


def calibration(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Reliability data, Brier score and expected calibration error (ECE)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(prob[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(conf - acc)
            rows.append({"mean_predicted": conf, "fraction_positive": acc, "count": int(m.sum())})
    return {"brier": float(brier_score_loss(y, prob)), "ece": float(ece), "bins": rows}


# ------------------------------------------------------------------------- plots
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_comparison(results: dict[str, Any], path: Path) -> None:
    sets = list(results)
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fs in zip(axes, sets):
        names = list(results[fs])
        x = np.arange(len(names))
        for k, (metric, off) in enumerate((("precision", -0.27), ("recall", 0.0), ("f1", 0.27))):
            vals = [results[fs][n]["test"][metric] for n in names]
            ax.bar(x + off, vals, 0.26, label=metric)
        ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_title(f"feature set: {fs} (test)")
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    _save(fig, path)


def plot_confusion_matrices(entries: dict[str, Any], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(entries), figsize=(3.2 * len(entries), 3.4))
    for ax, (name, e) in zip(np.atleast_1d(axes), entries.items()):
        c = e["test"]["confusion_matrix"]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] / m[i].sum() > 0.5 else "black")
        ax.set_xticks([0, 1], ["normal", "spoofed"], fontsize=7)
        ax.set_yticks([0, 1], ["normal", "spoofed"], fontsize=7)
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
    fig.suptitle(f"Test confusion matrices ({title})", fontsize=9)
    _save(fig, path)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name.replace("_", " "))
    ax.set(xlabel="recall", ylabel="precision", title=f"Precision-recall, test ({title})", ylim=(0, 1.02))
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_importance(names: list[str], values: np.ndarray, title: str, path: Path, top: int = 15) -> None:
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.barh([names[i] for i in order], values[order])
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def plot_generalization(gen: dict[str, Any], path: Path) -> None:
    sets = list(gen)
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 3.8), sharey=True)
    for ax, fs in zip(np.atleast_1d(axes), sets):
        atk = list(gen[fs])
        x = np.arange(len(atk))
        ax.bar(x - 0.2, [gen[fs][a]["recall_seen_in_training"] for a in atk], 0.38, label="attack type seen in training")
        ax.bar(x + 0.2, [gen[fs][a]["recall_held_out"] for a in atk], 0.38, label="attack type held out")
        ax.set_xticks(x, [a.replace("_", "\n") for a in atk], fontsize=8)
        ax.set_title(f"Random forest, feature set: {fs}", fontsize=9)
        ax.set_ylim(0, 1.05)
    np.atleast_1d(axes)[0].set_ylabel("row-level recall (test)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, path)


def plot_example_timeline(df: pd.DataFrame, prob: np.ndarray, thr: float, rule_alert: np.ndarray, title: str, path: Path) -> None:
    """Path, speed residual and model probability for one test trajectory."""
    g = df.reset_index(drop=True)
    t = g["timestamp"] - g["timestamp"].iloc[0]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    lab = g["label"].to_numpy() == 1
    ax[0].plot(g["lon"][~lab], g["lat"][~lab], ".", ms=3, label="label: not spoofed")
    ax[0].plot(g["lon"][lab], g["lat"][lab], ".", ms=3, color="tab:red", label="label: spoofed")
    ax[0].set(title="Reported track", xlabel="lon", ylabel="lat")
    ax[0].legend(fontsize=7)
    ax[1].plot(t, prob, lw=1)
    ax[1].axhline(thr, color="k", ls="--", lw=0.8, label=f"threshold {thr:.2f}")
    ax[1].fill_between(t, 0, 1, where=lab, color="tab:red", alpha=0.12, label="true attack window")
    ax[1].set(title="Model score over time", xlabel="seconds", ylim=(0, 1.02))
    ax[1].legend(fontsize=7)
    ax[2].step(t, rule_alert.astype(int), where="post")
    ax[2].fill_between(t, 0, 1, where=lab, color="tab:red", alpha=0.12)
    ax[2].set(title="Rule-based alert (separate from ML)", xlabel="seconds", yticks=[0, 1])
    fig.suptitle(title, fontsize=9)
    _save(fig, path)
