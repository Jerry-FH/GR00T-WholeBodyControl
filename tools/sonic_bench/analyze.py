"""Compute per-trial metrics and the release-vs-low_latency comparison report.

Reads the results tree produced by run_benchmark.py and writes summary.md
(plus optional PNGs if matplotlib is available). numpy + scipy only.

    .venv_sim/bin/python tools/sonic_bench/analyze.py \
        --results-root gear_sonic_deploy/logs/sonic_bench
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CONTROL_DT, JOINT_GROUPS, OVERRUN_DT_MS, RE_FREQ_TEST  # noqa: E402

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


# --------------------------------------------------------------------------
# per-trial metric computation
# --------------------------------------------------------------------------

def load_csv_matrix(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (meta[N,5], values[N,D]) from a state-logger CSV, or None."""
    if not path.exists():
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0 or data.shape[1] <= 5:
        return None
    return data[:, :5], data[:, 5:]


def quat_geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Angle between quaternion arrays [N,4] (wxyz), in degrees."""
    na = np.linalg.norm(qa, axis=1, keepdims=True)
    nb = np.linalg.norm(qb, axis=1, keepdims=True)
    valid = (na[:, 0] > 1e-8) & (nb[:, 0] > 1e-8)
    dot = np.abs(np.sum((qa / np.maximum(na, 1e-12)) * (qb / np.maximum(nb, 1e-12)), axis=1))
    ang = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
    return np.degrees(np.where(valid, ang, np.nan))


def playing_index_window(trial_dir: Path) -> tuple[float, float] | None:
    """[min,max] state-logger index where motion_playing flag is 1."""
    loaded = load_csv_matrix(trial_dir / "motion_playing.csv")
    if loaded is None:
        return None
    meta, vals = loaded
    playing = vals[:, 0] > 0.5
    if not playing.any():
        return None
    idx = meta[playing, 0]
    return float(idx.min()), float(idx.max())


def trial_metrics(trial_dir: Path) -> dict | None:
    meta_file = trial_dir / "trial_meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text())
    out = {
        "variant": meta["variant"],
        "motion": meta["motion"],
        "status": meta["status"],
        "success": meta["status"] == "success",
    }

    # -- latency from stdout timing samples --------------------------------
    tfile = trial_dir / "timing_samples.json"
    if tfile.exists():
        samples = json.loads(tfile.read_text())
        if samples:
            pol = np.array([s["policy_us"] for s in samples], dtype=float)
            o2m = np.array([s["obs2motor_us"] for s in samples], dtype=float)
            out.update(
                policy_us_mean=pol.mean(), policy_us_p50=np.percentile(pol, 50),
                policy_us_p95=np.percentile(pol, 95), policy_us_max=pol.max(),
                obs2motor_us_mean=o2m.mean(), obs2motor_us_p95=np.percentile(o2m, 95),
                n_timing=len(samples),
            )

    # -- control-loop overruns from q.csv monotonic time -------------------
    loaded = load_csv_matrix(trial_dir / "q.csv")
    if loaded is not None:
        tmono = loaded[0][:, 3]
        dt = np.diff(tmono)
        dt = dt[dt > 0]
        if dt.size:
            out.update(
                overrun_count=int((dt > OVERRUN_DT_MS).sum()),
                overrun_rate=float((dt > OVERRUN_DT_MS).mean()),
                loop_dt_ms_max=float(dt.max()),
            )

    # -- tracking metrics over the motion-playing window -------------------
    stream_file = trial_dir / "stream.npz"
    window = playing_index_window(trial_dir)
    if stream_file.exists() and window is not None:
        z = np.load(stream_file)
        if "index" in z and "body_q_measured" in z and "body_q_target" in z:
            sidx = z["index"]
            lo, hi = window
            mask = (sidx >= lo) & (sidx <= hi)
            if mask.sum() >= 10:
                qm = z["body_q_measured"][mask]
                qt = z["body_q_target"][mask]
                err = qm - qt
                out["n_stream_playing"] = int(mask.sum())
                # gap ratio: PUB HWM can drop frames; CSV remains lossless
                span = sidx[mask].max() - sidx[mask].min() + 1
                out["stream_drop_ratio"] = float(1.0 - mask.sum() / span) if span > 0 else 0.0
                for g, cols in JOINT_GROUPS.items():
                    e = err[:, cols]
                    out[f"rmse_{g}_rad"] = float(np.sqrt(np.mean(e ** 2)))
                    out[f"maxerr_{g}_rad"] = float(np.abs(e).max())
                out["rmse_all_rad"] = float(np.sqrt(np.mean(err ** 2)))
                if "base_quat_measured" in z and "base_quat_target" in z:
                    ang = quat_geodesic_deg(
                        z["base_quat_measured"][mask], z["base_quat_target"][mask]
                    )
                    ang = ang[np.isfinite(ang)]
                    if ang.size:
                        out["base_ori_err_deg_mean"] = float(ang.mean())
                        out["base_ori_err_deg_max"] = float(ang.max())
                if "last_action" in z:
                    act = z["last_action"][mask]
                    d1 = np.diff(act, axis=0)
                    out["daction_rms"] = float(np.sqrt(np.mean(d1 ** 2)))
                    if d1.shape[0] >= 2:
                        d2 = np.diff(d1, axis=0)
                        out["d2action_rms"] = float(np.sqrt(np.mean(d2 ** 2)))
    return out


# --------------------------------------------------------------------------
# aggregation + report
# --------------------------------------------------------------------------

METRIC_COLUMNS = [
    ("policy_us_p50", "Policy p50 (µs)"),
    ("policy_us_p95", "Policy p95 (µs)"),
    ("obs2motor_us_mean", "Obs→Motor mean (µs)"),
    ("overrun_count", "Loop overruns"),
    ("rmse_all_rad", "RMSE all (rad)"),
    ("rmse_legs_rad", "RMSE legs (rad)"),
    ("rmse_waist_rad", "RMSE waist (rad)"),
    ("rmse_arms_rad", "RMSE arms (rad)"),
    ("base_ori_err_deg_mean", "Base ori err (deg)"),
    ("daction_rms", "Δaction RMS"),
]


def agg(vals: list[float]) -> str:
    if not vals:
        return "-"
    a = np.array(vals, dtype=float)
    if a.size == 1:
        return f"{a[0]:.4g}"
    return f"{a.mean():.4g} ± {a.std():.2g}"


def per_motion_means(trials: list[dict], key: str) -> dict[str, float]:
    by_motion: dict[str, list[float]] = {}
    for t in trials:
        if key in t and t["success"]:
            by_motion.setdefault(t["motion"], []).append(t[key])
    return {m: float(np.mean(v)) for m, v in by_motion.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.results_root

    variants = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    trials_by_variant: dict[str, list[dict]] = {}
    for v in variants:
        trials = []
        for tdir in sorted((root / v).glob("*/trial*/")):
            if "_warmup" in str(tdir):
                continue
            m = trial_metrics(tdir)
            if m is not None:
                trials.append(m)
        if trials:
            trials_by_variant[v] = trials
    variants = list(trials_by_variant)
    if not variants:
        sys.exit(f"no analyzable trials under {root}")

    lines = ["# SONIC policy benchmark: " + " vs ".join(variants), ""]
    lines.append(f"Results root: `{root}`")
    lines.append("")

    # freq_test
    for v in variants:
        ft = root / v / "freq_test.txt"
        if ft.exists():
            m = RE_FREQ_TEST.search(ft.read_text(errors="replace"))
            if m:
                lines.append(
                    f"- `{v}` isolated ONNX graph latency (freq_test, **CPU EP** — "
                    f"not TRT; in-loop Policy µs is authoritative): {m.group(1)} µs"
                )
    lines.append("")

    # overview table
    lines.append("## Overview (mean ± std over successful trials)")
    lines.append("")
    header = "| Metric | " + " | ".join(variants) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(variants) + 1))
    n_row = ["| Trials (success/total) |"]
    for v in variants:
        ts = trials_by_variant[v]
        n_row.append(f" {sum(t['success'] for t in ts)}/{len(ts)} |")
    lines.append("".join(n_row))
    sr_row = ["| Success rate |"]
    for v in variants:
        ts = trials_by_variant[v]
        sr_row.append(f" {100.0 * np.mean([t['success'] for t in ts]):.0f}% |")
    lines.append("".join(sr_row))
    for key, label in METRIC_COLUMNS:
        row = [f"| {label} |"]
        for v in variants:
            vals = [t[key] for t in trials_by_variant[v] if key in t and t["success"]]
            row.append(f" {agg(vals)} |")
        lines.append("".join(row))
    lines.append("")

    # failure breakdown
    lines.append("## Trial status breakdown")
    lines.append("")
    for v in variants:
        counts: dict[str, int] = {}
        for t in trials_by_variant[v]:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        lines.append(f"- `{v}`: " + ", ".join(f"{k}={n}" for k, n in sorted(counts.items())))
    lines.append("")

    # paired per-motion comparison (only for exactly 2 variants)
    if len(variants) == 2 and scipy_stats is not None:
        va, vb = variants
        lines.append(f"## Paired per-motion comparison ({va} vs {vb})")
        lines.append("")
        lines.append("Motion means over successful trials; paired t-test + Wilcoxon "
                      "across motions present in both variants. `*` marks p < 0.05.")
        lines.append("")
        lines.append("| Metric | n motions | " + f"{va} mean | {vb} mean | "
                      "t-test p | wilcoxon p |")
        lines.append("|---|---|---|---|---|---|")
        for key, label in METRIC_COLUMNS:
            ma = per_motion_means(trials_by_variant[va], key)
            mb = per_motion_means(trials_by_variant[vb], key)
            common_motions = sorted(set(ma) & set(mb))
            if len(common_motions) < 2:
                continue
            a = np.array([ma[m] for m in common_motions])
            b = np.array([mb[m] for m in common_motions])
            try:
                p_t = scipy_stats.ttest_rel(a, b).pvalue
            except Exception:
                p_t = float("nan")
            try:
                if np.allclose(a, b):
                    p_w = 1.0
                else:
                    p_w = scipy_stats.wilcoxon(a, b).pvalue
            except Exception:
                p_w = float("nan")
            flag = " *" if (np.isfinite(p_t) and p_t < 0.05) else ""
            lines.append(
                f"| {label}{flag} | {len(common_motions)} | {a.mean():.4g} | "
                f"{b.mean():.4g} | {p_t:.3g} | {p_w:.3g} |"
            )
        lines.append("")

    # per-motion detail table
    lines.append("## Per-motion detail (rmse_all_rad, success)")
    lines.append("")
    all_motions = sorted({t["motion"] for ts in trials_by_variant.values() for t in ts})
    lines.append("| Motion | " + " | ".join(variants) + " |")
    lines.append("|" + "---|" * (len(variants) + 1))
    for m in all_motions:
        row = [f"| {m} |"]
        for v in variants:
            ts = [t for t in trials_by_variant[v] if t["motion"] == m]
            ok = [t for t in ts if t["success"]]
            rm = agg([t["rmse_all_rad"] for t in ok if "rmse_all_rad" in t])
            row.append(f" {rm} ({len(ok)}/{len(ts)}) |")
        lines.append("".join(row))
    lines.append("")

    # optional plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for v in variants:
            pol = [t["policy_us_p50"] for t in trials_by_variant[v]
                   if "policy_us_p50" in t]
            if pol:
                axes[0].hist(pol, bins=20, alpha=0.6, label=v)
            rmse = [t["rmse_all_rad"] for t in trials_by_variant[v]
                    if "rmse_all_rad" in t and t["success"]]
            if rmse:
                axes[1].hist(rmse, bins=20, alpha=0.6, label=v)
        axes[0].set_title("policy p50 latency per trial (µs)")
        axes[1].set_title("joint tracking RMSE per trial (rad)")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / "summary_plots.png", dpi=120)
        lines.append(f"![plots](summary_plots.png)")
        lines.append("")
    except ImportError:
        lines.append("_matplotlib not installed — skipping plots._")
        lines.append("")

    out = root / "summary.md"
    out.write_text("\n".join(lines))
    # also dump raw per-trial metrics
    raw = [t for ts in trials_by_variant.values() for t in ts]
    (root / "trial_metrics.json").write_text(json.dumps(raw, indent=2))
    print(f"[analyze] {len(raw)} trials -> {out}")


if __name__ == "__main__":
    main()
