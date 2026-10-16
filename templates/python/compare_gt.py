"""Compare CFAR detections against the ideal point cloud (``pointcloud_gt.csv``).

    python compare_gt.py --data ../../dataset/demo --frames 20

Reports range / velocity / azimuth accuracy plus detection and false-alarm
rates, and writes ``gt_compare.png`` plus ``detections.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cfar import extract_pointcloud  # noqa: E402
from radar_io import RadarFile  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# a detection counts as a hit when it lands inside these gates
WINDOWS = {"range_m": 3.0, "velocity_mps": 0.6, "azimuth_deg": 12.0}


def collapse_truth(truth: dict, rf: RadarFile) -> dict:
    """Keep only the strongest truth path per (range, velocity, angle) cell.

    A wall is discretised into dozens of scatterers but the radar can only
    resolve one cell's worth of them; without this step the "detection rate"
    metric is meaningless.
    """
    if np.size(truth.get("range_m", ())) == 0:
        return truth
    r_bin = np.round(np.asarray(truth["range_m"]) / rf.range_bin_m).astype(int)
    v_bin = np.round(
        np.asarray(truth["radial_vel_mps"]) / rf.velocity_resolution_mps
    ).astype(int)
    sin_res = rf.wavelength / rf.element_spacing / max(rf.n_elements, 1)
    a_bin = np.round(np.sin(np.deg2rad(np.asarray(truth["azimuth_deg"]))) / sin_res).astype(int)
    amplitude = np.asarray(truth["amplitude_db"])

    best: dict[tuple, int] = {}
    for i, key in enumerate(zip(r_bin.tolist(), v_bin.tolist(), a_bin.tolist())):
        current = best.get(key)
        if current is None or amplitude[i] > amplitude[current]:
            best[key] = i
    index = np.asarray(sorted(best.values()), dtype=int)
    return {k: np.asarray(v)[index] for k, v in truth.items()}


def associate(det: dict, truth: dict) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour association in (range, velocity, azimuth).

    Returns ``(pairs, unmatched_detections)`` where ``pairs`` is an
    ``(n, 2)`` array of ``(detection_index, truth_index)``.
    """
    if det["range_m"].size == 0 or truth["range_m"].size == 0:
        return np.zeros((0, 2), int), np.arange(det["range_m"].size)

    # scale each axis by its gate so the metric is dimensionless
    scale = np.array([WINDOWS["range_m"], WINDOWS["velocity_mps"], WINDOWS["azimuth_deg"]])
    d = np.stack([det["range_m"], det["velocity_mps"], det["azimuth_deg"]], axis=1) / scale
    t = np.stack([truth["range_m"], truth["radial_vel_mps"],
                  truth["azimuth_deg"]], axis=1) / scale

    distance = np.linalg.norm(d[:, None, :] - t[None, :, :], axis=2)
    pairs: list[tuple[int, int]] = []
    used_t: set[int] = set()
    for i in np.argsort(distance.min(axis=1)):
        j = int(np.argmin(distance[i]))
        if j in used_t or distance[i, j] > 1.0:
            continue
        pairs.append((int(i), j))
        used_t.add(j)
    matched = {p[0] for p in pairs}
    unmatched = np.array([i for i in range(det["range_m"].size) if i not in matched], int)
    return np.asarray(pairs, int).reshape(-1, 2), unmatched


def main() -> int:
    parser = argparse.ArgumentParser(description="detections vs ground truth")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--min-truth-db", type=float, default=-25.0,
                        help="ignore truth paths weaker than this (un-detectable)")
    parser.add_argument("--direct-only", action="store_true", default=True)
    parser.add_argument("--no-collapse", action="store_true",
                        help="do not merge truth paths into resolution cells")
    args = parser.parse_args()

    errors = {"range": [], "velocity": [], "azimuth": []}
    rows: list[list] = []
    total_det = total_truth = total_hit = 0
    owner_seen: dict[int, bool] = {}
    owner_truth: dict[int, int] = {}

    with RadarFile(args.data) as rf:
        frames = min(args.frames, rf.num_frames)
        for frame in range(frames):
            truth = rf.truth(frame)
            if not truth:
                print("找不到真值，导出时请勾选真值点云")
                return 1
            keep = np.asarray(truth["amplitude_db"]) >= args.min_truth_db
            if args.direct_only:
                keep &= np.asarray(truth["path_type"]) == "direct"
            truth = {k: np.asarray(v)[keep] for k, v in truth.items()}
            if not args.no_collapse:
                truth = collapse_truth(truth, rf)

            det = extract_pointcloud(rf, frame, pfa=args.pfa)
            pairs, unmatched = associate(det, truth)

            for owner in np.unique(truth["target_id"]):
                owner_truth[int(owner)] = owner_truth.get(int(owner), 0) + 1
                owner_seen.setdefault(int(owner), False)
            for i, j in pairs:
                owner_seen[int(truth["target_id"][j])] = True

            total_det += det["range_m"].size
            total_truth += truth["range_m"].size
            total_hit += pairs.shape[0]

            for i, j in pairs:
                dr = det["range_m"][i] - truth["range_m"][j]
                dv = det["velocity_mps"][i] - truth["radial_vel_mps"][j]
                da = det["azimuth_deg"][i] - truth["azimuth_deg"][j]
                da = (da + 180.0) % 360.0 - 180.0
                errors["range"].append(dr)
                errors["velocity"].append(dv)
                errors["azimuth"].append(da)
                rows.append([frame, det["range_m"][i], truth["range_m"][j], dr,
                             det["velocity_mps"][i], truth["radial_vel_mps"][j], dv,
                             det["azimuth_deg"][i], truth["azimuth_deg"][j], da,
                             det["snr_db"][i]])

    print(f"frames        : {frames}")
    print(f"truth scatterers: {total_truth}   (+-unresolvable, see note below)")
    print(f"detections    : {total_det}")
    print(f"matched       : {total_hit}")
    if total_det:
        print(f"false alarm   : {100.0 * (total_det - total_hit) / total_det:.1f} %")
    seen = sum(1 for k, v in owner_seen.items() if v)
    if owner_truth:
        print(f"object visible: {seen}/{len(owner_truth)} primitives detected "
              f"({100.0 * seen / len(owner_truth):.0f} %)")
    print()
    print("note: CFAR detects resolution cells, not individual scatterers -- a wall")
    print("      split into 100 pieces legitimately yields only a few detections.")
    print("      Use the accuracy table below, not a scatterer-by-scatterer hit rate.")
    print()
    print(f"{'':>10}{'RMS':>12}{'bias':>12}{'std':>12}{'P95 |.|':>12}")
    for key in ("range", "velocity", "azimuth"):
        values = np.asarray(errors[key])
        if values.size == 0:
            continue
        unit = {"range": " m", "velocity": " m/s", "azimuth": " deg"}[key]
        print(f"{key:>10}{np.sqrt((values ** 2).mean()):12.4f}"
              f"{values.mean():12.4f}{values.std():12.4f}"
              f"{np.percentile(np.abs(values), 95):12.4f}   {unit.strip()}")

    out_csv = args.out / "detections.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "range_det", "range_gt", "range_err",
                         "vel_det", "vel_gt", "vel_err",
                         "az_det", "az_gt", "az_err", "snr_db"])
        writer.writerows(rows)
    print(f"\n匹配明细: {out_csv}")

    if rows:
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for axes_i, (key, label) in enumerate((("range", "距离误差 (m)"),
                                               ("velocity", "速度误差 (m/s)"),
                                               ("azimuth", "方位误差 (°)"))):
            values = np.asarray(errors[key])
            axes[axes_i].hist(values, bins=30, color="#4c93c9", alpha=0.85)
            axes[axes_i].axvline(0.0, color="red", lw=1.0, ls="--")
            axes[axes_i].set_title(f"{label}  RMS={np.sqrt((values ** 2).mean()):.4f}")
            axes[axes_i].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.out / "gt_compare.png", dpi=130)
        print(f"误差直方图: {args.out / 'gt_compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
