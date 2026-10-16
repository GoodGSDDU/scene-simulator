"""End-to-end processing demo: binary -> range-Doppler -> CFAR -> point cloud.

Run it right after a GUI export or a ``radar-sim-cli`` run::

    python process_demo.py --data ../../dataset/demo --out figures

Produces four PNGs (raw IQ, range profile, range-Doppler map, point cloud) and
prints the detection statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cfar import extract_pointcloud  # noqa: E402
from radar_io import RadarFile, periodic_window  # noqa: E402

# CJK-capable font chain (the default DejaVu Sans has no Chinese glyphs)
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_iq(axes, cube, rf: RadarFile) -> None:
    chirp = cube[0, 0, :]
    n = min(128, chirp.size)
    axes.plot(np.arange(n), chirp[:n].real, lw=1.0, label="I")
    axes.plot(np.arange(n), chirp[:n].imag, lw=1.0, label="Q")
    axes.set_title(f"原始 IQ（chirp 0 / 阵元 0，前 {n} 点）")
    axes.set_xlabel("采样点")
    axes.set_ylabel("幅度")
    axes.legend()
    axes.grid(alpha=0.25)


def plot_range(axes, cube, rf: RadarFile) -> None:
    w = periodic_window(rf.n_samples, "hann")[None, None, :]
    spec = np.fft.fft(cube * w, axis=-1)
    profile = np.sqrt((np.abs(spec) ** 2).sum(axis=(0, 1)))
    axes.plot(rf.range_axis(), 20 * np.log10(np.maximum(profile, 1e-30)), lw=1.0)
    axes.set_title("距离像")
    axes.set_xlabel(f"距离 (m)    ΔR = {rf.range_bin_m:.3f} m")
    axes.set_ylabel("幅度 (dB)")
    axes.set_xlim(0.0, rf.max_range_m)
    axes.grid(alpha=0.25)


def plot_rd(axes, cloud, rf: RadarFile) -> None:
    rd = cloud["rd_map"]
    peak = float(rd.max())
    axes.imshow(rd, aspect="auto", origin="lower", cmap="turbo",
                extent=(0.0, rf.max_range_m, -rf.max_velocity_mps, rf.max_velocity_mps),
                vmin=peak - 45.0, vmax=peak, interpolation="nearest")
    if cloud["range_m"].size:
        axes.plot(cloud["range_m"], cloud["velocity_mps"], "o", ms=5, mfc="none",
                  mec="red", mew=1.0, label="CFAR 检测")
        axes.legend(loc="upper right", fontsize=8)
    axes.set_title(f"距离-多普勒（dv = {rf.velocity_resolution_mps:.4f} m/s）")
    axes.set_xlabel("距离 (m)")
    axes.set_ylabel("速度 (m/s)")
    axes.set_xlim(0.0, rf.max_range_m)


def plot_cloud(axes, cloud, rf: RadarFile, truth: dict) -> None:
    if cloud["x"].size:
        sc = axes.scatter(cloud["x"], cloud["y"], c=cloud["snr_db"], s=26,
                          cmap="autumn", label="检测")
        plt.colorbar(sc, ax=axes, label="SNR (dB)")
    if truth and np.size(truth.get("range_m", ())):
        r = np.asarray(truth["range_m"])
        a = np.deg2rad(np.asarray(truth["azimuth_deg"]))
        axes.scatter(r * np.cos(a), r * np.sin(a), marker="+", s=40, c="lime",
                     label="真值")
    axes.plot([0.0], [0.0], marker="^", ms=9, color="cyan", label="雷达")
    axes.set_title("点云 vs 真值（雷达坐标系）")
    axes.set_xlabel("前方 X (m)")
    axes.set_ylabel("横向 Y (m)")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser(description="2D radar dataset processing demo")
    parser.add_argument("--data", type=Path, required=True, help="dataset directory")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--guard", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--train", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--angle-fft", type=int, default=180)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with RadarFile(args.data) as rf:
        print(rf.summary())
        print()

        cube = rf.frame(args.frame)
        cloud = extract_pointcloud(
            rf, args.frame, guard=tuple(args.guard), train=tuple(args.train),
            pfa=args.pfa, n_angle_fft=args.angle_fft,
        )
        truth = rf.truth(args.frame)

        print(f"frame {args.frame}: 检测点数 = {cloud['range_m'].size}")
        print(f"CFAR alpha = {float(cloud['cfar_alpha']):.4f} (Pfa = {args.pfa:g})")
        if cloud["range_m"].size:
            order = np.argsort(-cloud["power_db"])[:10]
            print("\n  距离(m)   速度(m/s)   方位(°)    SNR(dB)")
            for i in order:
                print(f"{cloud['range_m'][i]:9.2f} {cloud['velocity_mps'][i]:+11.3f} "
                      f"{cloud['azimuth_deg'][i]:9.2f} {cloud['snr_db'][i]:9.1f}")

        figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        plot_iq(axes[0, 0], cube, rf)
        plot_range(axes[0, 1], cube, rf)
        plot_rd(axes[1, 0], cloud, rf)
        plot_cloud(axes[1, 1], cloud, rf, truth)
        figure.tight_layout()
        target = args.out / f"frame_{args.frame:04d}.png"
        figure.savefig(target, dpi=130)
        print(f"\n图已保存: {target}")

        if truth and np.size(truth.get("range_m", ())):
            print(f"真值点数 = {np.size(truth['range_m'])}"
                  f"（仅显示 20 条最强的）")
        else:
            print("未找到 pointcloud_gt.csv —— 导出时勾选“真值点云”即可获得")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
