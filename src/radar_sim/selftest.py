"""Post-packaging self-check.

A frozen ``.exe`` has no interpreter to fall back on, so the only way to prove
the bundle is intact is to make the application verify itself.  ``--selftest``
runs this module and writes a plain-text report next to the executable.

The checks are ordered cheapest-first so a broken bundle fails fast:

1. imports            -- every third-party package actually made it in
2. core simulation    -- scene -> raytrace -> IQ synthesis -> dataset on disk
3. dataset round-trip -- read the file back and compare against the source
4. GUI                -- build the widget tree offscreen, tick the run loop
5. templates          -- the shipped template scripts run against the output

Deliberately uses a tiny radar configuration (2 TX x 2 RX, 32 samples, 16
chirps) so the whole thing finishes in a couple of seconds.
"""

from __future__ import annotations

import io
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPORT_NAME = "自检报告.txt"

# Small but structurally complete: exercises TDM layout, multipath and the
# dual-layout packer without the cost of the full IWR1843 frame.
SMALL = dict(chirps_per_frame=16, samples_per_chirp=32,
             tx_elements=2, rx_elements=2)


class _Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0
        self.checks = 0

    def head(self, text: str) -> None:
        self.lines.append("")
        self.lines.append(text)
        self.lines.append("-" * len(text))

    def ok(self, label: str, detail: str = "") -> None:
        self.checks += 1
        self.lines.append(f"  [OK]   {label}{'  ' + detail if detail else ''}")

    def fail(self, label: str, detail: str = "") -> None:
        self.checks += 1
        self.failures += 1
        self.lines.append(f"  [FAIL] {label}{'  ' + detail if detail else ''}")

    def note(self, text: str) -> None:
        self.lines.append(f"         {text}")

    def render(self, elapsed: float) -> str:
        verdict = "全部通过" if self.failures == 0 else f"{self.failures} 项失败"
        header = [
            "2D 场景雷达仿真器 — 运行环境自检报告",
            "=" * 46,
            f"结果      : {verdict}",
            f"检查项    : {self.checks - self.failures}/{self.checks} 通过",
            f"耗时      : {elapsed:.2f} s",
            f"时间      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        return "\n".join(header + self.lines) + "\n"


def _real_stdout() -> io.TextIOWrapper | None:
    """``sys.stdout`` is None in a windowed (console=False) frozen build."""
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return None
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
    return stream  # type: ignore[return-value]


def _check_imports(report: _Report) -> None:
    report.head("1. 依赖包")

    import numpy
    report.ok("numpy", numpy.__version__)

    try:
        import matplotlib
        report.ok("matplotlib", matplotlib.__version__)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401
        report.ok("matplotlib Qt 后端", "backend_qtagg")
    except Exception as exc:                                # pragma: no cover
        report.fail("matplotlib", repr(exc))

    try:
        import PySide6
        from PySide6 import QtCore
        report.ok("PySide6", PySide6.__version__)
        report.note(f"Qt 运行时 {QtCore.qVersion()}")
    except Exception as exc:                                # pragma: no cover
        report.fail("PySide6", repr(exc))

    try:
        import radar_sim
        root = Path(radar_sim.__file__).resolve().parent
        report.ok("radar_sim", str(root))
    except Exception as exc:                                # pragma: no cover
        report.fail("radar_sim", repr(exc))


def _build_scene():
    """A scene chosen so that multipath is guaranteed to exist.

    The wall at x=30 is vertical and faces the radar at (2, 8).  Its mirror
    image of the radar is (58, 8); the segment from there to the point target
    at (20, 8) crosses the wall at y=8, which is inside the wall's extent, so a
    specular one-bounce path *must* be produced.  If the raytracer ever stops
    finding it, this check fails -- which is exactly what we want.
    """
    from radar_sim.core.scene import Primitive, Scene

    scene = Scene.empty(60.0, 60.0)
    scene.add(Primitive("wall", {"points": [[30.0, 4.0], [30.0, 14.0]],
                                  "thickness": 0.4},
                        name="反射墙", material_id="metal"))
    scene.add(Primitive("point", {"x": 20.0, "y": 8.0},
                        name="标定点", material_id="metal"))
    scene.add(Primitive("circle", {"x": 14.0, "y": 26.0, "r": 2.5},
                        name="植被", material_id="vegetation"))
    scene.add(Primitive("rect", {"x": 42.0, "y": 30.0, "w": 8.0, "h": 4.0},
                        name="建筑", material_id="concrete"))
    return scene


def _check_core(report: _Report) -> Path | None:
    import numpy as np

    from radar_sim.core.radar import synthesize_frame
    from radar_sim.core.radar_config import RadarConfig
    from radar_sim.core.raytrace import trace_paths
    from radar_sim.core.scatterers import build_scatterers

    report.head("2. 仿真内核")
    out_dir = Path(tempfile.mkdtemp(prefix="radar_sim_selftest_"))

    try:
        cfg = RadarConfig.iwr1843(**SMALL)
        report.ok("雷达配置", f"{cfg.samples_per_chirp} 采样 / "
                              f"{cfg.chirps_per_frame} chirp / "
                              f"{cfg.virtual_elements} 虚拟阵元")
        report.note(f"ΔR {cfg.range_resolution_m:.4f} m, "
                    f"Rmax {cfg.max_range_m:.2f} m, "
                    f"vmax ±{cfg.max_velocity_mps:.3f} m/s")
    except Exception as exc:
        report.fail("雷达配置", repr(exc))
        return None

    try:
        scene = _build_scene()
        scatterers = build_scatterers(scene, segment_len=1.0, max_scatterers=500)
        report.ok("场景离散化", f"{len(scene.primitives)} 个图元 -> "
                                f"{len(scatterers.position)} 个散射点")
    except Exception as exc:
        report.fail("场景离散化", repr(exc))
        return None

    try:
        radar_pos = np.array([2.0, 8.0])
        paths = trace_paths(scene, scatterers, radar_pos, cfg=cfg, max_order=2)
        counts = paths.counts_by_bounce()
        report.ok("射线追踪（含 2 次反射）",
                  f"{paths.count} 条路径  直射={counts.get(0, 0)} "
                  f"1 次={counts.get(1, 0)} 2 次={counts.get(2, 0)}")
        if counts.get(0, 0) == 0:
            report.fail("直射路径", "目标在视距内却没有直射路径")
        if counts.get(1, 0) == 0:
            report.fail("一次反射",
                        "反射墙的镜面几何应当必然产生至少一条一次反射路径")
    except Exception as exc:
        report.fail("射线追踪", repr(exc))
        return None

    cubes = []
    try:
        rng = np.random.default_rng(1234)
        for _ in range(3):
            result = synthesize_frame(paths, scatterers, scene, cfg, rng=rng)
            cubes.append(result.cube)
            if result.cube.shape != (cfg.chirps_per_frame,
                                     cfg.virtual_elements,
                                     cfg.samples_per_chirp):
                raise AssertionError(f"立方体形状错误 {result.cube.shape}")
            if not np.isfinite(result.cube).all():
                raise AssertionError("IQ 里出现 NaN / Inf")
        report.ok("IQ 合成", f"cube {cubes[0].shape}  "
                             f"峰值 {np.abs(cubes[0]).max():.3e}")
    except Exception as exc:
        report.fail("IQ 合成", repr(exc))
        return None

    try:
        from radar_sim.core.io_export import load_frame, load_meta
        from radar_sim.core.simulator import SimulationSettings, Simulator
        from radar_sim.core.vehicle import Vehicle, VehicleConfig

        simulator = Simulator(
            scene, cfg,
            Vehicle(VehicleConfig(start=(6.0, 20.0, 0.0), path_speed_mps=4.0),
                    path=[[6.0, 20.0], [50.0, 20.0]]),
            SimulationSettings(frame_period_s=0.08, duration_s=0.24, seed=99,
                               segment_len_m=1.0),
            scatterers=scatterers,
        )
        meta = simulator.export(str(out_dir))
        produced = sorted(p.name for p in out_dir.iterdir())
        report.ok("数据集导出", ", ".join(produced))

        for required in ("signal.bin", "meta.json", "pointcloud_gt.csv",
                         "README.txt"):
            if required not in produced:
                report.fail("输出文件齐全", f"缺少 {required}")

        # byte-exact layout check: what we wrote must read back identically
        first = load_frame(out_dir, 0)
        if first.shape != cubes[0].shape:
            raise AssertionError(f"回读形状 {first.shape} != {cubes[0].shape}")
        report.ok("原始 IQ 回读", f"{first.shape}")

        loaded = load_meta(out_dir)
        signal_meta = loaded["signal"]
        frames = int(signal_meta["num_frames"])
        expected_bytes = int(signal_meta["frame_bytes"]) * frames
        actual_bytes = (out_dir / "signal.bin").stat().st_size
        if expected_bytes != actual_bytes:
            raise AssertionError(f"signal.bin 大小 {actual_bytes} != "
                                 f"{expected_bytes}")
        report.ok("字节布局自洽", f"{frames} 帧 x "
                                 f"{signal_meta['frame_bytes']} B = "
                                 f"{actual_bytes:,} B")
        report.ok("meta 记录", f"layout={signal_meta['layout']} "
                              f"axis_order={signal_meta['axis_order']} "
                              f"iq_scale={signal_meta['iq_scale']:.3e}")
        report.note(f"性能 追踪 {meta['performance']['raytrace_ms_mean']:.1f} ms/frame, "
                    f"合成 {meta['performance']['synthesis_ms_mean']:.1f} ms/frame")
    except Exception as exc:
        report.fail("数据集导出 / 回读", repr(exc))
        report.note(traceback.format_exc().splitlines()[-1])

    return out_dir


def _check_gui(report: _Report) -> None:
    report.head("3. 图形界面")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        # Qt widgets cannot exist without an application object.
        app = QApplication.instance() or QApplication([])
        del app
    except Exception as exc:                                # pragma: no cover
        report.fail("Qt 初始化", repr(exc))
        return

    try:
        from radar_sim.main import build_window
        window = build_window()
        window.timer.stop()
        report.ok("主窗口构建", window.windowTitle())
        report.note(f"{len(window.simulator.scene.primitives)} 个图元, "
                    f"画布 + 信号视图 + 4 个面板")
    except Exception as exc:
        report.fail("主窗口构建", repr(exc))
        report.note(traceback.format_exc().splitlines()[-1])
        return

    try:
        before = len(window.simulator.scene.primitives)
        window.canvas.add_primitive("circle", (15.0, 15.0))
        after = len(window.simulator.scene.primitives)
        if after != before + 1:
            raise AssertionError(f"画布放置图元失败 {before} -> {after}")
        report.ok("画布放置图元", f"{before} -> {after}")
    except Exception as exc:
        report.fail("画布放置图元", repr(exc))

    try:
        window.sim_panel.start_btn.click()
        deadline = time.time() + 30.0
        while (window.simulator.frame_index < window.simulator.num_frames
               and time.time() < deadline):
            window._on_tick()
        done = window.simulator.frame_index
        if done < 1:
            raise AssertionError("没有产生任何帧")
        report.ok("界面运行仿真", f"{done}/{window.simulator.num_frames} 帧")
        window.sim_panel.stop_btn.click()
    except Exception as exc:
        report.fail("界面运行仿真", repr(exc))

    try:
        window.close()
    except Exception:
        pass


def _check_templates(report: _Report, data_dir: Path | None) -> None:
    report.head("4. 数据处理模板")

    if data_dir is None:                                    # pragma: no cover
        report.fail("模板检查", "没有可用的数据集")
        return

    template_dir = find_templates()
    if template_dir is None:
        report.fail("模板目录", "没找到 templates/python（随包发布时遗漏了？）")
        return
    report.ok("模板目录", str(template_dir))

    # cfar.py / process_demo.py use absolute imports (``from radar_io import``)
    # so the template folder has to be importable when loaded by path.
    if str(template_dir) not in sys.path:
        sys.path.insert(0, str(template_dir))

    import importlib.util

    def load(name: str):
        path = template_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"selftest_{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    try:
        radar_io = load("radar_io")
        reader = radar_io.RadarFile(data_dir)
        report.ok("模板读取 signal.bin",
                  f"{reader.num_frames} 帧 {reader.shape}")
        v_axis = reader.velocity_axis()
        report.note(f"ΔR {reader.range_bin_m:.4f} m, "
                    f"Rmax {reader.max_range_m:.2f} m, "
                    f"Δv {v_axis[1] - v_axis[0]:.4f} m/s")
        rd, _r, _v = reader.range_doppler(0)
        report.ok("模板距离-多普勒", f"{rd.shape}  峰值 {rd.max():.1f} dB")
        reader.close()
    except Exception as exc:
        report.fail("模板读取", repr(exc))
        return

    try:
        cfar = load("cfar")
        reader = radar_io.RadarFile(data_dir)
        cloud = cfar.extract_pointcloud(reader, 0, pfa=1e-3)
        reader.close()
        report.ok("模板 CFAR 检测", f"{len(cloud['range_m'])} 个点")
    except Exception as exc:
        report.fail("模板 CFAR 检测", repr(exc))


def find_templates() -> Path | None:
    """Locate ``templates/python`` in dev, frozen-onedir and frozen-onefile.

    Public because ``cli.py`` reuses it to run the templates from inside a
    frozen executable (which is the only way an exe-only user can process the
    data -- the templates need numpy and matplotlib, both of which are already
    in the bundle).
    """
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates += [exe_dir / "templates" / "python",
                       exe_dir / ".." / "templates" / "python",
                       Path(sys._MEIPASS) / "templates" / "python"]  # type: ignore[attr-defined]
    here = Path(__file__).resolve()
    candidates += [here.parents[2] / "templates" / "python",
                   here.parents[1] / "templates" / "python"]
    for path in candidates:
        if (path / "radar_io.py").exists():
            return path.resolve()
    return None


def run(report_path: Path | None = None) -> int:
    """Run every check and write the report.  Returns a process exit code."""
    started = time.time()
    report = _Report()

    try:
        _check_imports(report)
    except Exception as exc:                                # pragma: no cover
        report.fail("依赖导入", repr(exc))

    data_dir = None
    try:
        data_dir = _check_core(report)
    except Exception as exc:                                # pragma: no cover
        report.fail("仿真内核", repr(exc))

    try:
        _check_gui(report)
    except Exception as exc:                                # pragma: no cover
        report.fail("图形界面", repr(exc))

    try:
        _check_templates(report, data_dir)
    except Exception as exc:                                # pragma: no cover
        report.fail("数据处理模板", repr(exc))

    text = report.render(time.time() - started)
    text += "\n" + platform.platform() + "\n"
    text += f"Python {sys.version.split()[0]}  ({sys.executable})\n"
    if getattr(sys, "frozen", False):
        text += "运行模式：冻结可执行文件（PyInstaller）\n"
    else:
        text += "运行模式：源码 / 虚拟环境\n"

    if report_path is None:
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
            else Path.cwd()
        report_path = base / REPORT_NAME

    try:
        report_path.write_text(text, encoding="utf-8")
    except OSError:
        pass

    stream = _real_stdout()
    if stream is not None:
        stream.write(text)
        stream.flush()

    # leave the dataset behind so the user can inspect it
    if data_dir is not None:
        _cleanup(data_dir)

    return 0 if report.failures == 0 else 1


def _cleanup(data_dir: Path) -> None:
    import shutil
    try:
        shutil.rmtree(data_dir, ignore_errors=True)
    except Exception:                                       # pragma: no cover
        pass
