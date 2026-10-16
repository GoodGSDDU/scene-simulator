"""One-command packaging for the 2D radar scene simulator.

    python packaging/build.py                 # folder build + single-file build
    python packaging/build.py --only dir      # just the folder build
    python packaging/build.py --only onefile
    python packaging/build.py --no-selftest   # skip the post-build verification

Output lands in ``dist/``::

    dist/雷达仿真器/              <- double-click 雷达仿真器.exe
       雷达仿真器.exe
       radar-sim-cli.exe
       _internal/
       templates/  examples/  docs/
       使用说明.txt   运行自检.cmd
    dist/雷达仿真器-单文件版.exe   <- everything in one file

Every build is verified by running the frozen executable's own ``--selftest``,
so a broken bundle is caught here instead of by the user.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACKAGING = Path(__file__).resolve().parent
PROJECT = PACKAGING.parent
SPEC = PACKAGING / "radar_sim.spec"
DIST = PROJECT / "dist"
BUILD = PROJECT / "build"

# A Windows console defaults to GBK, so redirected logs turn into mojibake.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # pragma: no cover
    pass

DIR_NAME = "雷达仿真器"
ONEFILE_NAME = "雷达仿真器-单文件版"

_SKIP_SELFTEST = False

# Written next to the executables so a non-programmer can get started.
# Raw string: the Windows paths below are meant to be literal backslashes.
USAGE_TEXT = r"""\
2D 场景雷达仿真器  v0.1.0
=========================================================

【怎么启动】

    双击「雷达仿真器.exe」即可。

    首次启动需要十几秒解压运行库，之后会快一些。
    如果 Windows 弹出「已保护你的电脑」提示，点「更多信息」→「仍要运行」。

【怎么用】

    1. 左侧点形状按钮（圆 / 矩形 / 多边形 / 墙 / 点目标），
       然后在画布上点击，图元就放下了。
    2. 选中图元后，右侧「图元属性」里可以改尺寸、位置、角度，
       也可以换材质（金属 / 混凝土 / 玻璃 / 木头 / 人体 / 植被）。
    3. 「雷达」页可以改雷达参数，改完下面会实时显示派生指标
       （距离分辨率、最大距离、最大不模糊速度……）。
    4. 选中小车（车辆页），可以设置起点朝向、路径点、行驶速度。
    5. 点底部「▶ 开始仿真」。
       - 路径模式：小车自动沿路径开。
       - 手动模式：用方向键 / WASD 驾驶。
    6. 右侧信号视图可以切换看：原始 IQ、距离像、距离-多普勒、点云。
    7. 点「💾 导出数据」把原始的二进制 IQ 存下来，
       再用下面的方法处理它。

【拿到数据之后怎么处理】

    最省事：直接用自带的命令行版，不用装 Python：

        radar-sim-cli.exe --process <导出的文件夹> --out figures
        radar-sim-cli.exe --compare <导出的文件夹> --out .

    第一条出四张图（原始 IQ / 距离像 / 距离-多普勒 / 点云），
    第二条给出与真值的误差统计（距离 / 速度 / 方位 RMS、虚警率）。

【想改代码 / 用自己的工具】

    templates/python    Python 模板（只要 numpy / matplotlib，可直接拷走）
    templates/matlab    MATLAB 模板
    templates/notebook  Jupyter 教学 Notebook

    模板里就是上面两条命令跑的同一份代码：

        cd templates\python
        python process_demo.py --data <导出的文件夹> --out figures

    详细说明见 templates\python\README.md，
    格式与公式定义见 docs\DESIGN.md 第 8 章和第 10 章。

【想用命令行批量生成数据】

    radar-sim-cli.exe --demo --out dataset\demo --frames 20
    radar-sim-cli.exe --help

【怀疑程序装坏了】

    双击「运行自检.cmd」，会把每一项检查的结果写进「自检报告.txt」。
    全绿就没问题；有 FAIL 的那一行会把原因写清楚。

【常见问题】

    启动很慢 / 卡一下       单文件版每次启动都要解压（实测约 3 秒），属正常。
                            嫌慢就用文件夹版，启动快得多。
    杀毒软件报警             PyInstaller 打包的程序常被误报，
                            加白名单即可，源码都在 src\ 下可以自己核对。
    字体显示成方块          系统缺少中文字体，装一个「微软雅黑」即可。
    导出的 signal.bin 很大   默认一帧 1.5 MiB，128 帧就是 192 MiB。
                            帧数可以在「仿真」面板里调小。
"""

SELFTEST_CMD = """\
@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo 正在自检，请稍候（约 5-20 秒）...
echo.
"%~dp0雷达仿真器.exe" --selftest "%~dp0自检报告.txt"
if errorlevel 1 (
  echo.
  echo ^>^> 自检未通过，详见「自检报告.txt」。
) else (
  echo.
  echo ^>^> 自检全部通过。
)
echo.
echo 报告已写入：%~dp0自检报告.txt
pause
"""


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(" ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kwargs)


# --------------------------------------------------------------------------- #
def make_icon() -> bool:
    log("生成图标 ...")
    result = run([sys.executable, str(PACKAGING / "make_icon.py")],
                 cwd=str(PROJECT), capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  图标生成失败，继续打包但不带图标：{result.stderr.strip()[:200]}")
        return False
    return (PACKAGING / "icon.ico").exists()


def pyinstaller(onefile: bool, name: str, clean: bool) -> bool:
    env = dict(os.environ)
    env["RADAR_SIM_DIST_NAME"] = name
    if onefile:
        env["RADAR_SIM_ONEFILE"] = "1"
    else:
        env.pop("RADAR_SIM_ONEFILE", None)
    # PyInstaller must not inherit a Qt platform override from a test shell.
    env.pop("QT_QPA_PLATFORM", None)

    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC),
           "--noconfirm", "--distpath", str(DIST), "--workpath", str(BUILD)]
    if clean:
        cmd.append("--clean")
    # NOTE: --onefile / --name / --icon are ignored when a .spec file is given;
    # the spec reads RADAR_SIM_ONEFILE / RADAR_SIM_DIST_NAME from the env
    # instead.  --distpath and --workpath are honoured here.

    result = run(cmd, cwd=str(PROJECT), env=env)
    return result.returncode == 0


def copy_shared_files(target: Path) -> None:
    for folder in ("templates", "examples", "docs"):
        src = PROJECT / folder
        if not src.exists():
            log(f"  警告：{folder}/ 不存在，跳过")
            continue
        dst = target / folder
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "figures"))
    log("  已随包复制 templates/ examples/ docs/")


def write_helpers(target: Path) -> None:
    (target / "使用说明.txt").write_text(USAGE_TEXT, encoding="utf-8")
    (target / "运行自检.cmd").write_text(SELFTEST_CMD, encoding="utf-8")
    log("  已写入 使用说明.txt / 运行自检.cmd")


def verify(executable: Path, label: str) -> bool:
    """Run the frozen app's own selftest.  This is the point of the exercise."""
    if _SKIP_SELFTEST:
        log(f"  [{label}] 已跳过自检（--no-selftest）")
        return True
    if not executable.exists():
        log(f"  [{label}] 找不到 {executable}")
        return False

    report = executable.parent / f"自检报告_{label}.txt"
    report.unlink(missing_ok=True)

    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"     # no display needed
    env["PYTHONIOENCODING"] = "utf-8"

    started = time.time()
    try:
        result = run([str(executable), "--selftest", str(report)],
                     cwd=str(executable.parent), env=env,
                     capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        log(f"  [{label}] 自检超时（900s）")
        return False
    elapsed = time.time() - started

    if report.exists():
        text = report.read_text(encoding="utf-8", errors="replace")
        failures = [ln.strip() for ln in text.splitlines() if "[FAIL]" in ln]
        if failures:
            log(f"  [{label}] 自检 {len(failures)} 项失败  ({elapsed:.1f}s)")
            for line in failures:
                log(f"      {line}")
            print(text, flush=True)
            return False
        log(f"  [{label}] 自检全部通过  ({elapsed:.1f}s)")
        return result.returncode == 0

    log(f"  [{label}] 没有生成报告，退出码 {result.returncode}")
    if result.stdout:
        log(result.stdout[-2000:])
    if result.stderr:
        log(result.stderr[-2000:])
    return False


def verify_cli_workflow(executable: Path) -> bool:
    """Verify the CLI the way a user actually drives it.

    The generic ``--selftest`` exercises the core through the Python API, which
    would not catch a broken argparse wiring, a missing scene loader or a
    frozen-path problem in the exporter.  So generate a real dataset instead.
    """
    if _SKIP_SELFTEST:
        log("  [CLI] 已跳过工作流验证（--no-selftest）")
        return True
    if not executable.exists():
        log(f"  [CLI] 找不到 {executable}")
        return False

    import json
    import tempfile

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.TemporaryDirectory(prefix="radar_sim_cli_") as tmp:
        out_dir = Path(tmp) / "ds"
        started = time.time()
        try:
            result = run([str(executable), "--demo", "--out", str(out_dir),
                          "--frames", "4", "--seed", "7", "--order", "1",
                          "--quiet"],
                         cwd=tmp, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=900)
        except subprocess.TimeoutExpired:
            log("  [CLI] 生成数据集超时（900s）")
            return False
        elapsed = time.time() - started

        if result.returncode != 0:
            log(f"  [CLI] 退出码 {result.returncode}")
            log((result.stderr or result.stdout or "")[-1500:])
            return False

        produced = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        required = {"signal.bin", "meta.json", "pointcloud_gt.csv", "README.txt"}
        missing = required - set(produced)
        if missing:
            log(f"  [CLI] 缺少输出文件：{sorted(missing)}")
            return False

        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        frames = int(meta["signal"]["num_frames"])
        expected = int(meta["signal"]["frame_bytes"]) * frames
        actual = (out_dir / "signal.bin").stat().st_size
        if frames != 4 or expected != actual:
            log(f"  [CLI] 字节数不符：{frames} 帧 x "
                f"{meta['signal']['frame_bytes']} B != {actual} B")
            return False

        mb = actual / 1024 / 1024
        log(f"  [CLI] 生成数据集正常：{frames} 帧 / {mb:.1f} MiB / "
            f"layout={meta['signal']['layout']}  ({elapsed:.1f}s)")
    return True


def report_size(target: Path) -> None:
    if target.is_file():
        mb = target.stat().st_size / 1024 / 1024
        log(f"  {target.name}: {mb:.1f} MiB")
        return
    total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    count = sum(1 for f in target.rglob("*") if f.is_file())
    log(f"  {target.name}/: {total / 1024 / 1024:.1f} MiB，{count} 个文件")


def measure_startup(executable: Path) -> float | None:
    """Time the exe from launch to the interpreter being ready.

    This is the floor: it includes the PyInstaller bootloader unpacking the
    archive (the dominant cost for onefile) but exits before Qt and matplotlib
    are imported.  The whole-bundle self check is the ceiling -- it does far
    more work than opening a window -- so the true "double-click to a usable
    window" sits between the two numbers reported here.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.time()
    try:
        result = run([str(executable), "--version"], env=env,
                     capture_output=True, text=True,
                     encoding="utf-8", errors="replace", timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return time.time() - started


# --------------------------------------------------------------------------- #
def build_dir(clean: bool) -> Path | None:
    log("=" * 62)
    log("构建：文件夹版（启动快，推荐）")
    log("=" * 62)
    if not pyinstaller(onefile=False, name=DIR_NAME, clean=clean):
        log("PyInstaller 失败")
        return None

    target = DIST / DIR_NAME
    if not target.exists():
        log(f"没找到产出目录 {target}")
        return None

    log("附带文件 ...")
    copy_shared_files(target)
    write_helpers(target)
    report_size(target)

    log("验证 ...")
    gui = target / f"{DIR_NAME}.exe"
    cli = target / "radar-sim-cli.exe"
    ok_gui = verify(gui, "GUI")
    # The CLI gets a workflow test instead: the generic selftest goes through
    # the Python API and would not notice a broken argument parser.
    ok_cli = verify_cli_workflow(cli) and verify(cli, "CLI")
    if not (ok_gui and ok_cli):
        log("自检未通过。")
        return None

    seconds = measure_startup(gui)
    if seconds is not None:
        log(f"  文件夹版进程就绪 {seconds:.1f}s（双击后的实际等待另含 Qt/matplotlib 加载）")
    log("文件夹版构建成功。")
    return target


def build_onefile(clean: bool) -> Path | None:
    log("=" * 62)
    log("构建：单文件版（一个 exe，启动较慢）")
    log("=" * 62)
    if not pyinstaller(onefile=True, name=ONEFILE_NAME, clean=clean):
        log("PyInstaller 失败")
        return None

    exe = DIST / f"{ONEFILE_NAME}.exe"
    if not exe.exists():
        # PyInstaller may place it in a folder of the same name
        alt = DIST / ONEFILE_NAME / f"{ONEFILE_NAME}.exe"
        if alt.exists():
            exe = alt
        else:
            log(f"没找到产出文件 {exe}")
            return None

    report_size(exe)

    log("验证 ...")
    if not verify(exe, "单文件"):
        log("自检未通过。")
        return None

    seconds = measure_startup(exe)
    if seconds is not None:
        log(f"  单文件版进程就绪 {seconds:.1f}s（含解压归档）")
    log("单文件版构建成功。")
    return exe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打包 2D 场景雷达仿真器")
    parser.add_argument("--only", choices=["dir", "onefile"], default=None,
                        help="只构建其中一种（默认两种都构建）")
    parser.add_argument("--clean", action="store_true", default=True,
                        help="构建前清理缓存（默认开启）")
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.add_argument("--no-selftest", dest="selftest", action="store_false",
                        default=True, help="跳过构建后的自检")
    args = parser.parse_args(argv)

    global _SKIP_SELFTEST
    _SKIP_SELFTEST = not args.selftest

    started = time.time()
    log(f"项目根目录：{PROJECT}")
    DIST.mkdir(exist_ok=True)
    make_icon()

    results: dict[str, bool] = {}
    if args.only in (None, "dir"):
        results["文件夹版"] = build_dir(args.clean) is not None
    if args.only in (None, "onefile"):
        results["单文件版"] = build_onefile(args.clean) is not None

    log("=" * 62)
    log(f"完成，用时 {time.time() - started:.0f}s")
    for name, ok in results.items():
        log(f"  {name}: {'成功' if ok else '失败'}")
    log(f"产出目录：{DIST}")
    log("=" * 62)
    return 0 if all(results.values()) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
