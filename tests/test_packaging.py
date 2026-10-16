"""Regression tests for the packaging / frozen-app path.

These do not run PyInstaller (that takes minutes); they cover everything that
can silently break the *bundle* while the source tree still works:

* the icon generator (a missing icon makes PyInstaller fail outright)
* the ``--selftest`` entry point, which is the only way a frozen exe can prove
  it is intact
* the frozen-path helpers inside ``selftest.py`` (``_find_templates``)
* the parts of the spec that PyInstaller would not report as a build error,
  e.g. an exclude list that drops something the app needs at runtime

The point is that a bad exclude or a stale hidden-import list shows up here in
seconds instead of as "the exe opens but the signal view is blank".
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
SPEC = PACKAGING / "radar_sim.spec"


# --------------------------------------------------------------------------- #
# icon
# --------------------------------------------------------------------------- #
def test_icon_generator_produces_a_usable_ico(tmp_path, monkeypatch):
    pytest.importorskip("PIL")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "packaging_make_icon", PACKAGING / "make_icon.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["packaging_make_icon"] = module
    spec.loader.exec_module(module)

    image = module.build()
    assert image.size == (module.SIZE, module.SIZE)
    assert image.mode == "RGBA"

    # the plate is opaque in the middle and transparent at the corners
    assert image.getpixel((module.SIZE // 2, module.SIZE // 2))[3] == 255
    assert image.getpixel((1, 1))[3] == 0


def test_committed_icon_exists_and_is_not_empty():
    icon = PACKAGING / "icon.ico"
    if not icon.exists():
        pytest.skip("icon.ico not generated yet (run packaging/make_icon.py)")
    assert icon.stat().st_size > 1024, "icon.ico looks truncated"


# --------------------------------------------------------------------------- #
# selftest entry point
# --------------------------------------------------------------------------- #
def test_selftest_entry_point_writes_a_passing_report(tmp_path):
    """The frozen app's only self-diagnosis path must actually work."""
    report = tmp_path / "report.txt"
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.main", "--selftest", str(report)],
        cwd=str(ROOT),
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen",
             "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300,
    )
    assert report.exists(), f"no report written\n{result.stdout}\n{result.stderr}"
    text = report.read_text(encoding="utf-8")

    failures = [ln.strip() for ln in text.splitlines() if "[FAIL]" in ln]
    assert not failures, "selftest reported failures:\n" + "\n".join(failures)
    assert result.returncode == 0

    # every stage must have actually run, not been skipped
    for section in ("依赖包", "仿真内核", "图形界面", "数据处理模板"):
        assert section in text, f"selftest skipped section {section}"
    assert "全部通过" in text


def test_version_flag_works_without_starting_qt():
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.main", "--version"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0
    assert "radar-sim" in result.stdout


# --------------------------------------------------------------------------- #
# CLI: the frozen binary is verified by actually generating a dataset, so the
# argument wiring has to be covered too.  A missing option here shows up as
# "exit code 2 + usage text" from a shipped executable.
# --------------------------------------------------------------------------- #
def test_cli_accepts_selftest():
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.cli", "--selftest"],
        cwd=str(ROOT),
        env={"QT_QPA_PLATFORM": "offscreen", "PATH": os.environ.get("PATH", ""),
             "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, \
        f"CLI rejected --selftest (rc={result.returncode})\n{result.stderr[-1500:]}"


def test_cli_generates_a_dataset_end_to_end(tmp_path):
    """Mirror of what packaging/build.py runs against the frozen CLI."""
    import json

    out = tmp_path / "ds"
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.cli", "--demo", "--out", str(out),
         "--frames", "2", "--seed", "7", "--order", "1", "--quiet"],
        cwd=str(ROOT),
        env={"QT_QPA_PLATFORM": "offscreen", "PATH": os.environ.get("PATH", ""),
             "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, \
        f"CLI failed\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}"

    produced = {p.name for p in out.iterdir()}
    assert {"signal.bin", "meta.json", "pointcloud_gt.csv",
            "README.txt"} <= produced, f"missing outputs: {produced}"

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    frames = int(meta["signal"]["num_frames"])
    assert frames == 2, f"asked for 2 frames, meta says {frames}"
    expected = int(meta["signal"]["frame_bytes"]) * frames
    assert (out / "signal.bin").stat().st_size == expected, \
        "signal.bin size disagrees with meta.json"


# --------------------------------------------------------------------------- #
# --process / --compare: the self-sufficiency path.  A frozen .exe has no
# interpreter to fall back on, so if these break, an exe-only user can generate
# data but never look at it.
# --------------------------------------------------------------------------- #
def _clean_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "APPDATA",
            "LOCALAPPDATA", "COMSPEC", "NUMBER_OF_PROCESSORS")
    return {k: os.environ[k] for k in keep if k in os.environ}


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("proc_ds") / "ds"
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.cli", "--demo", "--out", str(out),
         "--frames", "2", "--seed", "5", "--quiet"],
        cwd=str(ROOT), env={**_clean_env(), "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return out


def test_cli_process_runs_the_shipped_template(small_dataset, tmp_path):
    """--process must run templates/python/process_demo.py inside the app."""
    out = tmp_path / "figures"          # deliberately does not exist yet
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.cli", "--process", str(small_dataset),
         "--out", str(out)],
        cwd=str(ROOT), env={**_clean_env(), "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, \
        f"--process failed\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}"
    assert (out / "frame_0000.png").exists(), \
        "process_demo did not produce a figure"
    # the CLI must create the output directory rather than making the user do it
    assert out.is_dir()


def test_cli_compare_runs_the_shipped_template(small_dataset, tmp_path):
    out = tmp_path / "cmp"
    result = subprocess.run(
        [sys.executable, "-m", "radar_sim.cli", "--compare", str(small_dataset),
         "--out", str(out)],
        cwd=str(ROOT), env={**_clean_env(), "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, \
        f"--compare failed\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}"
    assert (out / "detections.csv").exists()
    assert "range" in result.stdout


def test_process_reports_a_helpful_error_when_templates_are_missing(
        small_dataset, tmp_path, monkeypatch):
    """Without templates/ the user must get an explanation, not a traceback."""
    import radar_sim.cli as cli
    from radar_sim import selftest

    monkeypatch.setattr(selftest, "find_templates", lambda: None)
    rc = cli.main(["--process", str(small_dataset), "--out", str(tmp_path / "x")])
    assert rc == 2


# --------------------------------------------------------------------------- #
# frozen-path helpers
# --------------------------------------------------------------------------- #
def test_find_templates_locates_the_shipped_templates():
    from radar_sim.selftest import find_templates

    found = find_templates()
    assert found is not None, "selftest cannot find templates/python"
    assert (found / "radar_io.py").exists()
    assert (found / "cfar.py").exists()


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
def _spec_source() -> str:
    if not SPEC.exists():
        pytest.skip("packaging/radar_sim.spec is missing")
    return SPEC.read_text(encoding="utf-8")


def test_spec_excludes_nothing_the_app_needs():
    """An over-eager exclude produces an exe that starts and then dies."""
    source = _spec_source()

    # pull the EXCLUDES and HIDDEN lists out of the spec by parsing it
    tree = ast.parse(source)
    lists: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lists[target.id] = [
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ]

    excluded = set(lists.get("EXCLUDES", []))
    hidden = set(lists.get("HIDDEN", []))

    assert excluded, "no EXCLUDES found in the spec"
    assert hidden, "no HIDDEN found in the spec"

    # things the app genuinely imports at runtime
    for required in ("numpy", "matplotlib", "PySide6"):
        assert required not in excluded, \
            f"{required} is imported by the app but excluded from the bundle"

    # matplotlib resolves its Qt binding dynamically, so the backend must be
    # listed explicitly
    assert any(h.startswith("matplotlib.backends.backend_qt") for h in hidden), \
        "the matplotlib Qt backend is not in hiddenimports"


def test_excluded_modules_are_not_in_the_real_import_closure():
    """The guard that would have caught excluding ``unittest``.

    Checking a hand-written list of "important" packages is not enough: the
    bundle that failed shipped numpy, matplotlib and PySide6 and still died,
    because matplotlib imports ``unittest`` internally.  So import the whole
    application for real, then intersect ``sys.modules`` with the exclude list.
    """
    import json
    import os

    tree = ast.parse(_spec_source())
    excluded: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXCLUDES":
                    excluded = [elt.value for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)]
    assert excluded, "no EXCLUDES found in the spec"

    probe = textwrap.dedent("""
        import json, sys
        excluded = set(json.loads(sys.argv[1]))

        # touch everything the frozen bundle must be able to do
        import radar_sim.main
        from radar_sim.ui.main_window import MainWindow          # noqa: F401
        from radar_sim.ui.signal_view import SignalView          # noqa: F401
        import matplotlib
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401
        from matplotlib.figure import Figure                      # noqa: F401
        import PySide6.QtWidgets, PySide6.QtGui, PySide6.QtCore
        import numpy as np

        loaded = {name.split(".")[0] for name in sys.modules}
        # a bare "test" directory in the cwd would show up spuriously
        hit = sorted((loaded & excluded) - {"test"})
        print(json.dumps(hit))
    """)

    result = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(excluded)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
             "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, \
        f"import probe failed\n{result.stdout}\n{result.stderr}"

    hit = json.loads(result.stdout.strip().splitlines()[-1])
    assert not hit, (
        "these modules are excluded in radar_sim.spec but the application "
        f"actually imports them: {hit}.  The bundle would build and then fail "
        "at runtime."
    )


def test_spec_does_not_use_upx():
    """UPX corrupts Qt DLLs; the spec must keep it off."""
    source = _spec_source()
    assert "upx=True" not in source.replace(" ", ""), \
        "UPX is enabled in the spec -- it is known to corrupt Qt DLLs"


def test_frozen_entry_points_import_statically():
    """PyInstaller cannot follow dynamic imports from an entry script."""
    for name, symbol in (("entry_gui.py", "radar_sim.main"),
                         ("entry_cli.py", "radar_sim.cli")):
        path = PACKAGING / name
        assert path.exists(), f"{name} is missing"
        source = path.read_text(encoding="utf-8")
        assert f"from {symbol} import main" in source, \
            f"{name} must import its entry point statically"
        assert "importlib" not in source, \
            f"{name} uses a dynamic import PyInstaller cannot follow"
        assert "freeze_support" in source, \
            f"{name} is missing multiprocessing.freeze_support()"


def test_usage_text_mentions_the_executable_that_exists():
    """The instructions handed to the user must name a file we actually ship."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "packaging_build", PACKAGING / "build.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["packaging_build"] = module
    spec.loader.exec_module(module)

    assert f"{module.DIR_NAME}.exe" in module.USAGE_TEXT
    assert "运行自检.cmd" in module.USAGE_TEXT
    assert module.DIR_NAME in module.SELFTEST_CMD
