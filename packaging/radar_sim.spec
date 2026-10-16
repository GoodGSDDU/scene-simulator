# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build recipe for the 2D radar scene simulator.

Driven by ``packaging/build.py``; not meant to be invoked by hand.  Behaviour is
switched with environment variables so one spec covers both layouts:

``RADAR_SIM_ONEFILE=1``
    Produce a single self-contained ``.exe``.  Templates, examples and docs are
    embedded in the archive and unpacked to a temp folder at runtime (read-only,
    but ``--selftest`` can still find them).

    otherwise
    Produce a folder containing both executables plus a shared ``_internal``
    directory, with ``templates/``, ``examples/`` and ``docs/`` copied next to
    the executables as plain editable files.

``RADAR_SIM_DIST_NAME``
    Folder (or file) name under ``dist/``.
"""

import os
import sys
from pathlib import Path

ONEFILE = os.environ.get("RADAR_SIM_ONEFILE") == "1"
DIST_NAME = os.environ.get("RADAR_SIM_DIST_NAME", "雷达仿真器")

PACKAGING = Path(SPECPATH).resolve()          # noqa: F821  (injected)
PROJECT = PACKAGING.parent
SRC = PROJECT / "src"
ICON = PACKAGING / "icon.ico"

# --------------------------------------------------------------------------- #
# Trimming.  Every entry here has to be genuinely unreachable.
#
# Lesson learned the hard way: this list originally contained "unittest", which
# matplotlib imports indirectly.  The bundle built fine, the simulation core and
# the templates all worked, and the GUI died with ModuleNotFoundError the moment
# it touched matplotlib -- exactly the "starts fine, breaks on a user action"
# failure mode.  tests/test_packaging.py now verifies that nothing in this list
# appears in the app's real import closure.
#
# Rule of thumb: only exclude whole third-party packages that the app has no
# path to, never stdlib, and never anything on numpy/matplotlib/Qt's import
# graph.
# --------------------------------------------------------------------------- #
EXCLUDES = [
    "tkinter", "_tkinter",             # we force the Qt backend; TkAgg is dead code
    "pandas", "scipy",                 # never imported anywhere in the project
    "IPython", "jupyter", "notebook", "nbformat", "nbconvert", "ipykernel",
    "pytest", "_pytest", "pluggy", "iniconfig",
    "sphinx", "docutils", "babel",
]

# matplotlib selects its Qt binding through a dynamic import, and the package
# layout changed across versions -- list both spellings so either resolves.
HIDDEN = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.qt_compat",
    "matplotlib.backends.qt_editor.figureoptions",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Files that ship beside the executables in onedir mode and inside the archive
# in onefile mode.
SHARED_DATA = [
    (str(PROJECT / "templates"), "templates"),
    (str(PROJECT / "examples"), "examples"),
    (str(PROJECT / "docs"), "docs"),
]


def make_analysis(entry: str, datas=None) -> "Analysis":     # noqa: F821
    return Analysis(                                          # noqa: F821
        [str(PACKAGING / entry)],
        pathex=[str(SRC)],
        binaries=[],
        datas=list(datas or []),
        hiddenimports=HIDDEN,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=EXCLUDES,
        noarchive=False,
        optimize=0,
    )


embed = SHARED_DATA if ONEFILE else []

a_gui = make_analysis("entry_gui.py", embed)
pyz_gui = PYZ(a_gui.pure)                                     # noqa: F821

a_cli = make_analysis("entry_cli.py", embed)
pyz_cli = PYZ(a_cli.pure)                                     # noqa: F821

gui_kwargs = dict(
    name=DIST_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                       # UPX corrupts Qt DLLs more often than it helps
    console=False,                   # double-click must not flash a console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
)

cli_kwargs = dict(
    name="radar-sim-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,                    # the CLI genuinely needs stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
)

if ONEFILE:
    exe_gui = EXE(pyz_gui, a_gui.scripts, a_gui.binaries, a_gui.datas,  # noqa: F821
                  [], exclude_binaries=False, **gui_kwargs)
    exe_cli = EXE(pyz_cli, a_cli.scripts, a_cli.binaries, a_cli.datas,  # noqa: F821
                  [], exclude_binaries=False, **cli_kwargs)
else:
    exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True,    # noqa: F821
                  **gui_kwargs)
    exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True,    # noqa: F821
                  **cli_kwargs)
    COLLECT(                                                            # noqa: F821
        exe_gui, a_gui.binaries, a_gui.datas,
        exe_cli, a_cli.binaries, a_cli.datas,
        strip=False,
        upx=False,
        name=DIST_NAME,
    )
