"""Render the main window offscreen and save a PNG.

    QT_QPA_PLATFORM=offscreen python scripts/screenshot.py out.png

Useful for documentation and for eyeballing the layout without a display.
Optional first argument is the output path (default: doc/main_window.png).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from radar_sim.main import build_window  # noqa: E402


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("doc/main_window.png")
    target.parent.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    window = build_window()
    window.resize(1600, 980)
    window.show()
    window.timer.stop()

    # let the layout settle, then produce one frame so the plot panels have data
    for _ in range(8):
        app.processEvents()

    window.sim_panel.start_btn.click()
    window._on_tick()
    window._on_tick()
    for _ in range(8):
        app.processEvents()
    window.sim_panel.stop_btn.click()
    for _ in range(8):
        app.processEvents()

    pixmap = window.grab()
    ok = pixmap.save(str(target), "PNG")
    print(f"{'wrote' if ok else 'FAILED'} {target}  {pixmap.width()}x{pixmap.height()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
