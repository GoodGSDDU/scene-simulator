"""Offscreen acceptance check for the whole D5-D7 GUI chain.

Run::

    QT_QPA_PLATFORM=offscreen python scripts/acceptance.py

Not part of the pytest suite -- this is the human-readable end-to-end
walkthrough: build the window, edit the scene, run the simulation, export a
dataset that the ``templates/`` scripts can then read.
"""

from __future__ import annotations

import pathlib
import sys
import time

# Windows consoles default to GBK and choke on the arrow / geometric glyphs
# used in the button labels.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from PySide6.QtWidgets import QApplication, QDockWidget, QPushButton  # noqa: E402

from radar_sim.main import build_window  # noqa: E402

OUT = pathlib.Path("_verify/gui_out")


def main() -> int:
    # Qt widgets cannot exist without an application object.
    app = QApplication.instance() or QApplication([])

    print("=" * 64)
    print("D5 + D6 + D7  acceptance (offscreen)")
    print("=" * 64)

    window = build_window()
    window.show()
    window.timer.stop()                     # drive _on_tick manually

    print(f"window title    : {window.windowTitle()}")
    print(f"dock count      : {len(window.findChildren(QDockWidget))}")
    labels = [b.text() for b in window.findChildren(QPushButton) if b.text()]
    print(f"buttons         : {labels}")
    print(f"scene prims     : {len(window.simulator.scene.primitives)}")
    cfg = window.simulator.cfg
    print(f"radar cfg       : {cfg.samples_per_chirp} samples / "
          f"{cfg.chirps_per_frame} chirps / {cfg.virtual_elements} virtual elements")

    # ---------------------------------------------------------------- D5
    before = len(window.simulator.scene.primitives)
    window.canvas.add_primitive("circle", (15.0, 15.0))
    after = len(window.simulator.scene.primitives)
    print(f"\n[D5] canvas add  : {before} -> {after}")
    assert after == before + 1, "canvas did not add a primitive"

    prim = window.simulator.scene.primitives[-1]
    window._on_selection_changed(prim)
    old = (prim.params.get("x"), prim.params.get("y"))
    prim.translate(3.0, 4.0)
    window._on_scene_edited(prim.id)
    new = (prim.params.get("x"), prim.params.get("y"))
    print(f"[D5] edit prim   : kind={prim.kind} material={prim.material_id} "
          f"pos {old} -> {new}")
    assert new != old, "editing did not move the primitive"

    # ---------------------------------------------------------------- D6
    window.sim_panel.start_btn.click()
    print(f"\n[D6] start       : running={window._running}")

    deadline = time.time() + 25.0
    while (window.simulator.frame_index < window.simulator.num_frames
           and time.time() < deadline):
        window._on_tick()
        time.sleep(0.004)

    last = window._last_frame
    print(f"[D6] frames done : {window.simulator.frame_index}/{window.simulator.num_frames}")
    print(f"[D6] finished    : {window.simulator.finished}")
    print(f"[D6] last frame  : index={getattr(last, 'index', None)} "
          f"cube_shape={last.result.cube.shape if last is not None else None} "
          f"truth_paths={len(last.result.truth.get('range_m', [])) if last is not None else None}")
    assert window.simulator.frame_index >= 1, "the GUI loop produced no frames"

    print(f"[D6] summary     : {window.simulator.summary()}")

    window.sim_panel.stop_btn.click()

    # ---------------------------------------------------------------- D7
    if OUT.exists():
        for child in OUT.iterdir():
            child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    window.simulator.export(str(OUT))
    produced = sorted(p.name for p in OUT.iterdir())
    print(f"\n[D7] exported    : {produced}")
    assert "signal.bin" in produced
    assert "meta.json" in produced
    assert "pointcloud_gt.csv" in produced

    print("\n" + "=" * 64)
    print("ALL OK")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
