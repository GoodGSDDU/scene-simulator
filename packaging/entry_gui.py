"""Frozen-entry point for the GUI executable.

PyInstaller analyses this file, so it must import the real entry point
statically rather than looking it up at runtime.
"""

from __future__ import annotations

import multiprocessing
import sys

from radar_sim.main import main

if __name__ == "__main__":
    # Required for frozen apps that spawn processes (multiprocessing on Windows
    # re-executes the binary).  Cheap insurance, harmless if unused.
    multiprocessing.freeze_support()
    sys.exit(main())
