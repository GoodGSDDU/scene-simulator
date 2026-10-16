"""Frozen-entry point for the command-line executable."""

from __future__ import annotations

import multiprocessing
import sys

from radar_sim.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
