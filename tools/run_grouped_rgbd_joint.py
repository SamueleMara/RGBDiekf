#!/usr/bin/env python3
"""Compatibility command for the corrected free-point grouped pipeline.

The superseded persistent-supernode estimator is not imported or executed.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_free_grouped_rgbd_joint import main

if __name__=='__main__':raise SystemExit(main())
