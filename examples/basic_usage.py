#!/usr/bin/env python3
"""
Basic Usage Example
====================

Generate IrO2(110) + water interface with default settings.

This is the simplest way to use the electrode solvation generator.
"""

import subprocess
import sys

# Run with default settings
# This creates:
#   - IrO2_110_waterbox.cif
#   - IrO2_110_waterbox.data

subprocess.run([sys.executable, "../electrode_solvation.py"])

# Or run from command line:
# python electrode_solvation.py
