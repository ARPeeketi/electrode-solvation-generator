#!/usr/bin/env python3
"""
Different Ions Example
=======================

This example shows how to use different electrolytes and how to add
new ion types to the force field.

Supported out-of-the-box:
- Cations: Na+, K+, H3O+
- Anions: ClO4-, Cl-

This file shows how to add: Li+, Cs+, Mg2+, Ca2+, Br-, I-, SO4^2-
"""

import subprocess
import sys
import os

# =============================================================================
# USING BUILT-IN IONS
# =============================================================================

print("=" * 60)
print("  Built-in Electrolyte Examples")
print("=" * 60)

examples = [
    # (concentration, cation, anion, description)
    (0.1, "Na", "ClO4", "Standard NaClO4 (OER experiments)"),
    (0.5, "Na", "Cl", "NaCl (seawater-like)"),
    (0.1, "K", "ClO4", "KClO4 (alternative supporting electrolyte)"),
    (0.1, "H3O", "ClO4", "Acidic HClO4 (low pH OER)"),
]

print("\nRun these from command line:\n")
for conc, cat, an, desc in examples:
    print(f"# {desc}")
    print(f"python electrode_solvation.py --concentration {conc} --cation {cat} --anion {an}")
    print()


# =============================================================================
# ADDING NEW IONS
# =============================================================================

print("=" * 60)
print("  Adding New Ion Types")
print("=" * 60)

# To add new ions, modify the ION_FF dictionary in electrode_solvation.py
# Here are parameters for common ions:

NEW_ION_PARAMETERS = """
# Alkali metals (Joung-Cheatham parameters for TIP3P water)
ION_FF["Li"] = {"mass":   6.941, "charge":  1.0, "sigma": 1.582, "epsilon": 0.118}
ION_FF["Cs"] = {"mass": 132.905, "charge":  1.0, "sigma": 3.883, "epsilon": 0.047}
ION_FF["Rb"] = {"mass":  85.468, "charge":  1.0, "sigma": 3.476, "epsilon": 0.057}

# Alkaline earth metals
ION_FF["Mg"] = {"mass":  24.305, "charge":  2.0, "sigma": 1.644, "epsilon": 0.875}
ION_FF["Ca"] = {"mass":  40.078, "charge":  2.0, "sigma": 2.412, "epsilon": 0.449}
ION_FF["Sr"] = {"mass":  87.620, "charge":  2.0, "sigma": 2.973, "epsilon": 0.118}
ION_FF["Ba"] = {"mass": 137.327, "charge":  2.0, "sigma": 3.380, "epsilon": 0.047}

# Transition metals (simplified - use with caution)
ION_FF["Zn"] = {"mass":  65.380, "charge":  2.0, "sigma": 1.948, "epsilon": 0.250}
ION_FF["Fe2"] = {"mass": 55.845, "charge":  2.0, "sigma": 2.000, "epsilon": 0.200}
ION_FF["Fe3"] = {"mass": 55.845, "charge":  3.0, "sigma": 1.900, "epsilon": 0.300}

# Halides
ION_FF["F"]  = {"mass": 18.998, "charge": -1.0, "sigma": 3.118, "epsilon": 0.710}
ION_FF["Br"] = {"mass": 79.904, "charge": -1.0, "sigma": 4.624, "epsilon": 0.080}
ION_FF["I"]  = {"mass": 126.90, "charge": -1.0, "sigma": 5.167, "epsilon": 0.070}

# Hydroxide (for alkaline solutions)
ION_FF["OH"] = {"mass": 17.007, "charge": -1.0, "sigma": 3.188, "epsilon": 0.102}
"""

print("\nAdd these to ION_FF dictionary in electrode_solvation.py:\n")
print(NEW_ION_PARAMETERS)


# =============================================================================
# POLYATOMIC IONS
# =============================================================================

print("=" * 60)
print("  Polyatomic Ions (SO4^2-, NO3-, etc.)")
print("=" * 60)

POLYATOMIC_EXAMPLE = '''
# For polyatomic ions like SO4^2-, you need to:
# 1. Define the geometry
# 2. Add multiple atom types
# 3. Modify add_ions_to_water() to handle the geometry

# Example: Sulfate (SO4^2-)
# Tetrahedral geometry with S at center

def add_sulfate(water_pos, center_pos):
    """Add SO4^2- at given position."""
    d_so = 1.49  # S-O bond length (Å)
    
    # Tetrahedral positions
    offsets = [
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ]
    
    atoms = ['S']
    positions = [center_pos]
    
    for dx, dy, dz in offsets:
        o_pos = center_pos + d_so * np.array([dx, dy, dz]) / np.sqrt(3)
        atoms.append('O')
        positions.append(o_pos)
    
    return atoms, positions

# Charges for SO4^2-:
# S: +1.20
# O: -0.80 each (4 × -0.80 = -3.20)
# Total: +1.20 - 3.20 = -2.00
'''

print(POLYATOMIC_EXAMPLE)


# =============================================================================
# COMPLETE EXAMPLE: Li+ Cl- Electrolyte
# =============================================================================

print("=" * 60)
print("  Complete Example: Modifying for LiCl")
print("=" * 60)

LICL_EXAMPLE = '''
# Step 1: Add Li to ION_FF in electrode_solvation.py
ION_FF["Li"] = {"mass": 6.941, "charge": 1.0, "sigma": 1.582, "epsilon": 0.118}

# Step 2: Update command line choices (in argparse section)
parser.add_argument('--cation', type=str, default='Na',
                    choices=['Na', 'K', 'H3O', 'Li', 'Cs'],  # Add Li, Cs
                    help='Cation species')

# Step 3: Run
python electrode_solvation.py --cation Li --anion Cl --concentration 1.0
'''

print(LICL_EXAMPLE)


# =============================================================================
# MIXED ELECTROLYTES
# =============================================================================

print("=" * 60)
print("  Mixed Electrolytes (Advanced)")
print("=" * 60)

MIXED_EXAMPLE = '''
# For mixed electrolytes (e.g., 0.1 M NaCl + 0.05 M KCl), you need to:
# 1. Calculate number of each ion pair separately
# 2. Modify add_ions_to_water() to handle multiple species

def add_mixed_ions(water, ion_configs):
    """
    ion_configs = [
        {"cation": "Na", "anion": "Cl", "n_pairs": 5},
        {"cation": "K", "anion": "Cl", "n_pairs": 3},
    ]
    """
    for config in ion_configs:
        water = add_single_ion_type(water, config)
    return water
'''

print(MIXED_EXAMPLE)

print("\n" + "=" * 60)
print("  See electrode_solvation.py for full implementation")
print("=" * 60)
