#!/usr/bin/env python3
"""
Custom Adsorbates Example
==========================

This example shows how to add custom adsorbate species beyond
the built-in OER intermediates (*OH, *O, *OOH, *OH2, *OO, *H).

Built-in adsorbates:
- *OH    : Hydroxyl
- *O     : Oxo
- *OOH   : Hydroperoxo
- *OH2   : Adsorbed water
- *OO    : Superoxo
- *H     : Hydrogen
- *      : Empty site

This file shows how to add:
- *CO    : Carbon monoxide
- *COOH  : Carboxyl (CO2RR intermediate)
- *CHO   : Formyl (CO2RR intermediate)
- *NH3   : Ammonia
- *NO    : Nitric oxide
"""

# =============================================================================
# ADSORBATE GEOMETRY FORMAT
# =============================================================================

print("=" * 60)
print("  Adsorbate Geometry Format")
print("=" * 60)

format_explanation = '''
Each adsorbate is defined by a dictionary with:

ADSORBATE_GEOMETRIES["*NAME"] = {
    "atoms": ["Element1", "Element2", ...],  # List of elements
    "offsets": [[x1,y1,z1], [x2,y2,z2], ...], # Positions relative to site
    "height": float,  # Height of first atom above surface (Å)
}

The first atom is placed at (site_x, site_y, site_z + height)
Other atoms are placed at (site_x + offset_x, site_y + offset_y, site_z + height + offset_z)
'''

print(format_explanation)

# =============================================================================
# EXAMPLE: CO2 REDUCTION INTERMEDIATES
# =============================================================================

print("=" * 60)
print("  Example: CO2 Reduction Intermediates")
print("=" * 60)

co2rr_adsorbates = '''
# Carbon monoxide (*CO)
# Linear geometry: C at surface, O pointing up
ADSORBATE_GEOMETRIES["*CO"] = {
    "atoms": ["C", "O"],
    "offsets": [[0, 0, 0], [0, 0, 1.13]],  # C-O bond = 1.13 Å
    "height": 1.85,  # C-surface distance
}

# Carboxyl (*COOH) - CO2RR intermediate
# C bonded to surface, with C=O and C-OH
ADSORBATE_GEOMETRIES["*COOH"] = {
    "atoms": ["C", "O", "O", "H"],
    "offsets": [
        [0.0, 0.0, 0.0],      # C (at surface)
        [1.2, 0.0, 0.3],      # =O (double bonded O)
        [-0.4, 0.0, 1.2],     # -O (hydroxyl O)
        [-0.4, 0.0, 2.2],     # H (on hydroxyl)
    ],
    "height": 1.9,
}

# Formyl (*CHO) - CO2RR intermediate  
ADSORBATE_GEOMETRIES["*CHO"] = {
    "atoms": ["C", "H", "O"],
    "offsets": [
        [0.0, 0.0, 0.0],      # C
        [0.6, 0.6, 0.0],      # H
        [0.0, 0.0, 1.2],      # O
    ],
    "height": 1.85,
}

# Methoxy (*OCH3)
ADSORBATE_GEOMETRIES["*OCH3"] = {
    "atoms": ["O", "C", "H", "H", "H"],
    "offsets": [
        [0.0, 0.0, 0.0],      # O (at surface)
        [0.0, 0.0, 1.4],      # C
        [0.5, 0.9, 1.7],      # H
        [0.5, -0.9, 1.7],     # H
        [-1.0, 0.0, 1.7],     # H
    ],
    "height": 1.7,
}

# Formate (*OCHO) - bidentate
ADSORBATE_GEOMETRIES["*OCHO"] = {
    "atoms": ["O", "C", "O", "H"],
    "offsets": [
        [-0.6, 0.0, 0.0],     # O1 (at surface)
        [0.0, 0.0, 1.0],      # C
        [0.6, 0.0, 0.0],      # O2 (at surface)  
        [0.0, 0.0, 2.1],      # H
    ],
    "height": 1.8,
}
'''

print(co2rr_adsorbates)

# =============================================================================
# EXAMPLE: NITROGEN SPECIES
# =============================================================================

print("=" * 60)
print("  Example: Nitrogen Species (NRR)")
print("=" * 60)

nrr_adsorbates = '''
# Ammonia (*NH3)
ADSORBATE_GEOMETRIES["*NH3"] = {
    "atoms": ["N", "H", "H", "H"],
    "offsets": [
        [0.0, 0.0, 0.0],          # N
        [0.5, 0.8, 0.4],          # H
        [0.5, -0.8, 0.4],         # H
        [-0.9, 0.0, 0.4],         # H
    ],
    "height": 2.0,
}

# Nitric oxide (*NO)
ADSORBATE_GEOMETRIES["*NO"] = {
    "atoms": ["N", "O"],
    "offsets": [[0, 0, 0], [0, 0, 1.15]],
    "height": 1.8,
}

# Dinitrogen (*N2) - end-on
ADSORBATE_GEOMETRIES["*N2"] = {
    "atoms": ["N", "N"],
    "offsets": [[0, 0, 0], [0, 0, 1.10]],
    "height": 1.9,
}

# Hydrazine (*N2H4)
ADSORBATE_GEOMETRIES["*N2H4"] = {
    "atoms": ["N", "N", "H", "H", "H", "H"],
    "offsets": [
        [0.0, 0.0, 0.0],          # N1
        [0.0, 0.0, 1.4],          # N2
        [0.5, 0.8, -0.3],         # H on N1
        [-0.9, 0.0, -0.3],        # H on N1
        [0.5, 0.8, 1.7],          # H on N2
        [-0.9, 0.0, 1.7],         # H on N2
    ],
    "height": 2.0,
}

# NNH (*NNH) - intermediate
ADSORBATE_GEOMETRIES["*NNH"] = {
    "atoms": ["N", "N", "H"],
    "offsets": [
        [0.0, 0.0, 0.0],          # N1 (at surface)
        [0.0, 0.0, 1.2],          # N2
        [0.5, 0.5, 1.6],          # H
    ],
    "height": 1.85,
}
'''

print(nrr_adsorbates)

# =============================================================================
# COMPLETE EXAMPLE: USING CUSTOM ADSORBATES
# =============================================================================

print("=" * 60)
print("  Complete Example: Using Custom Adsorbates")
print("=" * 60)

complete_example = '''
#!/usr/bin/env python3
"""Use custom adsorbates for CO2RR study."""

import sys
sys.path.insert(0, '..')

from electrode_solvation import (
    ADSORBATE_GEOMETRIES, AdsorbateConfig, SlabConfig,
    create_iro2_slab, add_adsorbates
)
from ase.io import write

# Step 1: Add custom adsorbates
ADSORBATE_GEOMETRIES["*CO"] = {
    "atoms": ["C", "O"],
    "offsets": [[0, 0, 0], [0, 0, 1.13]],
    "height": 1.85,
}

ADSORBATE_GEOMETRIES["*COOH"] = {
    "atoms": ["C", "O", "O", "H"],
    "offsets": [[0,0,0], [1.2,0,0.3], [-0.4,0,1.2], [-0.4,0,2.2]],
    "height": 1.9,
}

# Step 2: Configure coverage
ADSORBATES = AdsorbateConfig(
    coverage={
        "*CO": 2,      # 2 CO molecules
        "*COOH": 1,    # 1 COOH
        "*OH": 2,      # 2 OH (from OER)
    }
)

# Step 3: Create slab and add adsorbates
SLAB = SlabConfig()
slab, site_info, Lx, Ly = create_iro2_slab(SLAB)
slab = add_adsorbates(slab, site_info, ADSORBATES)

write("slab_with_co2rr_intermediates.cif", slab)
print(f"Created slab with custom adsorbates: {len(slab)} atoms")
'''

print(complete_example)

# =============================================================================
# TIPS FOR ADSORBATE GEOMETRIES
# =============================================================================

print("=" * 60)
print("  Tips for Creating Adsorbate Geometries")
print("=" * 60)

tips = '''
1. BOND LENGTHS (approximate values in Å):
   - C-H: 1.09
   - N-H: 1.01
   - O-H: 0.96
   - C-C: 1.54 (single), 1.34 (double), 1.20 (triple)
   - C-N: 1.47 (single), 1.27 (double), 1.16 (triple)
   - C-O: 1.43 (single), 1.23 (double), 1.13 (CO)
   - N-N: 1.45 (single), 1.25 (double), 1.10 (triple)
   - N-O: 1.40 (single), 1.21 (double), 1.15 (NO)

2. SURFACE-ADSORBATE DISTANCES (height parameter):
   - C on metal: 1.8-2.0 Å
   - N on metal: 1.8-2.0 Å
   - O on metal: 1.7-1.9 Å
   - H on metal: 1.5-1.8 Å
   - On oxide: add ~0.1-0.2 Å

3. USE DFT-OPTIMIZED GEOMETRIES:
   For production work, optimize adsorbate geometries with DFT first,
   then extract coordinates for initial structure generation.

4. BIDENTATE ADSORBATES:
   For species that bind through two atoms (like formate *OCHO),
   adjust both atoms to be at similar heights.

5. TESTING:
   Always visualize the structure after adding adsorbates to verify
   the geometry looks reasonable before running expensive calculations.
'''

print(tips)
