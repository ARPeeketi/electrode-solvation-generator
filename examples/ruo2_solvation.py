#!/usr/bin/env python3
"""
RuO2 Solvation Example
=======================

RuO2 has the same rutile structure as IrO2, so adaptation is simple:
just change the lattice parameters.

Rutile oxide lattice parameters:
- IrO2: a = 4.50 Å, c = 3.20 Å
- RuO2: a = 4.49 Å, c = 3.11 Å
- TiO2: a = 4.59 Å, c = 2.96 Å
- SnO2: a = 4.74 Å, c = 3.19 Å
- MnO2: a = 4.40 Å, c = 2.87 Å
"""

import subprocess
import sys
import os

# =============================================================================
# METHOD 1: Modify the script directly
# =============================================================================

print("=" * 60)
print("  Method 1: Modify SlabConfig in electrode_solvation.py")
print("=" * 60)

config_example = '''
# In electrode_solvation.py, change SlabConfig defaults:

@dataclass
class SlabConfig:
    facet: Tuple[int, int, int] = (1, 1, 0)
    n_layers: int = 5
    vacuum: float = 0.0
    repeat_x: int = 4
    repeat_y: int = 4
    # Change these for RuO2:
    a_lattice: float = 4.49   # was 4.50 for IrO2
    c_lattice: float = 3.11   # was 3.20 for IrO2
'''

print(config_example)

# =============================================================================
# METHOD 2: Create a wrapper script
# =============================================================================

print("=" * 60)
print("  Method 2: Create a RuO2-specific script")
print("=" * 60)

wrapper_example = '''
#!/usr/bin/env python3
"""RuO2 solvation wrapper."""
import sys
sys.path.insert(0, '..')

# Import and modify config
from electrode_solvation import SlabConfig, main

# Override lattice parameters for RuO2
SlabConfig.a_lattice = 4.49
SlabConfig.c_lattice = 3.11

# Run with modified parameters
if __name__ == "__main__":
    main()
'''

print(wrapper_example)

# =============================================================================
# METHOD 3: Full custom script (most flexible)
# =============================================================================

print("=" * 60)
print("  Method 3: Full custom script")
print("=" * 60)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FULL_SCRIPT = '''
#!/usr/bin/env python3
"""
Complete RuO2(110) Solvation Script
"""

import numpy as np
import os
import sys
import time

sys.path.insert(0, '..')

from ase import Atoms
from ase.build import surface, make_supercell
from ase.io import write

from electrode_solvation import (
    WaterConfig, IonConfig, SolvationConfig, LAMMPSConfig, AdsorbateConfig,
    create_ice_box, add_ions_to_water, write_lammps_data,
    write_lammps_equilibration_input, run_lammps,
    read_lammps_data, fix_z_boundary_molecules, 
    compute_density, identify_surface_sites, add_adsorbates,
    solvate_slab, count_waters, calculate_ion_pairs
)


def create_ruo2_bulk(a=4.49, c=3.11):
    """Create RuO2 rutile bulk structure.
    
    Same structure as IrO2, just different lattice parameters.
    """
    u = 0.305  # Oxygen parameter (similar to IrO2)
    
    positions = [
        [0, 0, 0],
        [0.5*a, 0.5*a, 0.5*c],
        [u*a, u*a, 0],
        [(1-u)*a, (1-u)*a, 0],
        [(0.5+u)*a, (0.5-u)*a, 0.5*c],
        [(0.5-u)*a, (0.5+u)*a, 0.5*c],
    ]
    
    bulk = Atoms(
        symbols=['Ru', 'Ru', 'O', 'O', 'O', 'O'],
        positions=positions,
        cell=[a, a, c],
        pbc=True
    )
    return bulk


def create_ruo2_slab(facet=(1,1,0), n_layers=5, repeat=(4,4)):
    """Create RuO2 slab."""
    bulk = create_ruo2_bulk()
    slab = surface(bulk, facet, n_layers, vacuum=0.0)
    
    if repeat[0] > 1 or repeat[1] > 1:
        slab = make_supercell(slab, [[repeat[0],0,0], [0,repeat[1],0], [0,0,1]])
    
    pos = slab.get_positions()
    pos[:, 2] -= pos[:, 2].min()
    slab.set_positions(pos)
    
    cell = slab.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]
    
    site_info = identify_surface_sites(slab)
    
    facet_str = f"{facet[0]}{facet[1]}{facet[2]}"
    print(f"  RuO2({facet_str}): {len(slab)} atoms, {Lx:.2f}×{Ly:.2f} Å")
    print(f"  Surface sites: μ1={len(site_info['mu1'])}, μ2={len(site_info['mu2'])}")
    
    return slab, site_info, Lx, Ly


def main():
    """Generate RuO2(110) + water interface."""
    
    start_time = time.time()
    
    print("=" * 60)
    print("  RuO2(110) + Water Interface Generator")
    print("=" * 60)
    
    # Configuration
    WATER = WaterConfig(height=25.0)
    IONS = IonConfig(concentration=0.1, cation="Na", anion="ClO4")
    SOLVATION = SolvationConfig()
    ADSORBATES = AdsorbateConfig(coverage={"*OH": 3, "*OOH": 1, "*OH2": 2})
    LAMMPS = LAMMPSConfig(lammps="lmp", np=4)
    
    os.makedirs(LAMMPS.tmp_dir, exist_ok=True)
    
    # Step 1: Create RuO2 slab
    print("\\n[1/6] Creating RuO2 slab...")
    slab, site_info, Lx, Ly = create_ruo2_slab(
        facet=(1, 1, 0),
        n_layers=5,
        repeat=(4, 4)
    )
    
    # Step 2: Add adsorbates
    print("\\n[2/6] Adding adsorbates...")
    slab = add_adsorbates(slab, site_info, ADSORBATES)
    write("01_ruo2_slab.cif", slab)
    
    # Step 3: Create water box
    print("\\n[3/6] Creating water box...")
    water_box, initial_dims, target_dims = create_ice_box(Lx, Ly, WATER.height, WATER)
    
    Lx_i, Ly_i, Lz_i = initial_dims
    n_ions = calculate_ion_pairs(Lx_i, Ly_i, Lz_i, IONS.concentration)
    water_box = add_ions_to_water(water_box, n_ions, IONS)
    
    write("02_water_initial.cif", water_box)
    
    # Step 4: LAMMPS files
    print("\\n[4/6] Preparing LAMMPS...")
    data_file = os.path.join(LAMMPS.tmp_dir, "water_initial.data")
    write_lammps_data(water_box, data_file, IONS)
    
    input_file = os.path.join(LAMMPS.tmp_dir, "in.equil")
    write_lammps_equilibration_input(
        input_file, "water_initial.data", WATER, IONS, initial_dims, target_dims
    )
    
    # Step 5: Run LAMMPS
    print("\\n[5/6] Running LAMMPS...")
    run_lammps(LAMMPS, "in.equil")
    
    final_data = os.path.join(LAMMPS.tmp_dir, "water_final.data")
    water_eq, wcell = read_lammps_data(final_data)
    water_eq = fix_z_boundary_molecules(water_eq)
    
    print(f"  Final density: {compute_density(water_eq):.3f} g/cc")
    
    # Step 6: Solvate
    print("\\n[6/6] Solvating slab...")
    combined = solvate_slab(slab, water_eq, wcell, SOLVATION, site_info)
    
    write("RuO2_110_waterbox.cif", combined)
    write_lammps_data(combined, "RuO2_110_waterbox.data", IONS)
    
    # Summary
    print("\\n" + "=" * 60)
    print(f"  ✅ Complete in {(time.time()-start_time)/60:.1f} min")
    print("=" * 60)
    print(f"\\n  Output: RuO2_110_waterbox.cif ({len(combined)} atoms)")


if __name__ == "__main__":
    main()
'''

print(FULL_SCRIPT)

# =============================================================================
# LATTICE PARAMETERS REFERENCE
# =============================================================================

print("\n" + "=" * 60)
print("  Rutile Oxide Lattice Parameters Reference")
print("=" * 60)

params = """
| Material | a (Å) | c (Å) | Notes |
|----------|-------|-------|-------|
| IrO2     | 4.50  | 3.20  | OER catalyst |
| RuO2     | 4.49  | 3.11  | OER catalyst |
| TiO2     | 4.59  | 2.96  | Photocatalyst |
| SnO2     | 4.74  | 3.19  | Gas sensor |
| MnO2     | 4.40  | 2.87  | Battery |
| GeO2     | 4.40  | 2.86  | |
| PbO2     | 4.95  | 3.38  | Lead-acid battery |
| VO2      | 4.55  | 2.85  | Phase change |
| CrO2     | 4.42  | 2.92  | Magnetic |
| NbO2     | 4.77  | 2.96  | |
"""

print(params)
