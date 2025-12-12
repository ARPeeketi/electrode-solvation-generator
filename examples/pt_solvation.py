#!/usr/bin/env python3
"""
Pt(111) Solvation Example
==========================

This example shows how to adapt the electrode solvation generator
for FCC metals like Pt, Au, Pd, Ag.

The main changes from the oxide version:
1. Different bulk structure (FCC instead of rutile)
2. No oxygen sites to identify
3. Simpler adsorbate placement on metal sites
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.io import write
from electrode_solvation import (
    WaterConfig, IonConfig, SolvationConfig, LAMMPSConfig,
    create_ice_box, add_ions_to_water, write_lammps_data,
    write_lammps_equilibration_input, run_lammps,
    read_lammps_data, fix_z_boundary_molecules, compute_density
)


def create_pt111_slab(size=(4, 4), layers=5, vacuum=0.0):
    """Create Pt(111) slab.
    
    Args:
        size: (nx, ny) supercell size
        layers: Number of Pt layers
        vacuum: Vacuum above/below slab (Å)
    
    Returns:
        Atoms object with Pt(111) slab
    """
    # Pt lattice constant
    a_pt = 3.92  # Å (experimental value)
    
    # Create Pt(111) surface
    slab = fcc111('Pt', size=(size[0], size[1], layers), a=a_pt, vacuum=vacuum)
    
    # Get cell dimensions
    cell = slab.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]
    
    # Shift to have z=0 at bottom
    pos = slab.get_positions()
    pos[:, 2] -= pos[:, 2].min()
    slab.set_positions(pos)
    
    print(f"  Pt(111): {len(slab)} atoms, {Lx:.2f}×{Ly:.2f} Å")
    print(f"  Layers: {layers}, Size: {size[0]}×{size[1]}")
    
    return slab, Lx, Ly


def add_co_adsorbates(slab, n_co=4):
    """Add CO adsorbates to Pt surface (example).
    
    Args:
        slab: Pt slab Atoms object
        n_co: Number of CO molecules to add
    """
    # Get top layer Pt positions
    pos = slab.get_positions()
    z_max = pos[:, 2].max()
    
    # Find top Pt atoms
    top_pt = [i for i, p in enumerate(pos) if p[2] > z_max - 0.5]
    
    if n_co > len(top_pt):
        n_co = len(top_pt)
    
    # Add CO at random top sites
    import random
    random.shuffle(top_pt)
    
    new_atoms = []
    new_pos = []
    
    for i in range(n_co):
        pt_pos = pos[top_pt[i]]
        # CO geometry: C at 1.85 Å, O at 1.85 + 1.13 = 2.98 Å
        c_pos = pt_pos + np.array([0, 0, 1.85])
        o_pos = pt_pos + np.array([0, 0, 2.98])
        new_atoms.extend(['C', 'O'])
        new_pos.extend([c_pos, o_pos])
    
    if new_atoms:
        combined_sym = list(slab.get_chemical_symbols()) + new_atoms
        combined_pos = list(slab.get_positions()) + new_pos
        slab = Atoms(combined_sym, combined_pos, cell=slab.get_cell(), pbc=slab.pbc)
    
    print(f"  Added {n_co} CO adsorbates")
    return slab


def solvate_metal_slab(slab, water, wcell, water_gap=2.0, overlap_cutoff=2.5):
    """Combine metal slab with water box.
    
    For metals, we use slightly larger gap since there's no H-bonding
    to surface oxygen (unlike oxides).
    """
    slab_pos = slab.get_positions()
    slab_sym = slab.get_chemical_symbols()
    slab_z_max = slab_pos[:, 2].max()
    
    water_pos = water.get_positions().copy()
    water_sym = list(water.get_chemical_symbols())
    
    # Shift water above slab
    water_pos[:, 2] += slab_z_max + water_gap
    
    Lx, Ly = wcell[0], wcell[1]
    water_z_max = water_pos[:, 2].max()
    new_Lz = water_z_max + 5.0
    
    # Remove overlapping water
    keep_indices = []
    for i, wpos in enumerate(water_pos):
        overlap = False
        for slab_p in slab_pos:
            d = np.linalg.norm(wpos - slab_p)
            if d < overlap_cutoff:
                overlap = True
                break
        if not overlap:
            keep_indices.append(i)
    
    keep_water_sym = [water_sym[i] for i in keep_indices]
    keep_water_pos = [water_pos[i] for i in keep_indices]
    
    combined_sym = list(slab_sym) + keep_water_sym
    combined_pos = list(slab_pos) + list(keep_water_pos)
    
    combined = Atoms(combined_sym, combined_pos,
                     cell=[Lx, Ly, new_Lz], pbc=[True, True, True])
    
    removed = len(water_sym) - len(keep_water_sym)
    print(f"  Removed {removed} overlapping water atoms")
    
    return combined


def main():
    """Generate Pt(111) + water interface."""
    
    print("=" * 60)
    print("  Pt(111) + Water Interface Generator")
    print("=" * 60)
    
    # Configuration
    WATER = WaterConfig(
        height=25.0,
        target_density=1.0,
    )
    
    IONS = IonConfig(
        concentration=0.1,
        cation="Na",
        anion="ClO4",
    )
    
    LAMMPS = LAMMPSConfig(
        lammps="lmp",  # Adjust to your LAMMPS path
        np=4,
    )
    
    os.makedirs(LAMMPS.tmp_dir, exist_ok=True)
    
    # Step 1: Create Pt slab
    print("\n[1/5] Creating Pt(111) slab...")
    slab, Lx, Ly = create_pt111_slab(size=(4, 4), layers=5)
    
    # Step 2: Add adsorbates (optional)
    print("\n[2/5] Adding CO adsorbates...")
    slab = add_co_adsorbates(slab, n_co=2)
    write("01_pt_slab.cif", slab)
    
    # Step 3: Create water box
    print("\n[3/5] Creating water box...")
    water_box, initial_dims, target_dims = create_ice_box(Lx, Ly, WATER.height, WATER)
    
    Lx_i, Ly_i, Lz_i = initial_dims
    from electrode_solvation import calculate_ion_pairs
    n_ions = calculate_ion_pairs(Lx_i, Ly_i, Lz_i, IONS.concentration)
    water_box = add_ions_to_water(water_box, n_ions, IONS)
    
    write("02_water_initial.cif", water_box)
    
    # Step 4: Run LAMMPS
    print("\n[4/5] Running LAMMPS equilibration...")
    data_file = os.path.join(LAMMPS.tmp_dir, "water_initial.data")
    write_lammps_data(water_box, data_file, IONS)
    
    input_file = os.path.join(LAMMPS.tmp_dir, "in.equil")
    write_lammps_equilibration_input(
        input_file, "water_initial.data", WATER, IONS, initial_dims, target_dims
    )
    
    run_lammps(LAMMPS, "in.equil")
    
    final_data = os.path.join(LAMMPS.tmp_dir, "water_final.data")
    water_eq, wcell = read_lammps_data(final_data)
    water_eq = fix_z_boundary_molecules(water_eq)
    
    print(f"  Final density: {compute_density(water_eq):.3f} g/cc")
    
    # Step 5: Solvate
    print("\n[5/5] Solvating Pt slab...")
    combined = solvate_metal_slab(slab, water_eq, wcell, 
                                   water_gap=2.0,      # Larger gap for metal
                                   overlap_cutoff=2.5)  # Larger cutoff for metal
    
    write("Pt111_waterbox.cif", combined)
    
    print("\n" + "=" * 60)
    print("  ✅ Complete!")
    print("=" * 60)
    print(f"\n  Output: Pt111_waterbox.cif ({len(combined)} atoms)")


if __name__ == "__main__":
    main()
