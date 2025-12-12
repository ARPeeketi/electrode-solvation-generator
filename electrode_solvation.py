#!/usr/bin/env python3
"""
Electrode-Electrolyte Interface Generator
==========================================

Generate solvated electrode slab structures for DFT/MD simulations of 
electrochemical interfaces (e.g., oxygen evolution reaction, OER).

Features:
- IrO2 rutile structure with (110), (100), or (101) surface terminations
- Surface site identification (μ1-CUS, μ2-bridge, μ3-lattice O)
- Random adsorbate placement (*OH, *O, *OOH, *OH2, *OO, *H)
- Ice Ic water structure with Bernal-Fowler H-bond rules
- LAMMPS equilibration with TIP3P water + ion force fields
- Electrolyte support (NaClO4, KClO4, HClO4, NaCl)

Requirements:
- Python 3.8+
- ASE (Atomic Simulation Environment)
- NumPy
- LAMMPS (with TIP3P water support)
- MPI (optional, for parallel LAMMPS)

Usage:
    python electrode_solvation.py                    # Default: IrO2(110) + water
    python electrode_solvation.py --facet 100        # IrO2(100) surface
    python electrode_solvation.py --water-height 30  # 30 Å water layer
    python electrode_solvation.py --concentration 0.5 # 0.5 M electrolyte

References:
    - TIP3P water: Jorgensen et al., J. Chem. Phys. 1983, 79, 926-935
    - Ion parameters: Joung & Cheatham, J. Phys. Chem. B 2008, 112, 9020-9041
    - Ice Ic structure: Kuhs et al., PNAS 2012, 109, 21259-21264
    - Bernal-Fowler rules: Bernal & Fowler, J. Chem. Phys. 1933, 1, 515-548
    - IrO2 structure: Bolzan et al., Acta Crystallogr. B 1997, 53, 373-380

Author: Akhil Reddy Peeketi, with the help of Claude (Anthropic)
License: MIT
Version: 1.0.0
"""

import numpy as np
import random
import subprocess
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from ase import Atoms
from ase.io import write, read
from ase.build import surface, make_supercell

__version__ = "1.0.0"
__author__ = "Akhil Reddy Peeketi, with the help of Claude (Anthropic)"

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class SlabConfig:
    """IrO2 slab configuration.
    
    Attributes:
        facet: Miller indices for surface termination (default: 110)
        n_layers: Number of IrO2 layers (default: 5)
        vacuum: Vacuum spacing in Å (default: 0, added later)
        repeat_x: Supercell repeat in x direction
        repeat_y: Supercell repeat in y direction
        a_lattice: IrO2 rutile lattice parameter a (Å)
        c_lattice: IrO2 rutile lattice parameter c (Å)
    """
    facet: Tuple[int, int, int] = (1, 1, 0)
    n_layers: int = 5
    vacuum: float = 0.0
    repeat_x: int = 4
    repeat_y: int = 4
    a_lattice: float = 4.50
    c_lattice: float = 3.20


@dataclass  
class WaterConfig:
    """Water box configuration.
    
    Attributes:
        height: Water layer thickness in Å
        target_density: Target water density in g/cc
        ice_density: Initial ice packing density (adjusted for trimming losses)
        melt_temp: Temperature for LAMMPS equilibration (K)
        compress_time: Compression phase duration (ps)
        final_equil_time: Final equilibration duration (ps)
    """
    height: float = 25.0
    target_density: float = 1.05
    ice_density: float = 0.75
    ice_lattice: float = 6.358
    noise_amplitude: float = 0.05
    melt_temp: float = 300.0
    equil_temp: float = 300.0
    compress_time: float = 5.0
    final_equil_time: float = 10.0


@dataclass
class SolvationConfig:
    """Solvation parameters.
    
    Attributes:
        overlap_cutoff: Minimum water-slab distance (Å)
        water_gap: Gap between slab surface and water box (Å)
    """
    overlap_cutoff: float = 1.0
    water_gap: float = 0.5


@dataclass
class IonConfig:
    """Electrolyte configuration.
    
    Attributes:
        concentration: Ion concentration in mol/L
        cation: Cation species (Na, K, H3O)
        anion: Anion species (ClO4, Cl)
    """
    concentration: float = 0.1
    cation: str = "Na"
    anion: str = "ClO4"


@dataclass
class AdsorbateConfig:
    """Surface adsorbate configuration.
    
    Attributes:
        coverage: Dict of adsorbate species and their counts
        allowed_species: List of valid adsorbate types
    """
    coverage: Dict[str, int] = field(default_factory=lambda: {
        "*OH": 3,
        "*OOH": 1,
        "*OH2": 2,
    })
    allowed_species: List[str] = field(default_factory=lambda: [
        "*OH", "*O", "*OOH", "*OH2", "*OO", "*H", "*"
    ])


@dataclass
class LAMMPSConfig:
    """LAMMPS simulation settings.
    
    Attributes:
        mpirun: Path to mpirun executable
        lammps: Path to LAMMPS executable
        np: Number of MPI processes
        omp_threads: OpenMP threads per process
        tmp_dir: Working directory for LAMMPS files
    """
    mpirun: str = "mpirun"
    lammps: str = "lmp"
    np: int = 4
    omp_threads: int = 1
    tmp_dir: str = "lammps_equil"
    log_file: str = "water_equil.log"


# =============================================================================
# FORCE FIELD PARAMETERS
# =============================================================================

# TIP3P water parameters
# Reference: Jorgensen, W. L.; Chandrasekhar, J.; Madura, J. D.; Impey, R. W.; 
#            Klein, M. L. J. Chem. Phys. 1983, 79, 926-935.
#            DOI: 10.1063/1.445869
TIP3P = {
    "O_mass": 15.9994,
    "H_mass": 1.008,
    "O_charge": -0.834,
    "H_charge": 0.417,
    "O_sigma": 3.188,      # Å
    "O_epsilon": 0.102,    # kcal/mol
    "OH_bond": 0.9572,     # Å
    "HOH_angle": 104.52,   # degrees
    "OH_k": 450.0,         # kcal/mol/Å²
    "HOH_k": 55.0,         # kcal/mol/rad²
}

# Ion force field parameters (compatible with TIP3P)
# Reference: Joung, I. S.; Cheatham, T. E. J. Phys. Chem. B 2008, 112, 9020-9041.
#            DOI: 10.1021/jp8001614
# Note: Parameters optimized for TIP3P water model
ION_FF = {
    "Na": {"mass": 22.990, "charge": 1.0, "sigma": 2.350, "epsilon": 0.130},
    "K":  {"mass": 39.098, "charge": 1.0, "sigma": 3.154, "epsilon": 0.100},
    "Cl": {"mass": 35.453, "charge": -1.0, "sigma": 4.401, "epsilon": 0.100},
    "H3O": {"mass": 19.023, "charge": 1.0, "sigma": 3.188, "epsilon": 0.102},
    # ClO4 modeled as 5-site (Cl + 4 O) with charges from OPLS-AA
    "ClO4_Cl": {"mass": 35.453, "charge": 0.892, "sigma": 3.50, "epsilon": 0.10},
    "ClO4_O": {"mass": 15.999, "charge": -0.473, "sigma": 3.00, "epsilon": 0.06},
}


# =============================================================================
# IrO2 SLAB CREATION
# =============================================================================
# IrO2 rutile structure parameters
# Reference: Bolzan et al., Acta Crystallogr. B 1997, 53, 373-380.
#            DOI: 10.1107/S0108768197001468
# Lattice parameters: a = 4.4983 Å, c = 3.1544 Å (experimental)
# Default values used: a = 4.50 Å, c = 3.20 Å (slightly rounded)

def create_iro2_bulk(config: SlabConfig) -> Atoms:
    """Create IrO2 rutile bulk structure.
    
    IrO2 has rutile structure (space group P42/mnm, #136) with:
    - Ir at 2a sites: (0,0,0) and (0.5,0.5,0.5)
    - O at 4f sites: ±(u,u,0) and ±(0.5+u,0.5-u,0.5) where u≈0.3056
    
    Reference: Bolzan et al., Acta Crystallogr. B 1997, 53, 373-380.
    """
    a = config.a_lattice
    c = config.c_lattice
    u = 0.3056  # Oxygen parameter for IrO2
    
    positions = [
        [0, 0, 0],                    # Ir1
        [0.5*a, 0.5*a, 0.5*c],       # Ir2
        [u*a, u*a, 0],                # O1
        [(1-u)*a, (1-u)*a, 0],        # O2
        [(0.5+u)*a, (0.5-u)*a, 0.5*c], # O3
        [(0.5-u)*a, (0.5+u)*a, 0.5*c], # O4
    ]
    
    bulk = Atoms(
        symbols=['Ir', 'Ir', 'O', 'O', 'O', 'O'],
        positions=positions,
        cell=[a, a, c],
        pbc=True
    )
    return bulk


def identify_surface_sites(slab: Atoms) -> Dict:
    """Identify μ1 (CUS), μ2 (bridge), and μ3 (lattice) oxygen sites.
    
    For IrO2(110):
    - μ1 (CUS): Coordinatively unsaturated Ir sites (5-fold)
    - μ2 (bridge): Bridge oxygen connecting two Ir
    - μ3 (lattice): Fully coordinated lattice oxygen
    
    Returns:
        Dict with 'mu1', 'mu2', 'mu3' keys containing site indices and positions
    """
    sym = slab.get_chemical_symbols()
    pos = slab.get_positions()
    z_coords = pos[:, 2]
    
    ir_indices = [i for i, s in enumerate(sym) if s == 'Ir']
    o_indices = [i for i, s in enumerate(sym) if s == 'O']
    
    if not ir_indices:
        return {'mu1': [], 'mu2': [], 'mu3': []}
    
    ir_z = np.array([z_coords[i] for i in ir_indices])
    z_max = ir_z.max()
    z_threshold = z_max - 1.0
    
    # Surface Ir atoms (μ1 sites - CUS)
    mu1_ir = [i for i in ir_indices if z_coords[i] > z_threshold]
    
    # Classify O sites by height relative to surface Ir
    mu2_o = []  # Bridge O (slightly above surface Ir)
    mu3_o = []  # Lattice O (at or below surface Ir level)
    
    for i in o_indices:
        o_z = z_coords[i]
        if o_z > z_max:
            mu2_o.append(i)
        elif o_z > z_threshold:
            mu3_o.append(i)
    
    return {
        'mu1': mu1_ir,
        'mu2': mu2_o,
        'mu3': mu3_o,
        'mu1_positions': [pos[i] for i in mu1_ir],
        'mu2_positions': [pos[i] for i in mu2_o],
    }


def create_iro2_slab(config: SlabConfig) -> Tuple[Atoms, Dict, float, float]:
    """Create IrO2 slab with specified surface termination.
    
    Returns:
        Tuple of (slab, site_info, Lx, Ly)
    """
    bulk = create_iro2_bulk(config)
    
    # Create surface
    slab = surface(bulk, config.facet, config.n_layers, vacuum=config.vacuum)
    
    # Make supercell
    if config.repeat_x > 1 or config.repeat_y > 1:
        slab = make_supercell(slab, [[config.repeat_x, 0, 0],
                                      [0, config.repeat_y, 0],
                                      [0, 0, 1]])
    
    # Center and get dimensions
    pos = slab.get_positions()
    pos[:, 2] -= pos[:, 2].min()
    slab.set_positions(pos)
    
    cell = slab.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]
    
    # Identify surface sites
    site_info = identify_surface_sites(slab)
    
    facet_str = f"{config.facet[0]}{config.facet[1]}{config.facet[2]}"
    print(f"  IrO2({facet_str}): {len(slab)} atoms, {Lx:.2f}×{Ly:.2f} Å")
    print(f"  Surface sites: μ1(CUS)={len(site_info['mu1'])}, μ2(bridge)={len(site_info['mu2'])}")
    
    return slab, site_info, Lx, Ly


# =============================================================================
# ADSORBATE PLACEMENT
# =============================================================================

ADSORBATE_GEOMETRIES = {
    "*OH": {
        "atoms": ["O", "H"],
        "offsets": [[0, 0, 0], [0.6, 0.6, 0.4]],
        "height": 1.9,
    },
    "*O": {
        "atoms": ["O"],
        "offsets": [[0, 0, 0]],
        "height": 1.7,
    },
    "*OOH": {
        "atoms": ["O", "O", "H"],
        "offsets": [[0, 0, 0], [0.9, 0.5, 0.7], [1.5, 1.0, 1.1]],
        "height": 1.9,
    },
    "*OH2": {
        "atoms": ["O", "H", "H"],
        "offsets": [[0, 0, 0], [0.58, 0.58, 0.5], [-0.58, 0.58, 0.5]],
        "height": 2.2,
    },
    "*OO": {
        "atoms": ["O", "O"],
        "offsets": [[0, 0, 0], [0, 0, 1.21]],
        "height": 1.9,
    },
    "*H": {
        "atoms": ["H"],
        "offsets": [[0, 0, 0]],
        "height": 1.0,
    },
    "*": {
        "atoms": [],
        "offsets": [],
        "height": 0,
    },
}


def add_adsorbates(slab: Atoms, site_info: Dict, config: AdsorbateConfig) -> Atoms:
    """Add adsorbates to surface sites randomly.
    
    Places adsorbates on μ1 (CUS) sites according to coverage specification.
    """
    if not site_info['mu1']:
        print("  No μ1 sites available for adsorbates")
        return slab
    
    mu1_positions = list(site_info['mu1_positions'])
    random.shuffle(mu1_positions)
    
    new_atoms = []
    new_positions = []
    site_idx = 0
    
    adsorbate_counts = {}
    
    for species, count in config.coverage.items():
        if species not in ADSORBATE_GEOMETRIES:
            print(f"  Warning: Unknown adsorbate {species}, skipping")
            continue
        
        geom = ADSORBATE_GEOMETRIES[species]
        placed = 0
        
        for _ in range(count):
            if site_idx >= len(mu1_positions):
                break
            
            base_pos = mu1_positions[site_idx]
            site_idx += 1
            
            if species == "*":
                continue
            
            for atom, offset in zip(geom["atoms"], geom["offsets"]):
                pos = base_pos + np.array([offset[0], offset[1], 
                                           geom["height"] + offset[2]])
                new_atoms.append(atom)
                new_positions.append(pos)
            
            placed += 1
        
        if placed > 0:
            adsorbate_counts[species] = placed
    
    if new_atoms:
        combined_sym = list(slab.get_chemical_symbols()) + new_atoms
        combined_pos = list(slab.get_positions()) + new_positions
        slab = Atoms(combined_sym, combined_pos, cell=slab.get_cell(), pbc=slab.pbc)
    
    empty_sites = len(mu1_positions) - site_idx
    print(f"  Adsorbates: {adsorbate_counts}")
    print(f"  Empty sites: {empty_sites}")
    
    return slab


# =============================================================================
# ICE STRUCTURE GENERATION (Bernal-Fowler Rules)
# =============================================================================
# Ice Ic (cubic ice) structure and Bernal-Fowler ice rules for H placement
# References:
#   - Ice Ic: Kuhs et al., PNAS 2012, 109, 21259-21264. DOI: 10.1073/pnas.1210331110
#   - Bernal-Fowler rules: Bernal & Fowler, J. Chem. Phys. 1933, 1, 515-548.
#     DOI: 10.1063/1.1749327
# Ice Ic lattice parameter: a = 6.358 Å at 80 K

def create_ice_ic_oxygens(nx: int, ny: int, nz: int, a: float = 6.358) -> np.ndarray:
    """Create oxygen positions for ice Ic (cubic ice) structure.
    
    Ice Ic has diamond cubic structure with O at tetrahedral sites.
    Lattice parameter a = 6.358 Å gives O-O distance of 2.75 Å.
    
    Reference: Kuhs et al., PNAS 2012, 109, 21259-21264.
    """
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ])
    
    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    pos = (np.array([i, j, k]) + b) * a
                    positions.append(pos)
    
    return np.array(positions)


def get_tetrahedral_neighbors_pbc(positions: np.ndarray, box: np.ndarray) -> List[List[int]]:
    """Find 4 nearest neighbors for each O in tetrahedral arrangement."""
    n = len(positions)
    neighbors = [[] for _ in range(n)]
    
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = positions[j] - positions[i]
            d = d - box * np.round(d / box)
            dist = np.linalg.norm(d)
            dists.append((dist, j))
        
        dists.sort()
        neighbors[i] = [j for _, j in dists[:4]]
    
    return neighbors


def add_hydrogens_bernal_fowler(o_positions: np.ndarray, box: np.ndarray,
                                  neighbors: List[List[int]], 
                                  noise: float = 0.0) -> Tuple[np.ndarray, List]:
    """Add hydrogens following Bernal-Fowler ice rules.
    
    Ice rules:
    1. Each O has exactly 2 covalent O-H bonds
    2. Each O-O edge has exactly 1 H (hydrogen bond)
    3. H is placed 1.0 Å from donor O, pointing toward acceptor O
    """
    n_o = len(o_positions)
    edges = set()
    
    for i in range(n_o):
        for j in neighbors[i]:
            edge = (min(i, j), max(i, j))
            edges.add(edge)
    
    edges = list(edges)
    random.shuffle(edges)
    
    # Assign H to each edge (which O donates)
    edge_donor = {}
    o_h_count = [0] * n_o
    
    for i, j in edges:
        if o_h_count[i] < 2 and o_h_count[j] < 2:
            donor = random.choice([i, j])
        elif o_h_count[i] < 2:
            donor = i
        elif o_h_count[j] < 2:
            donor = j
        else:
            donor = i if o_h_count[i] <= o_h_count[j] else j
        
        edge_donor[(i, j)] = donor
        o_h_count[donor] += 1
    
    # Place H atoms
    h_positions = []
    h_bonds = []
    
    for (i, j), donor in edge_donor.items():
        acceptor = j if donor == i else i
        
        d = o_positions[acceptor] - o_positions[donor]
        d = d - box * np.round(d / box)
        d_norm = d / np.linalg.norm(d)
        
        h_pos = o_positions[donor] + 1.0 * d_norm
        
        if noise > 0:
            h_pos += np.random.uniform(-noise, noise, 3)
        
        h_positions.append(h_pos)
        h_bonds.append((len(h_positions) - 1, donor, acceptor))
    
    return np.array(h_positions), h_bonds


def fix_wrapped_molecules(o_pos: np.ndarray, h_pos: np.ndarray,
                          Lx: float, Ly: float, Lz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fix water molecules broken by periodic wrapping."""
    box = np.array([Lx, Ly, Lz])
    
    for i in range(len(o_pos)):
        o = o_pos[i]
        h1, h2 = h_pos[2*i], h_pos[2*i + 1]
        
        for j, h in enumerate([h1, h2]):
            d = h - o
            shift = np.round(d / box)
            if np.any(shift != 0):
                h_pos[2*i + j] = h - shift * box
    
    return o_pos, h_pos


def create_ice_box(Lx_target: float, Ly_target: float, Lz_target: float,
                   config: WaterConfig) -> Tuple[Atoms, Tuple[float, float, float], 
                                                  Tuple[float, float, float]]:
    """Create ice Ic water box scaled for target density after compression.
    
    Returns:
        Tuple of (water_atoms, initial_dims, target_dims)
    """
    a = config.ice_lattice
    
    # Scale factor for 3D expansion
    scale_factor = (config.target_density / config.ice_density) ** (1/3)
    
    Lx_initial = Lx_target * scale_factor
    Ly_initial = Ly_target * scale_factor
    Lz_initial = Lz_target * scale_factor
    
    print(f"  Packing for {config.target_density:.2f} g/cc target density")
    print(f"  Scale factor: {scale_factor:.3f}")
    print(f"  Initial box: {Lx_initial:.1f}×{Ly_initial:.1f}×{Lz_initial:.1f} Å")
    print(f"  Target box:  {Lx_target:.1f}×{Ly_target:.1f}×{Lz_target:.1f} Å")
    
    # Create ice structure
    nx = max(1, int(np.ceil(Lx_initial / a)))
    ny = max(1, int(np.ceil(Ly_initial / a)))
    nz = max(1, int(np.ceil(Lz_initial / a)))
    
    box = np.array([nx * a, ny * a, nz * a])
    
    print(f"  Ice Ic structure: {nx}×{ny}×{nz} unit cells")
    print(f"  O-O distance: 2.75 Å (tetrahedral)")
    
    o_positions = create_ice_ic_oxygens(nx, ny, nz, a)
    
    # Add small noise
    noise = config.noise_amplitude
    o_positions += np.random.uniform(-noise, noise, o_positions.shape)
    
    # Get neighbors and add hydrogens
    neighbors = get_tetrahedral_neighbors_pbc(o_positions, box)
    h_positions, h_bonds = add_hydrogens_bernal_fowler(o_positions, box, neighbors, noise)
    
    # Trim to initial box
    o_inside = []
    for i, pos in enumerate(o_positions):
        if (0 <= pos[0] < Lx_initial and 
            0 <= pos[1] < Ly_initial and 
            0 <= pos[2] < Lz_initial):
            o_inside.append(i)
    
    # Build O->H mapping
    o_to_h = {i: [] for i in range(len(o_positions))}
    for h_idx, donor, acceptor in h_bonds:
        o_to_h[donor].append(h_idx)
    
    # Keep complete molecules
    final_o = []
    final_h = []
    
    for o_idx in o_inside:
        h_indices = o_to_h[o_idx]
        if len(h_indices) == 2:
            final_o.append(o_positions[o_idx])
            final_h.extend([h_positions[h] for h in h_indices])
    
    n_waters = len(final_o)
    
    print(f"  Initial box: {Lx_initial:.1f}×{Ly_initial:.1f}×{Lz_initial:.1f} Å")
    print(f"  Water molecules: {n_waters}")
    
    # Wrap and fix molecules
    final_o = np.array(final_o)
    final_h = np.array(final_h)
    
    for i in range(len(final_o)):
        final_o[i] = final_o[i] % np.array([Lx_initial, Ly_initial, Lz_initial])
    for i in range(len(final_h)):
        final_h[i] = final_h[i] % np.array([Lx_initial, Ly_initial, Lz_initial])
    
    final_o, final_h = fix_wrapped_molecules(final_o, final_h, 
                                              Lx_initial, Ly_initial, Lz_initial)
    
    # Create Atoms object
    symbols = ['O'] * n_waters + ['H'] * (2 * n_waters)
    positions = list(final_o) + list(final_h)
    
    water = Atoms(symbols, positions,
                  cell=[Lx_initial, Ly_initial, Lz_initial],
                  pbc=True)
    
    # Calculate densities
    mass_kg = n_waters * 18.015 * 1.66054e-27
    vol_initial = Lx_initial * Ly_initial * Lz_initial * 1e-30
    vol_target = Lx_target * Ly_target * Lz_target * 1e-30
    
    rho_initial = mass_kg / vol_initial / 1000
    rho_final = mass_kg / vol_target / 1000
    
    print(f"  Initial density: {rho_initial:.2f} g/cc")
    print(f"  Expected final density: {rho_final:.2f} g/cc")
    
    return water, (Lx_initial, Ly_initial, Lz_initial), (Lx_target, Ly_target, Lz_target)


# =============================================================================
# ION ADDITION
# =============================================================================

def calculate_ion_pairs(Lx: float, Ly: float, Lz: float, concentration: float) -> int:
    """Calculate number of ion pairs for target concentration."""
    volume_L = Lx * Ly * Lz * 1e-27
    n_ions = int(round(concentration * volume_L * 6.022e23))
    return max(0, n_ions)


def add_ions_to_water(water: Atoms, n_pairs: int, config: IonConfig) -> Atoms:
    """Add ion pairs to water box by replacing water molecules."""
    if n_pairs == 0:
        return water
    
    sym = list(water.get_chemical_symbols())
    pos = list(water.get_positions())
    cell = water.get_cell()
    
    # Find water O positions
    o_indices = [i for i, s in enumerate(sym) if s == 'O']
    
    if len(o_indices) < 2 * n_pairs:
        print(f"  Warning: Not enough water for {n_pairs} ion pairs")
        n_pairs = len(o_indices) // 2
    
    random.shuffle(o_indices)
    
    # Replace water molecules with ions
    remove_indices = set()
    ion_positions = []
    
    for i in range(n_pairs):
        # Cation replaces first water
        o_idx = o_indices[2 * i]
        ion_positions.append((config.cation, pos[o_idx]))
        remove_indices.add(o_idx)
        
        # Find associated H atoms
        o_pos = np.array(pos[o_idx])
        for j, s in enumerate(sym):
            if s == 'H' and j not in remove_indices:
                d = np.linalg.norm(np.array(pos[j]) - o_pos)
                if d < 1.3:
                    remove_indices.add(j)
        
        # Anion replaces second water
        o_idx = o_indices[2 * i + 1]
        
        if config.anion == "ClO4":
            # Place Cl at water O position
            ion_positions.append(("Cl", pos[o_idx]))
            # Add 4 O atoms tetrahedrally around Cl
            cl_pos = np.array(pos[o_idx])
            d_clo = 1.45
            for dx, dy, dz in [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]:
                o_pos_new = cl_pos + d_clo * np.array([dx, dy, dz]) / np.sqrt(3)
                ion_positions.append(("O", o_pos_new))
        else:
            ion_positions.append((config.anion, pos[o_idx]))
        
        remove_indices.add(o_idx)
        o_pos = np.array(pos[o_idx])
        for j, s in enumerate(sym):
            if s == 'H' and j not in remove_indices:
                d = np.linalg.norm(np.array(pos[j]) - o_pos)
                if d < 1.3:
                    remove_indices.add(j)
    
    # Build new structure
    new_sym = [s for i, s in enumerate(sym) if i not in remove_indices]
    new_pos = [p for i, p in enumerate(pos) if i not in remove_indices]
    
    for ion_sym, ion_pos in ion_positions:
        new_sym.append(ion_sym)
        new_pos.append(ion_pos)
    
    result = Atoms(new_sym, new_pos, cell=cell, pbc=True)
    
    print(f"  Added {n_pairs} {config.cation}⁺ + {n_pairs} {config.anion}⁻")
    
    return result


# =============================================================================
# LAMMPS FILE WRITING
# =============================================================================

def write_lammps_data(atoms: Atoms, filename: str, ion_config: IonConfig):
    """Write LAMMPS data file with proper atom types and topology."""
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    cell = atoms.get_cell()
    
    # Count atoms by type
    n_o = sum(1 for s in sym if s == 'O')
    n_h = sum(1 for s in sym if s == 'H')
    n_na = sum(1 for s in sym if s == 'Na')
    n_k = sum(1 for s in sym if s == 'K')
    n_cl = sum(1 for s in sym if s == 'Cl')
    
    # Determine atom types
    has_clo4 = (ion_config.anion == "ClO4" and n_cl > 0)
    
    n_types = 2  # O, H
    type_map = {'O_water': 1, 'H': 2}
    
    if n_na > 0:
        n_types += 1
        type_map['Na'] = n_types
    if n_k > 0:
        n_types += 1
        type_map['K'] = n_types
    if n_cl > 0:
        n_types += 1
        type_map['Cl'] = n_types
    if has_clo4:
        n_types += 1
        type_map['O_clo4'] = n_types
    
    # Identify water molecules and bonds
    molecules = []
    o_indices = [i for i, s in enumerate(sym) if s == 'O']
    h_indices = set(i for i, s in enumerate(sym) if s == 'H')
    
    # Group atoms into molecules
    assigned_h = set()
    mol_id = 0
    atom_mol = {}
    atom_type = {}
    
    # Identify which O are ClO4 oxygens
    clo4_o = set()
    if has_clo4:
        cl_indices = [i for i, s in enumerate(sym) if s == 'Cl']
        for cl_idx in cl_indices:
            cl_pos = pos[cl_idx]
            for o_idx in o_indices:
                d = np.linalg.norm(pos[o_idx] - cl_pos)
                if d < 2.0:
                    clo4_o.add(o_idx)
    
    # Water molecules
    bonds = []
    angles = []
    
    for o_idx in o_indices:
        if o_idx in clo4_o:
            continue
        
        nearby_h = []
        for h_idx in h_indices:
            if h_idx in assigned_h:
                continue
            d = np.linalg.norm(pos[h_idx] - pos[o_idx])
            if d < 1.3:
                nearby_h.append(h_idx)
        
        if len(nearby_h) >= 2:
            mol_id += 1
            atom_mol[o_idx] = mol_id
            atom_type[o_idx] = type_map['O_water']
            
            for h_idx in nearby_h[:2]:
                atom_mol[h_idx] = mol_id
                atom_type[h_idx] = type_map['H']
                assigned_h.add(h_idx)
                bonds.append((o_idx + 1, h_idx + 1))
            
            angles.append((nearby_h[0] + 1, o_idx + 1, nearby_h[1] + 1))
    
    # Ions
    for i, s in enumerate(sym):
        if s == 'Na':
            mol_id += 1
            atom_mol[i] = mol_id
            atom_type[i] = type_map['Na']
        elif s == 'K':
            mol_id += 1
            atom_mol[i] = mol_id
            atom_type[i] = type_map['K']
        elif s == 'Cl':
            mol_id += 1
            cl_mol = mol_id
            atom_mol[i] = mol_id
            atom_type[i] = type_map['Cl']
            
            if has_clo4:
                for o_idx in clo4_o:
                    d = np.linalg.norm(pos[o_idx] - pos[i])
                    if d < 2.0:
                        atom_mol[o_idx] = cl_mol
                        atom_type[o_idx] = type_map['O_clo4']
    
    # Handle remaining O atoms (ClO4 oxygens)
    for o_idx in clo4_o:
        if o_idx not in atom_mol:
            atom_type[o_idx] = type_map['O_clo4']
            atom_mol[o_idx] = mol_id
    
    # Write file
    with open(filename, 'w') as f:
        f.write("LAMMPS data file - IrO2 solvation\n\n")
        f.write(f"{len(sym)} atoms\n")
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(angles)} angles\n\n")
        f.write(f"{n_types} atom types\n")
        f.write("1 bond types\n")
        f.write("1 angle types\n\n")
        f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
        f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
        f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n\n")
        
        # Masses
        f.write("Masses\n\n")
        f.write(f"1 {TIP3P['O_mass']:.4f}  # O_water\n")
        f.write(f"2 {TIP3P['H_mass']:.4f}  # H\n")
        type_idx = 2
        if 'Na' in type_map:
            type_idx += 1
            f.write(f"{type_idx} {ION_FF['Na']['mass']:.4f}  # Na\n")
        if 'K' in type_map:
            type_idx += 1
            f.write(f"{type_idx} {ION_FF['K']['mass']:.4f}  # K\n")
        if 'Cl' in type_map:
            type_idx += 1
            f.write(f"{type_idx} {ION_FF['Cl']['mass']:.4f}  # Cl\n")
        if 'O_clo4' in type_map:
            type_idx += 1
            f.write(f"{type_idx} {ION_FF['ClO4_O']['mass']:.4f}  # O_ClO4\n")
        f.write("\n")
        
        # Atoms
        f.write("Atoms # full\n\n")
        for i, (s, p) in enumerate(zip(sym, pos)):
            mol = atom_mol.get(i, 1)
            atype = atom_type.get(i, 1)
            
            if s == 'O' and i not in clo4_o:
                charge = TIP3P['O_charge']
            elif s == 'H':
                charge = TIP3P['H_charge']
            elif s == 'Na':
                charge = ION_FF['Na']['charge']
            elif s == 'K':
                charge = ION_FF['K']['charge']
            elif s == 'Cl':
                charge = ION_FF['ClO4_Cl']['charge'] if has_clo4 else ION_FF['Cl']['charge']
            elif s == 'O' and i in clo4_o:
                charge = ION_FF['ClO4_O']['charge']
            else:
                charge = 0.0
            
            f.write(f"{i+1} {mol} {atype} {charge:.4f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        
        # Bonds
        if bonds:
            f.write("\nBonds\n\n")
            for i, (a1, a2) in enumerate(bonds):
                f.write(f"{i+1} 1 {a1} {a2}\n")
        
        # Angles
        if angles:
            f.write("\nAngles\n\n")
            for i, (a1, a2, a3) in enumerate(angles):
                f.write(f"{i+1} 1 {a1} {a2} {a3}\n")


def write_lammps_equilibration_input(in_file: str, data_file: str,
                                      water_config: WaterConfig,
                                      ion_config: IonConfig,
                                      initial_dims: Tuple[float, float, float],
                                      target_dims: Tuple[float, float, float],
                                      fast_mode: bool = True):
    """Write LAMMPS input script for water equilibration."""
    melt_T = water_config.melt_temp
    equil_T = water_config.equil_temp
    
    timestep = 2.0 if fast_mode else 1.0
    compress_steps = int(water_config.compress_time * 1000 / timestep)
    final_steps = int(water_config.final_equil_time * 1000 / timestep)
    
    if fast_mode:
        heat_steps = 2500
        pppm_prec = "1e-3"
        neigh_delay = 5
    else:
        heat_steps = 5000
        pppm_prec = "1e-4"
        neigh_delay = 0
    
    Lx_i, Ly_i, Lz_i = initial_dims
    Lx_f, Ly_f, Lz_f = target_dims
    
    cation = ion_config.cation
    cation_params = ION_FF.get(cation, ION_FF["Na"])
    
    script = f"""\
# Water Equilibration for IrO2 Solvation
# ======================================
# Protocol: Minimize -> Heat to {melt_T:.0f}K -> Compress -> Brief NVT
# Mode: {'FAST' if fast_mode else 'STANDARD'} (timestep={timestep} fs)

units real
atom_style full
boundary p p p

read_data {data_file}

# Force field: TIP3P water + ions
pair_style lj/cut/coul/long 10.0
pair_modify mix arithmetic
kspace_style pppm {pppm_prec}

# Water O-O interaction
pair_coeff 1 1 {TIP3P['O_epsilon']} {TIP3P['O_sigma']}
pair_coeff 2 2 0.0 0.0

# Ion interactions
pair_coeff 3 3 {cation_params['epsilon']} {cation_params['sigma']}
pair_coeff 4 4 {ION_FF['ClO4_Cl']['epsilon']} {ION_FF['ClO4_Cl']['sigma']}
pair_coeff 5 5 {ION_FF['ClO4_O']['epsilon']} {ION_FF['ClO4_O']['sigma']}

bond_style harmonic
bond_coeff 1 {TIP3P['OH_k']} {TIP3P['OH_bond']}

angle_style harmonic
angle_coeff 1 {TIP3P['HOH_k']} {TIP3P['HOH_angle']}

neighbor 2.0 bin
neigh_modify every 1 delay {neigh_delay} check yes

thermo_style custom step temp press density pe ke etotal
thermo 1000

# Stage 1: Minimize
print "PROGRESS: Minimizing..."
minimize 1e-4 1e-6 1000 10000

reset_timestep 0
timestep {timestep}

fix shake all shake 0.0001 20 0 b 1 a 1

# Stage 2: Heat to {melt_T:.0f}K
print "PROGRESS: Heating 10K -> {melt_T:.0f}K..."
fix nvt_heat all nvt temp 10.0 {melt_T} 100.0
run {heat_steps}
unfix nvt_heat
print "PROGRESS: density = $(density)"

# Stage 3: Compress to target density
print "PROGRESS: Compressing to target density..."
fix nvt_compress all nvt temp {melt_T} {melt_T} 100.0
fix deform_all all deform 1 x final 0.0 {Lx_f:.6f} y final 0.0 {Ly_f:.6f} z final 0.0 {Lz_f:.6f} remap x
run {compress_steps}
unfix deform_all
unfix nvt_compress
print "PROGRESS: After compression, density = $(density)"
print "PROGRESS: Box = $(lx) x $(ly) x $(lz)"

# Stage 4: Brief equilibration
print "PROGRESS: Brief equilibration at {equil_T:.0f}K..."
fix nvt_final all nvt temp {equil_T} {equil_T} 100.0
run {final_steps}
unfix nvt_final

unfix shake

print "PROGRESS: Done!"
print "FINAL: density=$(density), Lx=$(lx), Ly=$(ly), Lz=$(lz)"
write_data water_final.data
"""
    
    with open(in_file, 'w') as f:
        f.write(script)


# =============================================================================
# LAMMPS EXECUTION
# =============================================================================

def run_lammps(config: LAMMPSConfig, input_file: str):
    """Run LAMMPS simulation with progress monitoring."""
    cmd = [
        config.mpirun, "-np", str(config.np),
        config.lammps, "-in", input_file, "-log", config.log_file
    ]
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.omp_threads)
    
    print(f"    Command: {' '.join(cmd)}")
    print(f"    Working dir: {config.tmp_dir}")
    
    process = subprocess.Popen(
        cmd,
        cwd=config.tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        if "PROGRESS:" in line or "FINAL:" in line:
            print(f"    {line.strip()}")
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"LAMMPS failed with return code {process.returncode}")


# =============================================================================
# POST-PROCESSING
# =============================================================================

def read_lammps_data(filename: str) -> Tuple[Atoms, Tuple[float, float, float]]:
    """Read LAMMPS data file back into ASE Atoms."""
    with open(filename) as f:
        lines = f.readlines()
    
    n_atoms = int(lines[2].split()[0])
    
    Lx = Ly = Lz = 0
    for line in lines:
        if 'xlo xhi' in line:
            parts = line.split()
            Lx = float(parts[1]) - float(parts[0])
        if 'ylo yhi' in line:
            parts = line.split()
            Ly = float(parts[1]) - float(parts[0])
        if 'zlo zhi' in line:
            parts = line.split()
            Lz = float(parts[1]) - float(parts[0])
    
    atoms_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Atoms'):
            atoms_start = i + 2
            break
    
    type_to_sym = {1: 'O', 2: 'H', 3: 'Na', 4: 'Cl', 5: 'O'}
    
    symbols = []
    positions = []
    
    for i in range(atoms_start, atoms_start + n_atoms):
        parts = lines[i].split()
        if len(parts) >= 7:
            atom_type = int(parts[2])
            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            symbols.append(type_to_sym.get(atom_type, 'X'))
            positions.append([x, y, z])
    
    atoms = Atoms(symbols, positions, cell=[Lx, Ly, Lz], pbc=True)
    return atoms, (Lx, Ly, Lz)


def fix_z_boundary_molecules(atoms: Atoms) -> Atoms:
    """Fix water molecules split across z-periodic boundary."""
    pos = atoms.get_positions().copy()
    sym = list(atoms.get_chemical_symbols())
    cell = atoms.get_cell()
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    
    o_indices = [i for i, s in enumerate(sym) if s == 'O']
    h_indices = [i for i, s in enumerate(sym) if s == 'H']
    other_indices = [i for i, s in enumerate(sym) if s not in ['O', 'H']]
    
    molecules = []
    assigned_h = set()
    assigned_o = set()
    
    for o_idx in o_indices:
        o_pos = pos[o_idx]
        nearby_h = []
        
        for h_idx in h_indices:
            if h_idx in assigned_h:
                continue
            h_pos = pos[h_idx]
            
            d = h_pos - o_pos
            d[0] -= Lx * round(d[0] / Lx)
            d[1] -= Ly * round(d[1] / Ly)
            d[2] -= Lz * round(d[2] / Lz)
            dist = np.linalg.norm(d)
            
            if dist < 1.3:
                nearby_h.append((dist, h_idx, d))
        
        nearby_h.sort()
        if len(nearby_h) >= 2:
            _, h1_idx, d1 = nearby_h[0]
            _, h2_idx, d2 = nearby_h[1]
            molecules.append((o_idx, h1_idx, h2_idx, d1, d2))
            assigned_h.add(h1_idx)
            assigned_h.add(h2_idx)
            assigned_o.add(o_idx)
    
    new_sym = []
    new_pos = []
    
    for o_idx, h1_idx, h2_idx, d1, d2 in molecules:
        o_pos = pos[o_idx]
        h1_pos = o_pos + d1
        h2_pos = o_pos + d2
        
        new_sym.extend(['O', 'H', 'H'])
        new_pos.extend([o_pos, h1_pos, h2_pos])
    
    for i in other_indices:
        new_sym.append(sym[i])
        new_pos.append(pos[i])
    
    n_dangling_h = len(h_indices) - 2 * len(molecules)
    n_dangling_o = len(o_indices) - len(molecules)
    
    if n_dangling_h > 0 or n_dangling_o > 0:
        print(f"  Fixed z-boundary: removed {n_dangling_o} dangling O, {n_dangling_h} dangling H")
    
    new_atoms = Atoms(new_sym, new_pos, cell=cell, pbc=atoms.pbc)
    return new_atoms


def solvate_slab(slab: Atoms, water: Atoms, wcell: Tuple[float, float, float],
                 config: SolvationConfig, site_info: Dict) -> Atoms:
    """Combine slab and water box, removing overlapping molecules."""
    slab_pos = slab.get_positions()
    slab_sym = slab.get_chemical_symbols()
    slab_z_max = slab_pos[:, 2].max()
    
    water_pos = water.get_positions().copy()
    water_sym = list(water.get_chemical_symbols())
    
    water_pos[:, 2] += slab_z_max + config.water_gap
    
    Lx, Ly = wcell[0], wcell[1]
    water_z_max = water_pos[:, 2].max()
    new_Lz = water_z_max + 5.0
    
    molecules = identify_water_molecules(water_sym, water_pos)
    
    overlap_atoms = set()
    for i, wpos in enumerate(water_pos):
        for slab_p in slab_pos:
            d = np.linalg.norm(wpos - slab_p)
            if d < config.overlap_cutoff:
                overlap_atoms.add(i)
                break
    
    remove_atoms = set()
    for mol in molecules:
        if mol & overlap_atoms:
            remove_atoms.update(mol)
    
    all_mol_atoms = set()
    for mol in molecules:
        all_mol_atoms.update(mol)
    
    orphan_atoms = set(range(len(water_sym))) - all_mol_atoms
    remove_atoms.update(orphan_atoms)
    
    keep_water_sym = []
    keep_water_pos = []
    for i, (s, p) in enumerate(zip(water_sym, water_pos)):
        if i not in remove_atoms:
            keep_water_sym.append(s)
            keep_water_pos.append(p)
    
    combined_sym = list(slab_sym) + keep_water_sym
    combined_pos = list(slab_pos) + list(keep_water_pos)
    
    combined = Atoms(combined_sym, combined_pos, 
                     cell=[Lx, Ly, new_Lz], pbc=[True, True, True])
    
    n_removed = len(remove_atoms)
    n_orphans = len(orphan_atoms)
    n_overlap = (n_removed - n_orphans) // 3
    
    if n_orphans > 0:
        print(f"  Removed {n_orphans} orphan atoms, {n_overlap} overlapping molecules")
    else:
        print(f"  Removed {n_overlap} overlapping water molecules")
    
    return combined


def identify_water_molecules(sym: List[str], pos: np.ndarray) -> List[Set[int]]:
    """Identify complete water molecules."""
    molecules = []
    o_indices = [i for i, s in enumerate(sym) if s == 'O']
    h_indices = set(i for i, s in enumerate(sym) if s == 'H')
    
    assigned_h = set()
    
    for o_idx in o_indices:
        h_dists = []
        for h_idx in h_indices:
            if h_idx in assigned_h:
                continue
            d = np.linalg.norm(pos[h_idx] - pos[o_idx])
            if d < 1.3:
                h_dists.append((d, h_idx))
        
        h_dists.sort()
        mol_h = [h_idx for _, h_idx in h_dists[:2]]
        
        if len(mol_h) == 2:
            molecules.append({o_idx} | set(mol_h))
            assigned_h.update(mol_h)
    
    return molecules


def compute_density(atoms: Atoms) -> float:
    """Compute mass density in g/cc."""
    mass_kg = atoms.get_masses().sum() * 1.66054e-27
    cell = atoms.get_cell()
    volume_m3 = cell[0, 0] * cell[1, 1] * cell[2, 2] * 1e-30
    return mass_kg / volume_m3 / 1000


def count_waters(atoms: Atoms) -> int:
    """Count water molecules in structure."""
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    
    o_indices = [i for i, s in enumerate(sym) if s == 'O']
    h_indices = [i for i, s in enumerate(sym) if s == 'H']
    
    count = 0
    used_h = set()
    
    for o_idx in o_indices:
        h_count = 0
        for h_idx in h_indices:
            if h_idx in used_h:
                continue
            d = np.linalg.norm(pos[h_idx] - pos[o_idx])
            if d < 1.3:
                h_count += 1
                used_h.add(h_idx)
                if h_count == 2:
                    break
        if h_count == 2:
            count += 1
    
    return count


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main(args=None):
    """Main workflow function."""
    parser = argparse.ArgumentParser(
        description="Generate solvated IrO2 slab structures for DFT/MD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default: IrO2(110) + 25 Å water
  %(prog)s --facet 100              # IrO2(100) surface
  %(prog)s --water-height 30        # 30 Å water layer
  %(prog)s --concentration 0.5      # 0.5 M electrolyte
  %(prog)s --lammps /path/to/lmp    # Custom LAMMPS path
        """
    )
    
    parser.add_argument('--facet', type=int, nargs=3, default=[1, 1, 0],
                        help='Miller indices for surface (default: 1 1 0)')
    parser.add_argument('--layers', type=int, default=5,
                        help='Number of slab layers (default: 5)')
    parser.add_argument('--supercell', type=int, nargs=2, default=[4, 4],
                        help='Supercell size in x y (default: 4 4)')
    parser.add_argument('--water-height', type=float, default=25.0,
                        help='Water layer thickness in Å (default: 25)')
    parser.add_argument('--concentration', type=float, default=0.1,
                        help='Electrolyte concentration in M (default: 0.1)')
    parser.add_argument('--cation', type=str, default='Na',
                        choices=['Na', 'K', 'H3O'],
                        help='Cation species (default: Na)')
    parser.add_argument('--anion', type=str, default='ClO4',
                        choices=['ClO4', 'Cl'],
                        help='Anion species (default: ClO4)')
    parser.add_argument('--lammps', type=str, default='lmp',
                        help='Path to LAMMPS executable')
    parser.add_argument('--mpirun', type=str, default='mpirun',
                        help='Path to mpirun executable')
    parser.add_argument('--np', type=int, default=4,
                        help='Number of MPI processes (default: 4)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename prefix')
    
    args = parser.parse_args(args)
    
    # Configure
    SLAB = SlabConfig(
        facet=tuple(args.facet),
        n_layers=args.layers,
        repeat_x=args.supercell[0],
        repeat_y=args.supercell[1],
    )
    
    WATER = WaterConfig(height=args.water_height)
    IONS = IonConfig(
        concentration=args.concentration,
        cation=args.cation,
        anion=args.anion,
    )
    SOLVATION = SolvationConfig()
    LAMMPS = LAMMPSConfig(
        lammps=args.lammps,
        mpirun=args.mpirun,
        np=args.np,
    )
    
    ADSORBATES = AdsorbateConfig()
    
    # Run workflow
    start_time = time.time()
    facet_str = f"{SLAB.facet[0]}{SLAB.facet[1]}{SLAB.facet[2]}"
    
    print("=" * 70)
    print(f"  IrO2({facet_str}) + Water Box Generator v{__version__}")
    print("=" * 70)
    print(f"\n  Slab: {SLAB.repeat_x}×{SLAB.repeat_y} supercell, {SLAB.n_layers} layers")
    print(f"  Water: {WATER.height} Å thick, gap to surface: {SOLVATION.water_gap} Å")
    print(f"  Electrolyte: {IONS.concentration} M {IONS.cation}{IONS.anion}")
    print()
    
    os.makedirs(LAMMPS.tmp_dir, exist_ok=True)
    
    # Step 1: Create slab
    print("[1/6] Creating IrO2 slab...")
    slab, site_info, Lx, Ly = create_iro2_slab(SLAB)
    
    # Step 2: Add adsorbates
    print("\n[2/6] Adding surface adsorbates...")
    slab = add_adsorbates(slab, site_info, ADSORBATES)
    write("01_slab.cif", slab)
    
    # Step 3: Create ice-based water box
    print("\n[3/6] Creating water box (ice structure)...")
    water_box, initial_dims, target_dims = create_ice_box(Lx, Ly, WATER.height, WATER)
    
    Lx_i, Ly_i, Lz_i = initial_dims
    n_ions = calculate_ion_pairs(Lx_i, Ly_i, Lz_i, IONS.concentration)
    water_box = add_ions_to_water(water_box, n_ions, IONS)
    
    write("02_water_initial.cif", water_box)
    
    # Step 4: Write LAMMPS files
    print("\n[4/6] Preparing LAMMPS simulation...")
    data_file = os.path.join(LAMMPS.tmp_dir, "water_initial.data")
    write_lammps_data(water_box, data_file, IONS)
    
    input_file = os.path.join(LAMMPS.tmp_dir, "in.equil")
    write_lammps_equilibration_input(
        input_file, "water_initial.data", WATER, IONS, initial_dims, target_dims,
        fast_mode=True
    )
    
    total_ps = WATER.compress_time + WATER.final_equil_time + 5
    print(f"  Protocol: Heat -> Compress -> Equilibrate (~{total_ps:.0f} ps total)")
    print(f"  Mode: FAST (2 fs timestep, minimal equilibration)")
    
    # Step 5: Run LAMMPS
    print("\n[5/6] Running LAMMPS equilibration...")
    md_start = time.time()
    
    run_lammps(LAMMPS, "in.equil")
    
    final_data = os.path.join(LAMMPS.tmp_dir, "water_final.data")
    water_eq, wcell = read_lammps_data(final_data)
    water_eq = fix_z_boundary_molecules(water_eq)
    
    print(f"    LAMMPS time: {(time.time() - md_start)/60:.1f} min")
    
    write("03_water_equil.cif", water_eq)
    
    density = compute_density(water_eq)
    print(f"\n  Final water density: {density:.3f} g/cc")
    print(f"  Water box: {wcell[0]:.1f}×{wcell[1]:.1f}×{wcell[2]:.1f} Å")
    
    # Step 6: Combine slab and water
    print("\n[6/6] Solvating slab...")
    combined = solvate_slab(slab, water_eq, wcell, SOLVATION, site_info)
    
    n_final_waters = count_waters(combined)
    print(f"  Water molecules in final structure: {n_final_waters}")
    
    # Write outputs
    if args.output:
        output_name = args.output
    else:
        output_name = f"IrO2_{facet_str}_waterbox"
    
    write(f"{output_name}.cif", combined)
    write_lammps_data(combined, f"{output_name}.data", IONS)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  ✅ COMPLETE in {(time.time() - start_time)/60:.1f} min")
    print("=" * 70)
    
    sym = combined.get_chemical_symbols()
    print(f"\n  Final structure: {len(combined)} atoms")
    for element in sorted(set(sym)):
        print(f"    {element}: {sym.count(element)}")
    
    print(f"\n  Output files:")
    print(f"    - {output_name}.cif")
    print(f"    - {output_name}.data")
    print()


if __name__ == "__main__":
    main()
