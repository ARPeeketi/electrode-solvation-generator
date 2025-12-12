# Electrode-Electrolyte Interface Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generate solvated electrode slab structures for DFT and MD simulations of electrochemical interfaces. Originally designed for IrO2 oxygen evolution reaction (OER) studies, but easily extensible to other metal oxides and metals.

**Author:** Akhil Reddy Peeketi, with the help of Claude (Anthropic)

## Features

- 🔬 **Multiple electrode materials**: IrO2, RuO2, TiO2, PtO2, and pure metals (Pt, Au, Pd)
- 🌊 **Realistic water structure**: Ice Ic with Bernal-Fowler H-bonding rules
- ⚡ **Fast equilibration**: ~30 seconds for production-ready initial structures
- 🧪 **Electrolyte support**: NaClO4, KClO4, HClO4, NaCl, KCl
- 🎯 **Surface site identification**: CUS, bridge, and lattice oxygen sites
- 📦 **Multiple output formats**: CIF, LAMMPS data files

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Customization](#customization)
  - [Different Electrode Materials](#different-electrode-materials)
  - [Different Ions](#different-ions)
  - [Surface Adsorbates](#surface-adsorbates)
- [Algorithm Details](#algorithm-details)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

1. **Python 3.8+** with pip
2. **LAMMPS** compiled with molecule and kspace packages
3. **MPI** (optional, for parallel execution)

### Step 1: Clone the repository

```bash
git clone https://github.com/ARPeeketi/electrode-solvation-generator.git
cd electrode-solvation-generator
```

### Step 2: Install Python dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ase numpy
```

### Step 3: Install LAMMPS

**Ubuntu/Debian:**
```bash
sudo apt-get install lammps
```

**macOS (Homebrew):**
```bash
brew install lammps
```

**From source (recommended for HPC):**
```bash
git clone https://github.com/lammps/lammps.git
cd lammps/src
make yes-molecule yes-kspace
make mpi -j4
```

### Step 4: Verify installation

```bash
python electrode_solvation.py --help
```

## Quick Start

Generate a default IrO2(110) + water interface:

```bash
python electrode_solvation.py
```

This creates:
- `IrO2_110_waterbox.cif` - Final structure
- `IrO2_110_waterbox.data` - LAMMPS data file

## Usage

### Basic Examples

```bash
# IrO2(110) with 25 Å water (default)
python electrode_solvation.py

# IrO2(100) surface
python electrode_solvation.py --facet 1 0 0

# Larger supercell (6×6)
python electrode_solvation.py --supercell 6 6

# Thicker water layer (35 Å)
python electrode_solvation.py --water-height 35

# Higher electrolyte concentration (0.5 M)
python electrode_solvation.py --concentration 0.5

# Different electrolyte (KCl)
python electrode_solvation.py --cation K --anion Cl
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--facet X Y Z` | `1 1 0` | Miller indices for surface |
| `--layers N` | `5` | Number of electrode layers |
| `--supercell X Y` | `4 4` | Supercell size |
| `--water-height H` | `25.0` | Water thickness (Å) |
| `--concentration C` | `0.1` | Electrolyte concentration (M) |
| `--cation` | `Na` | Cation: Na, K, H3O |
| `--anion` | `ClO4` | Anion: ClO4, Cl |
| `--lammps PATH` | `lmp` | LAMMPS executable |
| `--mpirun PATH` | `mpirun` | MPI launcher |
| `--np N` | `4` | MPI processes |
| `--output NAME` | auto | Output filename prefix |

### Output Files

| File | Description |
|------|-------------|
| `*_waterbox.cif` | Final solvated structure (CIF format) |
| `*_waterbox.data` | LAMMPS data file for MD |
| `01_slab.cif` | Bare electrode with adsorbates |
| `02_water_initial.cif` | Initial ice structure |
| `03_water_equil.cif` | Equilibrated water box |

## Customization

### Different Electrode Materials

The code is designed for rutile-structure oxides but can be adapted for other materials.

#### Rutile Oxides (IrO2, RuO2, TiO2, SnO2)

Edit the lattice parameters in `SlabConfig`:

```python
# IrO2 (default)
SLAB = SlabConfig(
    a_lattice=4.50,
    c_lattice=3.20,
)

# RuO2
SLAB = SlabConfig(
    a_lattice=4.49,
    c_lattice=3.11,
)

# TiO2 (rutile)
SLAB = SlabConfig(
    a_lattice=4.59,
    c_lattice=2.96,
)

# SnO2
SLAB = SlabConfig(
    a_lattice=4.74,
    c_lattice=3.19,
)
```

#### Pure Metals (Pt, Au, Pd, Ag)

For FCC metals, you need to modify `create_electrode_bulk()`:

```python
def create_fcc_bulk(a: float, symbol: str) -> Atoms:
    """Create FCC metal bulk structure."""
    from ase.build import bulk
    return bulk(symbol, 'fcc', a=a, cubic=True)

# Example usage for Pt(111)
# a_Pt = 3.92 Å
slab = surface(create_fcc_bulk(3.92, 'Pt'), (1,1,1), layers=5)
```

See `examples/pt_solvation.py` for a complete Pt example.

#### Perovskites (SrTiO3, BaTiO3)

```python
def create_perovskite_bulk(a: float, A: str, B: str) -> Atoms:
    """Create ABO3 perovskite structure."""
    positions = [
        [0, 0, 0],           # A site
        [0.5*a, 0.5*a, 0.5*a],  # B site
        [0.5*a, 0.5*a, 0],   # O
        [0.5*a, 0, 0.5*a],   # O
        [0, 0.5*a, 0.5*a],   # O
    ]
    return Atoms([A, B, 'O', 'O', 'O'], positions, cell=[a,a,a], pbc=True)
```

### Different Ions

#### Adding New Cations

Add parameters to `ION_FF` dictionary:

```python
ION_FF = {
    # Existing
    "Na": {"mass": 22.990, "charge": 1.0, "sigma": 2.350, "epsilon": 0.130},
    "K":  {"mass": 39.098, "charge": 1.0, "sigma": 3.154, "epsilon": 0.100},
    
    # Add new cations
    "Li": {"mass": 6.941,  "charge": 1.0, "sigma": 1.582, "epsilon": 0.118},
    "Cs": {"mass": 132.91, "charge": 1.0, "sigma": 3.883, "epsilon": 0.047},
    "Mg": {"mass": 24.305, "charge": 2.0, "sigma": 1.644, "epsilon": 0.875},
    "Ca": {"mass": 40.078, "charge": 2.0, "sigma": 2.412, "epsilon": 0.449},
    "Zn": {"mass": 65.38,  "charge": 2.0, "sigma": 1.948, "epsilon": 0.250},
}
```

#### Adding New Anions

For simple anions (single atom):
```python
ION_FF["Br"] = {"mass": 79.904, "charge": -1.0, "sigma": 4.624, "epsilon": 0.080}
ION_FF["I"]  = {"mass": 126.90, "charge": -1.0, "sigma": 5.167, "epsilon": 0.070}
ION_FF["F"]  = {"mass": 18.998, "charge": -1.0, "sigma": 3.118, "epsilon": 0.710}
```

For polyatomic anions (like SO4²⁻, NO3⁻), see `examples/polyatomic_ions.py`.

#### Common Electrolyte Recipes

```python
# Seawater (~0.5 M NaCl)
IONS = IonConfig(concentration=0.5, cation="Na", anion="Cl")

# Battery electrolyte (1 M LiPF6) - simplified
IONS = IonConfig(concentration=1.0, cation="Li", anion="Cl")  # PF6 approximated

# Acidic OER (0.1 M HClO4)
IONS = IonConfig(concentration=0.1, cation="H3O", anion="ClO4")

# Alkaline OER (0.1 M KOH) - add OH⁻ to ION_FF first
IONS = IonConfig(concentration=0.1, cation="K", anion="OH")
```

### Surface Adsorbates

Configure OER intermediates or other adsorbates:

```python
ADSORBATES = AdsorbateConfig(
    coverage={
        "*OH": 4,    # 4 hydroxyl groups
        "*O": 2,     # 2 oxo groups
        "*OOH": 1,   # 1 hydroperoxo
        "*OH2": 2,   # 2 adsorbed water
    }
)
```

#### Adding Custom Adsorbates

```python
ADSORBATE_GEOMETRIES["*COOH"] = {
    "atoms": ["C", "O", "O", "H"],
    "offsets": [[0, 0, 0], [1.2, 0, 0.3], [-0.4, 0, 1.2], [-0.4, 0, 2.2]],
    "height": 1.9,
}

ADSORBATE_GEOMETRIES["*CO"] = {
    "atoms": ["C", "O"],
    "offsets": [[0, 0, 0], [0, 0, 1.13]],
    "height": 1.85,
}
```

## Algorithm Details

### 1. Electrode Slab Generation
- Creates bulk unit cell (rutile, FCC, etc.)
- Cuts surface with ASE's `surface()` function
- Identifies reactive sites (CUS, bridge O)

### 2. Water Box Generation
- **Ice Ic structure**: Diamond cubic oxygen arrangement
- **Bernal-Fowler rules**: Each O has exactly 2 covalent O-H bonds
- **H-bond network**: Each O-O edge has exactly 1 hydrogen bond
- **Scaling**: Box sized for target density after compression

### 3. LAMMPS Equilibration
```
Protocol (~20 ps total):
├── Minimize (energy minimization)
├── Heat: 10K → 300K (5 ps)
├── Compress: to target density (5 ps)  
└── Equilibrate: NVT at 300K (10 ps)
```

**Force Fields:**
- Water: TIP3P (rigid bonds with SHAKE)
- Ions: Joung-Cheatham parameters

### 4. Solvation
- Positions water 0.5 Å above electrode surface
- Removes overlapping molecules (< 1.0 Å)
- Fixes molecules split across periodic boundaries

## Examples

See the `examples/` directory:

| File | Description |
|------|-------------|
| `basic_usage.py` | Simple IrO2(110) + water |
| `pt_solvation.py` | Pt(111) metal electrode |
| `ruo2_solvation.py` | RuO2(110) for OER |
| `different_ions.py` | Various electrolyte examples |
| `custom_adsorbates.py` | Adding new adsorbate species |
| `large_system.py` | Building larger supercells |

## Troubleshooting

### LAMMPS not found
```
Error: LAMMPS executable not found
```
**Solution:** Specify path with `--lammps /full/path/to/lmp`

### MPI errors
```
Error: mpirun not found
```
**Solution:** 
- Install MPI: `sudo apt install openmpi-bin`
- Or use serial: `--np 1 --mpirun ""`

### Memory issues with large systems
```
Error: Out of memory
```
**Solution:** Reduce supercell size or use fewer MPI processes

### Water density too low/high
Check the compression worked:
```bash
grep "density" lammps_equil/water_equil.log
```
Adjust `target_density` in `WaterConfig` if needed.

### Dangling atoms in output
The code automatically fixes molecules split across periodic boundaries. If you still see issues, check the `fix_z_boundary_molecules()` function.

## Performance

| System Size | Atoms | Runtime |
|-------------|-------|---------|
| 4×4 supercell, 25 Å water | ~1500 | ~30 sec |
| 6×6 supercell, 25 Å water | ~3000 | ~60 sec |
| 4×4 supercell, 40 Å water | ~2500 | ~45 sec |
| 8×8 supercell, 30 Å water | ~6000 | ~120 sec |

Tested on M1 MacBook Pro (4 MPI processes).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{electrode_solvation,
  title = {Electrode-Electrolyte Interface Generator},
  author = {Peeketi, Akhil Reddy},
  year = {2024},
  url = {https://github.com/ARPeeketi/electrode-solvation-generator}
}
```

## References

This code uses parameters and methods from the following sources:

### Water Model
- **TIP3P Water Model:**
  Jorgensen, W. L.; Chandrasekhar, J.; Madura, J. D.; Impey, R. W.; Klein, M. L. 
  *Comparison of Simple Potential Functions for Simulating Liquid Water.* 
  J. Chem. Phys. **1983**, 79, 926–935. [DOI: 10.1063/1.445869](https://doi.org/10.1063/1.445869)

### Ion Parameters
- **Alkali Halide Ion Parameters (Na⁺, K⁺, Cl⁻):**
  Joung, I. S.; Cheatham, T. E. 
  *Determination of Alkali and Halide Monovalent Ion Parameters for Use in Explicitly Solvated Biomolecular Simulations.* 
  J. Phys. Chem. B **2008**, 112, 9020–9041. [DOI: 10.1021/jp8001614](https://doi.org/10.1021/jp8001614)

### Ice Structure
- **Ice Ic (Cubic Ice) Structure:**
  Kuhs, W. F.; Sippel, C.; Falenty, A.; Hansen, T. C. 
  *Extent and Relevance of Stacking Disorder in "Ice Ic".* 
  Proc. Natl. Acad. Sci. **2012**, 109, 21259–21264. [DOI: 10.1073/pnas.1210331110](https://doi.org/10.1073/pnas.1210331110)

- **Bernal-Fowler Ice Rules:**
  Bernal, J. D.; Fowler, R. H. 
  *A Theory of Water and Ionic Solution, with Particular Reference to Hydrogen and Hydroxyl Ions.* 
  J. Chem. Phys. **1933**, 1, 515–548. [DOI: 10.1063/1.1749327](https://doi.org/10.1063/1.1749327)

### IrO2 Structure
- **IrO2 Rutile Structure:**
  Bolzan, A. A.; Fong, C.; Kennedy, B. J.; Howard, C. J. 
  *Structural Studies of Rutile-Type Metal Dioxides.* 
  Acta Crystallogr. B **1997**, 53, 373–380. [DOI: 10.1107/S0108768197001468](https://doi.org/10.1107/S0108768197001468)

### Software
- **ASE (Atomic Simulation Environment):**
  Larsen, A. H.; et al. 
  *The Atomic Simulation Environment—A Python Library for Working with Atoms.* 
  J. Phys.: Condens. Matter **2017**, 29, 273002. [DOI: 10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

- **LAMMPS:**
  Thompson, A. P.; et al. 
  *LAMMPS - A Flexible Simulation Tool for Particle-Based Materials Modeling at the Atomic, Meso, and Continuum Scales.* 
  Comput. Phys. Commun. **2022**, 271, 108171. [DOI: 10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ASE](https://wiki.fysik.dtu.dk/ase/) - Atomic Simulation Environment
- [LAMMPS](https://www.lammps.org/) - Molecular Dynamics Simulator
- Claude (Anthropic) for code development assistance
