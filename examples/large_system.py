#!/usr/bin/env python3
"""
Large System Example
=====================

This example shows how to build larger systems for:
- Better statistics
- Realistic simulations
- DFT-MD with periodic boundary effects minimized

System size recommendations:
- Minimum DFT: 3×3 supercell, ~15 Å water
- Standard DFT: 4×4 supercell, 20-25 Å water
- Large DFT: 6×6 supercell, 30+ Å water
- Classical MD: 8×8 supercell, 40+ Å water

Memory requirements (approximate):
- 4×4, 25 Å water: ~1500 atoms, 4 GB RAM
- 6×6, 30 Å water: ~4000 atoms, 8 GB RAM
- 8×8, 40 Å water: ~10000 atoms, 16 GB RAM
"""

import subprocess
import sys

print("=" * 60)
print("  Large System Examples")
print("=" * 60)

# =============================================================================
# SIZE RECOMMENDATIONS
# =============================================================================

print("\n1. SIZE RECOMMENDATIONS:\n")

sizes = """
| Use Case | Supercell | Water (Å) | ~Atoms | ~RAM | ~Time |
|----------|-----------|-----------|--------|------|-------|
| Quick test | 2×2 | 15 | 400 | 2 GB | 15 s |
| DFT single point | 3×3 | 20 | 800 | 4 GB | 20 s |
| DFT optimization | 4×4 | 25 | 1500 | 4 GB | 30 s |
| DFT-MD | 5×5 | 30 | 2500 | 8 GB | 45 s |
| Large DFT | 6×6 | 35 | 4000 | 8 GB | 60 s |
| Classical MD | 8×8 | 40 | 8000 | 16 GB | 120 s |
| Large MD | 10×10 | 50 | 15000 | 32 GB | 300 s |
"""
print(sizes)

# =============================================================================
# COMMANDS FOR DIFFERENT SIZES
# =============================================================================

print("\n2. COMMAND EXAMPLES:\n")

commands = """
# Quick test (minimal)
python electrode_solvation.py --supercell 2 2 --water-height 15 --np 2

# Standard DFT setup
python electrode_solvation.py --supercell 4 4 --water-height 25 --np 4

# Large DFT setup
python electrode_solvation.py --supercell 6 6 --water-height 35 --np 8

# Classical MD setup
python electrode_solvation.py --supercell 8 8 --water-height 40 --np 16

# Very large (multi-node HPC)
python electrode_solvation.py --supercell 12 12 --water-height 60 --np 64
"""
print(commands)

# =============================================================================
# HPC JOB SCRIPTS
# =============================================================================

print("\n3. HPC JOB SCRIPTS:\n")

slurm_script = '''
#!/bin/bash
#SBATCH --job-name=large_electrode
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load python/3.10
module load openmpi/4.1.4
source ~/electrode-env/bin/activate

python electrode_solvation.py \\
    --supercell 8 8 \\
    --water-height 40 \\
    --concentration 0.1 \\
    --np 16 \\
    --lammps lmp_mpi \\
    --output IrO2_110_large
'''
print("SLURM script for large system:")
print(slurm_script)

# =============================================================================
# MEMORY OPTIMIZATION
# =============================================================================

print("\n4. MEMORY OPTIMIZATION TIPS:\n")

tips = """
a) Reduce number of MPI processes if memory is limited:
   python electrode_solvation.py --np 2  # Uses less memory per process

b) Run LAMMPS in serial for small systems:
   python electrode_solvation.py --np 1 --mpirun ""

c) Increase water equilibration time for better structure (slower):
   # Edit WaterConfig in the script:
   WATER.final_equil_time = 50.0  # ps (default: 10)

d) For very large systems, consider pre-equilibrated water boxes:
   # Use GROMACS or CHARMM-GUI to generate water, then import

e) Monitor memory usage:
   top -p $(pgrep -f lmp)  # Watch LAMMPS memory
"""
print(tips)

# =============================================================================
# BATCH GENERATION
# =============================================================================

print("\n5. BATCH GENERATION (multiple configurations):\n")

batch_script = '''
#!/bin/bash
# Generate multiple surface orientations

for facet in "1 1 0" "1 0 0" "1 0 1"; do
    name=$(echo $facet | tr ' ' '')
    python electrode_solvation.py \\
        --facet $facet \\
        --supercell 4 4 \\
        --output IrO2_${name}
done

# Generate different concentrations
for conc in 0.05 0.1 0.2 0.5; do
    python electrode_solvation.py \\
        --concentration $conc \\
        --output IrO2_110_${conc}M
done
'''
print("Batch generation script:")
print(batch_script)

# =============================================================================
# SCALING PERFORMANCE
# =============================================================================

print("\n6. SCALING PERFORMANCE:\n")

scaling = """
LAMMPS parallel scaling (approximate):

| Processes | 4×4 system | 8×8 system |
|-----------|------------|------------|
| 1 | 60 s | 300 s |
| 2 | 35 s | 160 s |
| 4 | 20 s | 90 s |
| 8 | 15 s | 50 s |
| 16 | 12 s | 30 s |

Note: Scaling efficiency decreases for small systems with many processes.
For best efficiency:
- ~500-1000 atoms per MPI process
- Don't use more processes than atoms / 100
"""
print(scaling)

print("\n" + "=" * 60)
print("  Run with: python electrode_solvation.py --supercell X Y")
print("=" * 60)
