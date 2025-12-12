# Installation Guide

This guide covers installation on Linux, macOS, and HPC clusters.

## Table of Contents

- [Quick Install (All Platforms)](#quick-install-all-platforms)
- [Linux Installation](#linux-installation)
- [macOS Installation](#macos-installation)
- [HPC Cluster Installation](#hpc-cluster-installation)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

---

## Quick Install (All Platforms)

```bash
# 1. Clone repository
git clone https://github.com/ARPeeketi/electrode-solvation-generator.git
cd electrode-solvation-generator

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Verify
python electrode_solvation.py --help
```

---

## Linux Installation

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip

# Install LAMMPS
sudo apt install lammps

# Install MPI (optional, for parallel)
sudo apt install openmpi-bin libopenmpi-dev

# Install Python packages
pip3 install ase numpy

# Clone and test
git clone https://github.com/ARPeeketi/electrode-solvation-generator.git
cd electrode-solvation-generator
python3 electrode_solvation.py --help
```

### CentOS/RHEL/Rocky Linux

```bash
# Enable EPEL repository
sudo dnf install epel-release

# Install Python
sudo dnf install python3 python3-pip

# Install LAMMPS (may need to compile from source)
sudo dnf install openmpi openmpi-devel
module load mpi/openmpi-x86_64

# Compile LAMMPS from source (see HPC section)

# Install Python packages
pip3 install ase numpy
```

### Arch Linux

```bash
sudo pacman -S python python-pip python-numpy openmpi
pip install ase

# LAMMPS from AUR
yay -S lammps
```

---

## macOS Installation

### Using Homebrew (Recommended)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python
brew install open-mpi
brew install lammps

# Install Python packages
pip3 install ase numpy

# Clone and test
git clone https://github.com/ARPeeketi/electrode-solvation-generator.git
cd electrode-solvation-generator
python3 electrode_solvation.py --help
```

### Using Conda

```bash
# Install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # Apple Silicon
# or
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh  # Intel

bash Miniconda3-latest-*.sh

# Create environment
conda create -n electrode python=3.10
conda activate electrode

# Install packages
conda install -c conda-forge ase numpy
conda install -c conda-forge lammps

# Test
python electrode_solvation.py --help
```

### Compiling LAMMPS on macOS (for custom packages)

```bash
# Clone LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps/src

# Enable required packages
make yes-molecule
make yes-kspace
make yes-rigid

# Compile (Apple Silicon)
make mac -j8

# Or with MPI
brew install open-mpi
make mpi -j8

# Add to PATH
echo 'export PATH="$HOME/lammps/src:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## HPC Cluster Installation

### Module-based Systems (SLURM/PBS)

```bash
# Typical module loading
module load python/3.10
module load openmpi/4.1.4
module load lammps/stable

# Create virtual environment
python -m venv ~/electrode-env
source ~/electrode-env/bin/activate

# Install packages
pip install ase numpy

# Clone repository
cd ~/software
git clone https://github.com/ARPeeketi/electrode-solvation-generator.git
```

### Compiling LAMMPS on HPC

```bash
# Load modules
module load gcc/11.2.0
module load openmpi/4.1.4
module load fftw/3.3.10

# Clone LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps/src

# Enable packages
make yes-molecule
make yes-kspace
make yes-rigid
make yes-misc

# Compile with MPI
make mpi -j16

# Install to custom location
mkdir -p ~/local/bin
cp lmp_mpi ~/local/bin/
echo 'export PATH="$HOME/local/bin:$PATH"' >> ~/.bashrc
```

### Example SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=electrode_solv
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=short

# Load modules
module load python/3.10
module load openmpi/4.1.4
module load lammps/stable

# Activate environment
source ~/electrode-env/bin/activate

# Run
cd ~/electrode-solvation-generator
python electrode_solvation.py \
    --facet 1 1 0 \
    --water-height 30 \
    --np 8 \
    --lammps lmp_mpi \
    --output IrO2_110_large
```

### Example PBS Job Script

```bash
#!/bin/bash
#PBS -N electrode_solv
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:30:00
#PBS -q short

cd $PBS_O_WORKDIR

module load python/3.10
module load openmpi/4.1.4
source ~/electrode-env/bin/activate

python electrode_solvation.py --np 8 --lammps lmp_mpi
```

---

## Verifying Installation

### Check Python Dependencies

```bash
python -c "import ase; print(f'ASE version: {ase.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

### Check LAMMPS

```bash
# Check if LAMMPS is available
which lmp
# or
which lmp_mpi

# Check LAMMPS packages
lmp -h | grep -i molecule
lmp -h | grep -i kspace
```

### Test Run

```bash
cd electrode-solvation-generator

# Run with minimal system (fast test)
python electrode_solvation.py \
    --supercell 2 2 \
    --water-height 15 \
    --np 2 \
    --output test_output

# Check output
ls -la test_output*
```

---

## Troubleshooting

### "LAMMPS executable not found"

```bash
# Find LAMMPS
which lmp
which lmp_mpi
locate lmp

# Specify full path
python electrode_solvation.py --lammps /full/path/to/lmp
```

### "mpirun not found"

```bash
# Check MPI installation
which mpirun
which mpiexec

# Install OpenMPI
# Ubuntu: sudo apt install openmpi-bin
# macOS: brew install open-mpi

# Or run serial (single process)
python electrode_solvation.py --np 1 --mpirun ""
```

### "ModuleNotFoundError: No module named 'ase'"

```bash
# Install ASE
pip install ase

# Or with conda
conda install -c conda-forge ase

# Check installation
python -c "import ase; print(ase.__version__)"
```

### "LAMMPS error: Unknown pair style lj/cut/coul/long"

LAMMPS needs the KSPACE package:

```bash
cd lammps/src
make yes-kspace
make mpi -j4
```

### "Memory error" or "Killed"

System too large for available RAM:

```bash
# Reduce system size
python electrode_solvation.py --supercell 3 3 --water-height 20

# Or increase available memory
# (on HPC, request more memory in job script)
```

### "Permission denied" errors

```bash
# Make script executable
chmod +x electrode_solvation.py

# Check file permissions
ls -la electrode_solvation.py

# Run with python explicitly
python electrode_solvation.py
```

### macOS: "Library not loaded: libmpi.dylib"

```bash
# Reinstall OpenMPI
brew reinstall open-mpi

# Or use conda environment
conda activate electrode
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/ARPeeketi/electrode-solvation-generator/issues)
2. Open a new issue with:
   - Operating system and version
   - Python version (`python --version`)
   - Error message (full traceback)
   - Command you ran
