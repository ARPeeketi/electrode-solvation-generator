# Examples

This directory contains example scripts demonstrating various use cases.

## Files

| File | Description |
|------|-------------|
| `basic_usage.py` | Simplest usage - IrO2(110) + water |
| `pt_solvation.py` | Adapting for FCC metals (Pt, Au, Pd) |
| `ruo2_solvation.py` | Other rutile oxides (RuO2, TiO2, SnO2) |
| `different_ions.py` | Various electrolytes and adding new ions |
| `custom_adsorbates.py` | Adding new adsorbate species |
| `large_system.py` | Building larger systems for MD |

## Running Examples

```bash
# From repository root
cd examples
python basic_usage.py

# Or run from command line
python ../electrode_solvation.py --facet 1 1 0
```

## Quick Reference

### Change Surface
```bash
python electrode_solvation.py --facet 1 0 0   # (100)
python electrode_solvation.py --facet 1 0 1   # (101)
```

### Change Size
```bash
python electrode_solvation.py --supercell 6 6 --water-height 30
```

### Change Electrolyte
```bash
python electrode_solvation.py --cation K --anion Cl --concentration 0.5
```

### Different Oxide
Edit lattice parameters in `electrode_solvation.py`:
```python
SLAB = SlabConfig(
    a_lattice=4.49,  # RuO2
    c_lattice=3.11,
)
```
