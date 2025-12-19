# Constant Potential Molecular Dynamics: Complete Expert Analysis

## Executive Summary

This document provides a comprehensive analysis of constant potential molecular dynamics (CP-MLMD) implementation strategies for the IrO₂ OER project. It synthesizes input from four expert perspectives: electrochemistry, theoretical chemistry, ML potential architecture, and electrolyte modeling.

**Key Findings:**
1. Standard MLIPs (MACE, NequIP) are "charge-implicit" → incompatible with fix electrode
2. Charge-aware MLIPs must predict either (a) charges directly, or (b) electronegativity/hardness for QEq
3. Six implementation strategies are viable, each with distinct tradeoffs
4. Recommended approach: staged implementation starting with HIPPYNN + QEq wrapper

---

## Expert Panel

| Expert | Domain | Key Focus |
|--------|--------|-----------|
| **Dr. E** | Electrochemist | EDL physics, OER mechanisms, experimental validation |
| **Dr. T** | Theoretical Chemist | DFT, charge partitioning, ensemble theory |
| **Dr. M** | ML Potential Architect | NN architectures, charge prediction, HIPPYNN/MACE |
| **Dr. S** | Electrolyte Modeler | Solvation, ion transport, GCMC |

---

## Part 1: Theoretical Foundations

### 1.1 Why Constant Potential is Fundamental

Real electrodes connected to a potentiostat operate at **fixed potential Φ** with **fluctuating charge Q**. Standard NVT-MD fixes charge, which is fundamentally wrong.

**Three Pillars of Evidence:**

| Pillar | Research Group | Key Insight |
|--------|----------------|-------------|
| Jones (Operando XAS) | Cambridge | Bias causes charge accumulation, not just field. ΔG‡ ∝ stored oxidative charge |
| Durrant (TAS) | Imperial | OER is collective, 3rd-order in holes. Requires long timescales + fluctuating charges |
| Goddard (GC-DFT) | Caltech | Electrode is open system (μₑ fixed). Fixed-charge MD is thermodynamically inconsistent |

### 1.2 Electrochemical Ensembles

| Ensemble | Fixed | Fluctuating | Use Case |
|----------|-------|-------------|----------|
| **NVT** | N, V, T | E | Standard MD (WRONG for electrodes) |
| **μVT** | μ, V, T | N, E | GCMC for particle reservoir |
| **ΦVT** | Φ, V, T | Q, E | **Constant potential (correct)** |

**Mathematical Formulation:**

```
Helmholtz free energy in ΦVT:
F(Φ,V,T) = E - TS - ΦQ

Equilibrium: ∂F/∂Q = 0 → ∂E/∂Q = Φ
```

### 1.3 Capacitive vs Faradaic Charging

| Type | Physics | Correlates With |
|------|---------|-----------------|
| Capacitive | EDL rearrangement (Q = C·ΔU) | Nothing catalytic |
| Faradaic/Oxidative | Redox (Ir³⁺→Ir⁴⁺→Ir⁵⁺) | **OER activity** |

### 1.4 Metallic Screening

```
Thomas-Fermi Screening Length:
λ_TF = √(ε₀ / (e² D(E_F))) ≈ 0.5-1 Å for metals

→ External field penetrates only ~1 Å into metal
→ Charge accumulates on surface
→ Interior remains field-free (equipotential)
```

---

## Part 2: ML Potential Requirements

### 2.1 Required Outputs

| Output | Standard MLIP | CP-MLIP Required |
|--------|---------------|------------------|
| Energy E | ✓ | ✓ |
| Forces F | ✓ | ✓ |
| Stress σ | ✓ | ✓ |
| **Atomic charges qᵢ** | ✗ | ✓ |
| **Electronegativity χᵢ** | ✗ | ✓ (for QEq) |
| **Hardness ηᵢ** | ✗ | ✓ (for QEq) |
| **Charge derivatives ∂qᵢ/∂rⱼ** | ✗ | ✓ (force corrections) |

### 2.2 Three Levels of Charge Prediction

**Level 1: Fixed Charges**
```
qᵢ = constant
Problem: No response to environment or potential
```

**Level 2: Environment-Dependent (Static)**
```
qᵢ = NN(Gᵢ) where Gᵢ = local descriptor
Problem: No response to applied potential
Example: HIPPYNN HChargeNode
```

**Level 3: Self-Consistent (Dynamic)**
```
χᵢ = NN(Gᵢ), ηᵢ = NN(Gᵢ) or fixed
qᵢ = QEq_solve({χᵢ}, {ηᵢ}, Φ_applied)
Advantage: Full potential response
Example: 4G-HDNNP, CP-MACE
```

### 2.3 Charge Partitioning

| Method | Basis | Recommendation |
|--------|-------|----------------|
| Hirshfeld | Stockholder sharing | Underestimates ionic |
| **DDEC6** | Optimized stockholder | **Recommended for IrO₂** |
| Bader | Zero-flux surfaces | Grid-sensitive |
| Mulliken | Orbital populations | Basis-dependent |

---

## Part 3: Implementation Strategies

### Strategy 1: Classical QEq with Fixed Parameters

**Energy Functional:**
```
E_QEq = Σᵢ [χᵢ⁰ qᵢ + ½ηᵢ qᵢ²] + ½ Σᵢ≠ⱼ Jᵢⱼ qᵢqⱼ

Matrix form:
A·q = b where Aᵢⱼ = Jᵢⱼ + δᵢⱼηᵢ, bᵢ = -χᵢ⁰
```

**LAMMPS:**
```lammps
fix qeq all qeq/slater 1 10 1.0e-6 200 param.qeq
```

| Pros | Cons |
|------|------|
| Fast, stable | Fixed parameters |
| Well-tested | No environment dependence |

---

### Strategy 2: 4G-HDNNP (Environment-Dependent QEq)

**Architecture:**
```
Total Energy: E = E_short(R,Q) + E_elec(R,Q)

Electronegativity: χᵢ = NN_χ(Gᵢ)
Hardness: ηᵢ = fixed per element

Charges from QEq with environment-dependent χᵢ
```

**Recent Advancement (iQEq, 2025):**
- O(N²) instead of O(N³)
- Converges in 5-10 iterations
- Enables MD with >1000 atoms

| Pros | Cons |
|------|------|
| Most rigorous physics | Complex training |
| Environment-dependent χᵢ | n2p2 only |

---

### Strategy 3: HIPPYNN HChargeNode + QEq Wrapper

**From hippynn/layers/targets.py:**
```python
class HCharge(torch.nn.Module):
    def forward(self, all_features):
        partial_charges = [lay(x) for x, lay in zip(all_features, self.layers)]
        return sum(partial_charges)
```

**Wrapper for CP-MLMD:**
```python
q_predicted = hippynn.predict_charges(geometry)
chi_estimate = -partial_derivative(E, q)
q_final = qeq_solve(chi_estimate, eta_fixed, Phi)
```

| Pros | Cons |
|------|------|
| Existing infrastructure | Static charges |
| Proven for IrO₂ | Post-hoc wrapper |

---

### Strategy 4: CP-MACE (Potential as Input)

**Architecture:**
```
Input: (positions R, electron number Nₑ)
Output: E(R, Nₑ), F(R, Nₑ), E_Fermi(R, Nₑ)

Constant potential: solve for Nₑ that gives target Φ
```

| Pros | Cons |
|------|------|
| True constant-potential | Requires retraining |
| E(3)-equivariant | LAMMPS integration nascent |

---

### Strategy 5: GCMC (Particle Reservoir)

**CHE Chemical Potential:**
```
μ_H(U) = ½(E_H₂ + ZPE - TS) - e·U

Where: E_H₂ ≈ -6.75 eV, ZPE = 0.27 eV, TS = 0.41 eV
```

**LAMMPS:**
```lammps
variable mu equal 0.5*(-6.75 + 0.27 - 0.41) - ${U_applied}
fix gcmc region gcmc 100 100 0 1 54321 298.15 ${mu} 0.5 full_energy
```

| Pros | Cons |
|------|------|
| Thermodynamically rigorous | No electron transfer |
| Mature implementation | Low acceptance at high Φ |

---

### Strategy 6: Hybrid fix electrode + GCMC

**Dual Reservoir:**
```
Electrode: electron reservoir at Φ (fix electrode)
Electrolyte: particle reservoir at μ (GCMC + QEq)
Coupling: Coulomb interactions
```

**LAMMPS:**
```lammps
pair_style hybrid/overlay mliap unified model.pt coul/long 12.0
kspace_style pppm/electrode 1.0e-5
fix conp electrode electrode/conp ${U} 1.805 symm on
fix qeq electrolyte qeq/point 1 10 1.0e-6 200 param.qeq
fix gcmc interface gcmc 100 100 0 3 12345 298.15 ${mu} 0.5 full_energy
```

| Pros | Cons |
|------|------|
| Complete physics | Many moving parts |
| Capacitive + faradaic | Complex debugging |

---

## Part 4: Strategy Comparison

| Aspect | QEq | 4G-HDNNP | HIPPYNN+QEq | CP-MACE | GCMC | Hybrid |
|--------|-----|----------|-------------|---------|------|--------|
| **χᵢ source** | Fixed | NN(Gᵢ) | Post-fit | NN(R,Nₑ) | N/A | Mixed |
| **Potential response** | Via Φ | Via χ(Gᵢ)+Φ | Approximate | Exact | Via μ | fix electrode |
| **Computational cost** | O(N²-N³) | O(N²) iQEq | O(N)+wrapper | O(N)+opt | O(N)+MC | O(N)+O(N²) |
| **LAMMPS ready** | ✓ | ~ (n2p2) | ~ (wrapper) | ✗ | ✓ | ✓ |
| **IrO₂ validated** | ✗ | ✗ | In progress | ✗ | ✗ | ✗ |

---

## Part 5: Recommendations for IrO₂

### 5.1 Staged Roadmap

**Phase 1 (0-6 months): HIPPYNN + fix electrode**
- Train HIPPYNN with HChargeNode
- Implement QEq wrapper for LAMMPS
- Validate capacitance (~40 μF/cm²)

**Phase 2 (6-12 months): 4G-HDNNP exploration**
- Port to n2p2-LAMMPS
- Train electronegativity network
- Benchmark iQEq performance

**Phase 3 (12+ months): Production CP-MLMD**
- Add GCMC for ion thermodynamics
- Validate OER mechanism
- Compute Tafel slopes

### 5.2 LAMMPS Template

```lammps
units           metal
atom_style      full
boundary        p p f

pair_style      hybrid/overlay &
                mliap unified hippynn_iro2.pt &
                coul/long 12.0

kspace_style    pppm/electrode 1.0e-5

group electrode type 1 2        # Ir, O (slab)
group electrolyte type 3 4 5 6  # O, H, Na, Cl

fix conp electrode electrode/conp ${U_applied} 1.805 symm on
fix qeq electrolyte qeq/point 1 10 1.0e-6 200 chi_electrolyte.txt
```

### 5.3 Validation Targets

| Property | Experimental | Target |
|----------|--------------|--------|
| Differential capacitance | ~40 μF/cm² | 30-50 μF/cm² |
| OER onset | ~1.5 V vs RHE | 1.4-1.6 V |
| Tafel slope | 40-60 mV/dec | 40-80 mV/dec |
| Charge RMSE | — | < 0.05 e |
| Force RMSE | — | < 50 meV/Å |

---

## References

1. [Constant potential MD theory (arXiv)](https://arxiv.org/html/2308.01740)
2. [4G-HDNNP (Nature Comms 2021)](https://www.nature.com/articles/s41467-020-20427-2)
3. [iQEq for 4G-HDNNP (arXiv 2025)](https://arxiv.org/html/2502.07907v1)
4. [CP-MACE (JCTC 2025)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00784)
5. [LAMMPS electrode package](https://github.com/lammps/lammps/pull/3544)
6. [Electrochemical capacitance (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11077530/)
7. [ML-enhanced constant potential (Nature Comms 2025)](https://www.nature.com/articles/s41467-025-62824-5)
8. [ML Force Fields in Electrochemistry (ACS Nano)](https://pubs.acs.org/doi/10.1021/acsnano.5c05553)

---

*Generated: 2025-12-19*
*Expert Panel: ARIA (Advanced Reactive Interface Assistant)*
