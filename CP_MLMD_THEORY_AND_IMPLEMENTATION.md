# Constant Potential Molecular Dynamics: Complete Theoretical Formulation and Implementation

**Comprehensive guide to CP-MLMD theory, equations, and LAMMPS implementation**

---

## Table of Contents

1. [Thermodynamic Ensembles](#thermodynamic-ensembles)
2. [Constant Potential Method (CPM) Theory](#constant-potential-method-cpm-theory)
3. [Charge Equilibration Methods](#charge-equilibration-methods)
4. [MLIP Charge Prediction Theory](#mlip-charge-prediction-theory)
5. [Energy and Force Expressions](#energy-and-force-expressions)
6. [Numerical Implementation](#numerical-implementation)
7. [LAMMPS Implementation Details](#lammps-implementation-details)
8. [HIPPYNN Implementation](#hippynn-implementation)
9. [Complete Workflow](#complete-workflow)

---

## 1. Thermodynamic Ensembles

### Standard Ensembles

| Ensemble | Fixed Variables | Fluctuating | MD Algorithm |
|----------|-----------------|-------------|--------------|
| **NVE** | N, V, E | Nothing | Velocity Verlet |
| **NVT** | N, V, T | E | + Thermostat |
| **NPT** | N, P, T | V, E | + Barostat |
| **μVT** | μ, V, T | N, E | Grand canonical MC/MD |
| **ΦVT** | Φ (potential), V, T | Q (charge), E | **Constant potential MD** |

### Electrochemical Ensemble (ΦVT)

**Fixed:**
- **Φ** = Electrode potential (V vs reference)
- **V** = Volume
- **T** = Temperature

**Fluctuating:**
- **Q** = Total electrode charge
- **E** = Total energy
- **N** = Number of particles (if coupled with μVT)

**Physical meaning:** Simulates experimental potentiostat conditions.

---

## 2. Constant Potential Method (CPM) Theory

### 2.1 Fundamental Energy Expression

Total system energy in the constant potential ensemble:

```
E_total = E_short-range + E_electrostatic - Σₐ Φₐ Qₐ
```

Where:
- **E_short-range**: Short-range interactions (bonded, vdW, etc.)
- **E_electrostatic**: Coulomb energy
- **Φₐ**: Applied potential on electrode α
- **Qₐ**: Total charge on electrode α
- **Σₐ Φₐ Qₐ**: Work done by potentiostat (removes double-counting)

**Key Reference:** Ahrens-Iwers & Meissner, *J. Chem. Phys.* **155**, 104104 (2021)

### 2.2 Equipotential Constraint

For a metallic electrode with atoms i ∈ α:

```
φᵢ = Φₐ    for all i ∈ electrode α
```

Where φᵢ is the local electrostatic potential at atom i.

**Implementation:** This constraint is enforced by solving for electrode charges that minimize total energy.

### 2.3 Electrode Charge-Potential Relationship

The capacitance matrix equation:

```
Q = Q₀ᵥ + C·V
```

**Expanded form:**
```
┌ Q₁ ┐   ┌ Q₁⁰ ┐   ┌ C₁₁  C₁₂  ...  C₁ₙ ┐ ┌ Φ₁ ┐
│ Q₂ │ = │ Q₂⁰ │ + │ C₂₁  C₂₂  ...  C₂ₙ │ │ Φ₂ │
│ ⋮  │   │  ⋮  │   │  ⋮    ⋮   ⋱    ⋮  │ │ ⋮  │
└ Qₙ ┘   └ Qₙ⁰ ┘   └ Cₙ₁  Cₙ₂  ...  Cₙₙ ┘ └ Φₙ ┘
```

**Terms:**
- **Q₀ᵥ**: Charge at zero potential (depends on electrolyte configuration)
- **Cₐₐ**: Self-capacitance of electrode α
- **Cₐᵦ**: Mutual capacitance between electrodes α and β (negative)

**Physical meaning:**
- Diagonal elements Cₐₐ > 0: Increasing Φₐ increases Qₐ
- Off-diagonal Cₐᵦ < 0: Increasing Φᵦ decreases Qₐ (screening)

### 2.4 Capacitance Matrix Calculation

**Gaussian charge distribution model:**

Each electrode atom i has charge density:
```
ρᵢ(r) = qᵢ / (σ√(2π))³ exp(-|r - rᵢ|² / (2σ²))
```

Where:
- **qᵢ**: Point charge on atom i
- **σ = 1/η**: Gaussian width (η from LAMMPS input)

**Matrix elements from electrostatics:**

```
Cₐᵦ = Σᵢ∈ₐ Σⱼ∈ᵦ ∂φᵢ/∂qⱼ |ᵥ
```

Where φᵢ is electrostatic potential at atom i due to all charges.

**Self-capacitance contribution:**
```
∂φᵢ/∂qᵢ = 1/(4πε₀) · erf(ηrᵢⱼ)/rᵢⱼ + self-energy term
```

**LAMMPS implementation:** Pre-calculates C matrix using `kspace_style pppm/electrode`.

### 2.5 Energy Minimization

At each MD step, find electrode charges {qᵢ} that minimize:

```
F[{qᵢ}] = E_Coulomb[{qᵢ}] - Σₐ Φₐ Qₐ[{qᵢ}]

subject to: Σᵢ∈ₐ qᵢ = Qₐ  (fixed total charge per electrode)
           φᵢ = Φₐ     (equipotential constraint)
```

**Lagrange multiplier formulation:**
```
∂F/∂qᵢ = ∂E_Coulomb/∂qᵢ - Φₐ = 0    for i ∈ α
```

This gives **φᵢ = Φₐ** automatically (equipotential achieved).

### 2.6 Electrode vs Electrolyte Atoms

**Electrode atoms:**
- Charges {qᵢ} fluctuate to maintain Φₐ
- Update every MD step via capacitance matrix

**Electrolyte atoms:**
- Charges fixed (if using rigid model)
- OR charges from QEq (if using polarizable/reactive model)
- Positions evolve via Newtonian dynamics

**Critical:** Electrode and electrolyte charges are coupled through Coulomb interactions, creating the electric double layer (EDL).

---

## 3. Charge Equilibration Methods

### 3.1 QEq (Charge Equilibration) Method

**Physical basis:** Sanderson electronegativity equalization principle.

**Energy functional per atom:**
```
Eᵢ(qᵢ) = χᵢ⁰ qᵢ + ½ ηᵢ qᵢ² + Σⱼ≠ᵢ Jᵢⱼ qᵢqⱼ
```

Where:
- **χᵢ⁰**: Intrinsic electronegativity of atom i (eV)
- **ηᵢ**: Self-Coulomb potential / hardness (eV)
- **Jᵢⱼ**: Coulomb interaction between atoms i and j (eV)

**Electronegativity equalization:**

At equilibrium, all atoms have same electronegativity:
```
χᵢ = ∂Eᵢ/∂qᵢ = χᵢ⁰ + 2ηᵢ qᵢ + Σⱼ Jᵢⱼ qⱼ = constant
```

**Matrix form:**
```
┌ 2η₁   J₁₂  ...  J₁ₙ   1 ┐ ┌ q₁ ┐   ┌ -χ₁⁰ ┐
│ J₂₁   2η₂  ...  J₂ₙ   1 │ │ q₂ │   │ -χ₂⁰ │
│  ⋮     ⋮    ⋱    ⋮    ⋮ │ │ ⋮  │ = │  ⋮   │
│ Jₙ₁   Jₙ₂  ...  2ηₙ  1 │ │ qₙ │   │ -χₙ⁰ │
└  1     1   ...   1   0 ┘ └ λ  ┘   └  Q   ┘
```

Where:
- **λ**: Lagrange multiplier (system electronegativity)
- **Q**: Total charge constraint (Σᵢ qᵢ = Q)

**Solve:** Standard linear algebra (LU decomposition, iterative CG)

### 3.2 Coulomb Interaction Matrix Jᵢⱼ

**Three QEq variants differ in Jᵢⱼ:**

#### (a) Point Charges (qeq/point)
```
Jᵢⱼ = 1/(4πε₀ rᵢⱼ)    for i ≠ j
```

Simple but can diverge at short distances.

#### (b) Shielded Coulomb (qeq/shielded)
```
Jᵢⱼ = 1/(4πε₀) · erf(γᵢⱼ rᵢⱼ) / rᵢⱼ
```

Where γᵢⱼ = √(γᵢ² + γⱼ²) is combined shielding parameter.

**Physical meaning:** Accounts for electron cloud overlap.

#### (c) Slater Orbitals (qeq/slater)
```
Jᵢⱼ = 1/(4πε₀) ∫∫ ρᵢ(r₁) ρⱼ(r₂) / |r₁ - r₂| dr₁ dr₂
```

Where ρᵢ(r) = (ζᵢ³/π) exp(-2ζᵢ|r - rᵢ|) is Slater 1s orbital.

**Closed form:**
```
Jᵢⱼ = 1/(4πε₀) · [1/rᵢⱼ - (1 + 11ζᵢⱼrᵢⱼ/8) exp(-2ζᵢⱼrᵢⱼ)]
```

Where ζᵢⱼ = 2(ζᵢζⱼ)/(ζᵢ + ζⱼ).

### 3.3 ACKS2 (Atom-Condensed Kohn-Sham 2nd Order)

**Improvement over QEq:** Prevents unphysical long-range charge transfer.

**Energy functional:**
```
E = Σᵢ (χᵢ⁰ qᵢ + ½ ηᵢ qᵢ²) + ½ Σᵢⱼ Jᵢⱼ qᵢqⱼ + E_bond
```

**Bond energy (new term):**
```
E_bond = -½ b_s Σ_bonds (qᵢqⱼ / (rᵢⱼ/r₀)ᵖ)
```

Where:
- **b_s**: Bond softness parameter (global)
- **r₀**: Reference bond length
- **p**: Power (typically 3)

**Equilibrium condition:**
```
χᵢ = χᵢ⁰ + 2ηᵢ qᵢ + Σⱼ [Jᵢⱼ - b_s·δ_bond(i,j)/(rᵢⱼ/r₀)ᵖ] qⱼ = constant
```

**Key difference:** Bond term **subtracts** from Coulomb for bonded pairs, preventing charge separation at large r.

**Reference:** Verstraelen et al., *J. Chem. Phys.* **138**, 074108 (2013)

### 3.4 Parameters for Ir-O-H-Na-Cl

**QEq parameters (typical values):**

| Atom | χ⁰ (eV) | η (eV) | γ (Å⁻¹) | Source |
|------|---------|--------|---------|--------|
| Ir | 9.0 | 8.5 | 1.2 | ReaxFF-fitted |
| O | 8.5 | 8.3 | 1.0 | Water models |
| H | 4.5 | 6.0 | 1.8 | TIP4P/ReaxFF |
| Na | 2.8 | 3.0 | 0.8 | Alkali fit |
| Cl | 9.5 | 7.5 | 1.1 | Halide fit |

**Note:** These are **approximate**. For accurate CP-MLMD, fit from DFT using:
```
χᵢ⁰ = -∂E_DFT/∂qᵢ |q=0
ηᵢ = ∂²E_DFT/∂qᵢ² |q=0
```

**ACKS2 parameters:**
- Bond softness: b_s ~ 0.5-1.5 (system-dependent)
- Cutoff: 2.5× covalent radius

---

## 4. MLIP Charge Prediction Theory

### 4.1 Direct Charge Prediction (HChargeNode)

**HIPPYNN approach:** Predict partial charges from local atomic environments.

**Architecture:**
```
Input: Atomic positions {rᵢ}
       ↓
Hierarchical feature extraction (GNN)
       ↓
Local features hᵢ
       ↓
Linear layer per atom
       ↓
Output: Partial charges qᵢ
```

**Mathematical form:**
```
qᵢ = Σₗ Wₗ · hᵢ^(l) + bᵢ
```

Where:
- **hᵢ^(l)**: Features at hierarchy level l
- **Wₗ**: Learned weights
- **bᵢ**: Per-atom bias

**Constraint:** Total charge conservation enforced during training:
```
Loss_charge = || Σᵢ qᵢ - Q_total ||²
```

**Limitation:** Predicted charges are **static** (don't respond to applied potential).

### 4.2 Electronegativity Prediction (χ, η Approach)

**4G-HDNNP / CP-MACE approach:** Predict charge response parameters.

**Architecture:**
```
Input: Atomic positions {rᵢ}, applied potential Φ
       ↓
Environment descriptor Gᵢ
       ↓
Neural network
       ↓
Electronegativity: χᵢ = NN_χ(Gᵢ)
Hardness: ηᵢ = NN_η(Gᵢ) [optional]
       ↓
QEq solver: Σⱼ A_ij q_j = b_i
       ↓
Output: Potential-dependent charges qᵢ(Φ)
```

**Electronegativity functional form:**
```
χᵢ = χᵢ⁰ + Σₗ Wₗ^χ · σ(Wₗ^h · Gᵢ)
```

**Hardness (if predicted):**
```
ηᵢ = ηᵢ⁰ + Σₗ Wₗ^η · σ(Wₗ^h · Gᵢ)
```

**Charge equilibration at MD step t:**
```
χᵢ(Gᵢ) + 2ηᵢ qᵢ^(t) + Σⱼ Jᵢⱼ qⱼ^(t) = λ^(t)
```

Where λ^(t) adjusts to satisfy:
- Equipotential on electrodes
- Total charge conservation

**Advantage:** Charges **respond to potential changes** automatically.

### 4.3 Charge Energy Coupling

**Local charge energy contribution** (HIPPYNN LocalChargeEnergy):

```
E_charge = Σᵢ [½ κᵢ qᵢ² + μᵢ qᵢ]
```

Where:
- **κᵢ**: Quadratic coefficient (like ηᵢ)
- **μᵢ**: Linear coefficient (like χᵢ)

**Training:** Fit κᵢ and μᵢ to reproduce DFT energy differences when charges vary.

**Implementation in HIPPYNN:**
```python
atom_charge_energy = ½ * quad_term * charges² + linear_term * charges
```

**Total energy:**
```
E_total = E_short + E_electrostatic + E_charge
```

---

## 5. Energy and Force Expressions

### 5.1 Total Energy Decomposition

**Full energy in constant potential MD:**

```
E_total = E_bonds + E_angles + E_dihedrals
        + E_vdW
        + E_electrostatic
        + E_MLIP_local
        - Σₐ Φₐ Qₐ
```

**Terms:**

1. **E_bonds, E_angles, E_dihedrals**: Bonded interactions (if using force field overlay)

2. **E_vdW**: van der Waals (Lennard-Jones):
   ```
   E_vdW = Σᵢⱼ 4εᵢⱼ [(σᵢⱼ/rᵢⱼ)¹² - (σᵢⱼ/rᵢⱼ)⁶]
   ```

3. **E_electrostatic**: Coulomb energy with screening:
   ```
   E_electrostatic = ½ Σᵢⱼ qᵢ qⱼ / (4πε₀ rᵢⱼ) · f_screen(rᵢⱼ)
   ```

   Where f_screen accounts for dielectric or metallic screening.

4. **E_MLIP_local**: Neural network prediction (short-range, non-Coulombic)

5. **-Σₐ Φₐ Qₐ**: Work by potentiostat (removes double-counting of electrode energy)

### 5.2 Forces on Atoms

**Force on atom i:**
```
Fᵢ = -∂E_total/∂rᵢ
```

**Expanded:**
```
Fᵢ = Fᵢ^bonds + Fᵢ^vdW + Fᵢ^Coulomb + Fᵢ^MLIP
```

**Coulomb force (fixed charges):**
```
Fᵢ^Coulomb = -Σⱼ qᵢ qⱼ / (4πε₀ rᵢⱼ²) · r̂ᵢⱼ
```

**Coulomb force (fluctuating charges, QEq):**
```
Fᵢ^Coulomb = -∂E_Coulomb/∂rᵢ |_q_fixed - Σⱼ (∂E_Coulomb/∂qⱼ) · (∂qⱼ/∂rᵢ)
            = Fᵢ^direct + Fᵢ^induced
```

**Direct term:**
```
Fᵢ^direct = -Σⱼ qᵢ qⱼ / (4πε₀ rᵢⱼ²) · r̂ᵢⱼ
```

**Induced term (charge response to geometry change):**
```
Fᵢ^induced = -Σⱼ φⱼ · (∂qⱼ/∂rᵢ)
```

Where φⱼ is the electrostatic potential at atom j.

**Calculation of ∂qⱼ/∂rᵢ:**

From QEq equilibrium condition:
```
Σⱼ Aⱼₖ qⱼ = bₖ
```

Taking derivative w.r.t. rᵢ:
```
Σⱼ Aⱼₖ (∂qⱼ/∂rᵢ) = ∂bₖ/∂rᵢ - Σⱼ (∂Aⱼₖ/∂rᵢ) qⱼ
```

**Computational cost:** Requires solving linear system at each force evaluation (expensive!).

**LAMMPS optimization:** Uses matrix factorization, updates charges every N steps (typically N=1-10).

### 5.3 Stress Tensor (for NPT)

**Virial stress:**
```
σ_αβ = -1/V Σᵢ [mᵢ vᵢ_α vᵢ_β + Σⱼ Fᵢⱼ_α rᵢⱼ_β]
```

**Electrostatic contribution:**
```
σ_αβ^Coulomb = -1/V Σᵢⱼ qᵢ qⱼ / (4πε₀ rᵢⱼ³) · rᵢⱼ_α rᵢⱼ_β
```

**Induced charge contribution:** Additional term from ∂q/∂V (complex!).

---

## 6. Numerical Implementation

### 6.1 MD Integration Algorithm

**Velocity Verlet with constant potential:**

```
At step t:
1. Compute forces Fᵢ(t) from current positions rᵢ(t), charges qᵢ(t)
2. Update velocities: vᵢ(t+Δt/2) = vᵢ(t) + Δt/2 · Fᵢ(t)/mᵢ
3. Update positions: rᵢ(t+Δt) = rᵢ(t) + Δt · vᵢ(t+Δt/2)
4. Solve QEq for new charges qᵢ(t+Δt) given rᵢ(t+Δt)
5. Solve fix electrode for electrode charges given qᵢ(t+Δt)
6. Compute new forces Fᵢ(t+Δt)
7. Update velocities: vᵢ(t+Δt) = vᵢ(t+Δt/2) + Δt/2 · Fᵢ(t+Δt)/mᵢ
8. Apply thermostat/barostat
```

**Critical:** Steps 4-5 are coupled and iterative.

### 6.2 QEq Solver Iteration

**Conjugate Gradient (CG) method:**

```
Solve: A·q = b

Initialize: q⁰ = 0, r⁰ = b, p⁰ = r⁰
For k = 0, 1, 2, ... until ||rᵏ|| < tolerance:
    αₖ = (rᵏ·rᵏ) / (pᵏ·A·pᵏ)
    qᵏ⁺¹ = qᵏ + αₖ pᵏ
    rᵏ⁺¹ = rᵏ - αₖ A·pᵏ
    βₖ = (rᵏ⁺¹·rᵏ⁺¹) / (rᵏ·rᵏ)
    pᵏ⁺¹ = rᵏ⁺¹ + βₖ pᵏ
```

**Typical convergence:** 10-50 iterations for 10⁻⁶ tolerance.

### 6.3 Capacitance Matrix Update

**When to update C matrix:**

- Every step (expensive, most accurate)
- Every 10-100 steps (compromise)
- Only when electrolyte moves significantly (RMSD > threshold)

**LAMMPS default:** Pre-calculate at start, assume C constant (valid for rigid electrode).

**Advanced:** Update C when ΔQ/ΔΦ deviates from C by >10%.

### 6.4 Timestep Selection

**Constraints:**

1. **Bonded vibrations:** Δt < 2π/ω_max
   - O-H stretch: ω ~ 3500 cm⁻¹ → Δt < 1 fs

2. **Electrostatic relaxation:** Δt < τ_relax
   - Metal screening: τ ~ 10⁻¹⁵ s (instantaneous)
   - QEq convergence: τ ~ 10⁻¹³ s (sub-timestep)

3. **Charge equilibration:** Nevery_QEq ≤ Δt/τ_charge
   - For reactive chemistry: Nevery = 1 (every step)
   - For rigid ions: Nevery = 10 (acceptable)

**Recommendation:** Δt = 0.5-1.0 fs for electrochemistry, Nevery_QEq = 1.

---

## 7. LAMMPS Implementation Details

### 7.1 Complete LAMMPS Input Script

```lammps
# ============================================
# CONSTANT POTENTIAL MD: IrO2-WATER INTERFACE
# ============================================

# ------------------------------------
# 1. INITIALIZATION
# ------------------------------------
units           metal
atom_style      full
boundary        p p f       # Periodic x,y; non-periodic z (slab geometry)

# ------------------------------------
# 2. READ STRUCTURE
# ------------------------------------
read_data       interface.data

# Define groups
group electrode type 1 2      # Ir, O in electrode
group water type 3 4          # O, H in water
group ions type 5 6           # Na, Cl
group bot electrode id < 500  # Bottom electrode atoms
group top electrode id > 2500 # Top electrode atoms

# ------------------------------------
# 3. FORCE FIELD: MLIP + QEq
# ------------------------------------

# MLIP for short-range interactions
pair_style      hybrid/overlay &
                mliap unified model.pt Ir O H Na Cl &
                coul/long 12.0

pair_coeff      * * mliap Ir O H Na Cl
pair_coeff      * * coul/long

# Long-range electrostatics with electrode correction
kspace_style    pppm/electrode 1.0e-5
kspace_modify   slab 3.0            # Slab correction for 2D periodicity
kspace_modify   amat twostep        # Accuracy setting

# ------------------------------------
# 4. CHARGE EQUILIBRATION (QEq)
# ------------------------------------

# QEq parameters file (chi.txt):
# itype chi(eV) eta(eV) gamma(1/Angstrom)
# 1     9.0      8.5     1.2     # Ir
# 2     8.5      8.3     1.0     # O
# 3     4.5      6.0     1.8     # H
# 4     2.8      3.0     0.8     # Na
# 5     9.5      7.5     1.1     # Cl

fix f_qeq all qeq/point 1 10 1.0e-6 200 chi.txt
# Args: Nevery Niter tolerance maxiter paramfile

# ------------------------------------
# 5. CONSTANT POTENTIAL (fix electrode)
# ------------------------------------

# Apply +1.5V vs RHE equivalent (in vacuum scale: ~5.94V)
variable U_bottom equal 5.94    # Bottom electrode
variable U_top equal 5.94       # Top at same potential (zero bias)

fix conp bot electrode/conp ${U_bottom} 1.805 &
    couple top ${U_top} symm on algo mat_inv

fix_modify conp energy yes      # Include in thermodynamic energy

# ------------------------------------
# 6. DYNAMICS
# ------------------------------------

timestep        0.5             # 0.5 fs (conservative)

# Thermostat (Langevin damping on non-electrode atoms)
fix f_lang water langevin 298.0 298.0 100.0 48279
fix f_lang ions langevin 298.0 298.0 100.0 48280

# NVE integration for all atoms
fix f_nve all nve

# ------------------------------------
# 7. OUTPUT
# ------------------------------------

# Thermodynamic output
thermo_style    custom step temp pe ke etotal &
                f_conp f_f_qeq press vol
thermo          100

# Dump trajectory
dump traj all custom 1000 traj.lammpstrj &
     id type x y z q fx fy fz

# Electrode charge output
fix ave_charge bot ave/time 10 10 100 c_bot[1] &
    file electrode_charge.dat
# c_bot[1] = total charge on bottom electrode group

# Capacitance calculation
variable Q_bot equal c_bot[1]           # Bottom electrode charge
variable Q_top equal c_top[1]           # Top electrode charge
variable dV equal ${U_top}-${U_bottom}  # Potential difference
variable C_diff equal (v_Q_bot-v_Q_top)/(v_dV+1e-10)  # Differential capacitance

fix ave_cap all ave/time 10 10 100 v_C_diff &
    file capacitance.dat

# ------------------------------------
# 8. RUN
# ------------------------------------

# Equilibration (constant potential)
run             10000           # 5 ps

# Production
run             100000          # 50 ps

write_data      final.data
```

### 7.2 QEq Parameter File (chi.txt)

```
# QEq parameters for Ir-O-H-Na-Cl system
# Format: itype chi(eV) eta(eV) gamma(1/Angstrom)

1   9.000  8.500  1.200    # Ir (fitted from DFT)
2   8.500  8.300  1.000    # O  (fitted from DFT)
3   4.500  6.000  1.800    # H  (from ReaxFF water)
4   2.800  3.000  0.800    # Na (alkali metal)
5   9.500  7.500  1.100    # Cl (halide)
```

**Fitting procedure:**

```python
# Fit chi and eta from VASP
import numpy as np
from ase.io import read

# Load DFT calculations with varied charges
structures = read('dft_charge_scan.extxyz', index=':')

charges = []
energies = []
for atoms in structures:
    q = atoms.info['charge']
    E = atoms.get_potential_energy()
    charges.append(q)
    energies.append(E)

# Fit E = chi*q + 0.5*eta*q^2
from scipy.optimize import curve_fit
def E_model(q, chi, eta):
    return chi*q + 0.5*eta*q**2

popt, _ = curve_fit(E_model, charges, energies)
chi, eta = popt
print(f"chi = {chi:.3f} eV")
print(f"eta = {eta:.3f} eV")
```

### 7.3 MLIP Integration (pair_style mliap)

**LAMMPS mliap syntax:**
```lammps
pair_style mliap unified model.pt Ir O H Na Cl
pair_coeff * * Ir O H Na Cl
```

**model.pt requirements:**
- TorchScript compiled HIPPYNN model
- Input: positions, cell, pbc
- Output: energy, forces (charges optional)

**Export from HIPPYNN:**
```python
import torch
from hippynn.experiment import assemble_for_training

model, loss, target = assemble_for_training(...)

# Train model...

# Export as TorchScript
scripted = torch.jit.script(model)
scripted.save('model.pt')
```

**LAMMPS call sequence:**
```
For each MD step:
1. LAMMPS passes {rᵢ} to model.pt
2. PyTorch computes E, F (and optionally Q)
3. LAMMPS receives E, F
4. If Q predicted: use for initial QEq guess
5. QEq solver refines charges
6. fix electrode adjusts electrode charges
7. Final forces used for integration
```

---

## 8. HIPPYNN Implementation

### 8.1 Training Database Structure

**Extended XYZ format with charges:**

```extxyz
224
energy=-1456.234 Lattice="8.98 0 0 0 6.38 0 0 0 20.0" pbc="T T T" charges="0.45 0.32 ..." Properties=species:S:1:pos:R:3
Ir       0.000   0.000   5.000
Ir       4.490   3.190   5.000
O        2.245   1.595   5.500
H        2.245   1.595   6.100
...
```

**Database fields:**
- `energy`: Total DFT energy (eV)
- `charges`: Hirshfeld or DDEC6 charges (e)
- `forces`: DFT forces (eV/Å)
- `Lattice`: Cell vectors (Å)
- `pbc`: Periodic boundary conditions

### 8.2 HIPPYNN Graph Construction

```python
import hippynn
from hippynn.graphs import inputs, networks, targets, physics

# Input nodes
species = inputs.SpeciesNode(db_name="Z")
positions = inputs.PositionsNode(db_name="R")
cell = inputs.CellNode(db_name="cell")

# Network
network = networks.Hipnn(
    "HIPNN",
    species,
    positions,
    cell=cell,
    module_kwargs={
        "n_features": 20,
        "n_sensitivities": 20,
        "dist_soft_min": 0.5,
        "dist_soft_max": 5.0,
        "dist_hard_max": 6.0,
        "n_interaction_layers": 3,
        "n_atom_layers": 3,
    }
)

# Energy prediction
henergy = targets.HEnergyNode("T", network, db_name="E")

# Force prediction (automatic via autograd)
hforce = physics.GradientNode(
    "F",
    henergy,
    positions,
    sign=-1,
    db_name="F"
)

# Charge prediction
hcharge = targets.HChargeNode("Q", network, db_name="Q")

# Assemble prediction graph
from hippynn.experiment import assemble_for_training

training_modules, db_info = assemble_for_training(
    henergy,
    hforce,
    hcharge,
)
```

### 8.3 Loss Function

```python
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau

loss = {
    henergy: 1.0,      # Energy weight
    hforce: 10.0,      # Force weight (10× energy)
    hcharge: 5.0,      # Charge weight (5× energy)
}

# L2 loss on all quantities
```

**Why these weights?**
- Forces have 3N components vs 1 energy → need higher weight
- Charges have N components → intermediate weight
- Typical ratios: E:F:Q = 1:10:5

### 8.4 Training Loop

```python
from hippynn.experiment import setup_and_train

database = hippynn.databases.DirectoryDatabase(
    name="IrO2_database",
    directory="./training_data",
    **db_info
)

setup_and_train(
    training_modules=training_modules,
    database=database,
    loss_weights=loss,
    optimizer=torch.optim.Adam,
    learning_rate=0.001,
    max_epochs=1000,
    batch_size=32,
    eval_batch_size=128,
    device='cuda',
    controller=RaiseBatchSizeOnPlateau(
        max_batch_size=256,
        patience=10,
        factor=1.5
    ),
    callbacks=[
        hippynn.experiment.serialization.Checkpointer(
            checkpoint_interval=10,
            best_model_file="best_model.pt"
        ),
    ]
)
```

### 8.5 Charge Prediction Enhancement

**Option 1: Predict χ and η instead of Q directly**

```python
# Electronegativity node
class ElectronegativityNode(hippynn.graphs.GraphModule):
    def __init__(self, name, parents, module_kwargs=None):
        module_kwargs = module_kwargs or {}
        module = ElectronegativityModule(**module_kwargs)
        super().__init__(name, parents, module)

    _input_names = ["features"]
    _output_names = ["chi"]

class ElectronegativityModule(torch.nn.Module):
    def __init__(self, n_features, n_output=1):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_output)

    def forward(self, features):
        # features: (n_atoms, n_features)
        chi = self.linear(features).squeeze(-1)
        return chi

# Use in graph
chi_node = ElectronegativityNode("chi", network)
eta_node = HardnessNode("eta", network)  # Similar implementation

# Then use chi, eta with QEq solver externally
```

**Option 2: Joint energy-charge prediction**

```python
# Local charge energy
from hippynn.layers.targets import LocalChargeEnergy

charge_energy = LocalChargeEnergy(
    "Echarge",
    (henergy, hcharge),
    quadratic_form="default"
)

# Total energy includes charge contribution
total_energy = henergy + charge_energy

# Train on total
training_modules, db_info = assemble_for_training(
    total_energy,
    hforce,
    hcharge,
)
```

---

## 9. Complete Workflow

### 9.1 End-to-End Pipeline

```
STEP 1: Generate Training Data
├── Use MLIP_Dataset_Generator
├── Generate ~20,000 structures
├── Coverage: dimers, trimers, tetramers, clusters, TS, templates
└── Output: structures.extxyz

STEP 2: Run DFT Calculations
├── VASP with r2SCAN+rVV10
├── Extract: E, F, Hirshfeld Q
├── Apply potential: GCE at different U
└── Output: dft_database/

STEP 3: Train HIPPYNN Model
├── Database: dft_database/
├── Architecture: Hipnn with HChargeNode
├── Loss: E + 10×F + 5×Q
├── Epochs: 1000
└── Output: model.pt

STEP 4: Fit QEq Parameters
├── From DFT charge scans
├── Fit: χᵢ, ηᵢ per element
└── Output: chi.txt

STEP 5: Prepare LAMMPS Input
├── Structure: interface.data
├── MLIP: model.pt
├── QEq params: chi.txt
└── Script: input.lammps

STEP 6: Run CP-MLMD
├── LAMMPS with fix electrode
├── Duration: 100-500 ps
├── Output: traj.lammpstrj, charges.dat
└── Analysis: EDL structure, capacitance

STEP 7: Validation
├── Compare C_diff vs experiments
├── Check EDL density profile
├── Verify charge conservation
└── Benchmark vs DFT-GCMD (if available)

STEP 8: Active Learning (Optional)
├── Identify high-uncertainty configs
├── Run DFT on those
├── Retrain HIPPYNN
└── Iterate steps 6-8
```

### 9.2 Validation Checklist

**Energy conservation:**
```
ΔE_total / E_total < 10⁻⁴  (NVE ensemble)
```

**Charge conservation:**
```
|Σᵢ qᵢ - Q_total| < 10⁻⁶ e
```

**Equipotential constraint:**
```
max|φᵢ - Φₐ| < 10⁻³ V  for i ∈ electrode α
```

**Capacitance physical range:**
```
C_diff ~ 10-50 μF/cm²  (typical for metal-electrolyte)
```

**EDL structure:**
```
ρ(z) shows:
- Helmholtz layer (z < 3 Å from electrode)
- Diffuse layer (3 Å < z < 10 Å)
- Bulk electrolyte (z > 10 Å)
```

**Charge response linearity:**
```
ΔQ/ΔΦ ≈ constant for |ΔΦ| < 0.5 V
```

---

## SUMMARY: Key Equations

### Constant Potential Method
```
E_total = E_short + E_Coulomb - Σₐ Φₐ Qₐ
Q = Q₀ + C·V
```

### QEq Charge Equilibration
```
χᵢ = χᵢ⁰ + 2ηᵢ qᵢ + Σⱼ Jᵢⱼ qⱼ = constant
```

### MLIP Charge Prediction
```
qᵢ = HChargeNode(rᵢ, neighbors)
OR
χᵢ = NN_χ(Gᵢ), then QEq solver
```

### Forces with Fluctuating Charges
```
Fᵢ = Fᵢ^direct + Fᵢ^induced
   = -∂E/∂rᵢ|_q - Σⱼ φⱼ (∂qⱼ/∂rᵢ)
```

### Capacitance
```
C_diff = ∂Q/∂Φ = lim[ΔΦ→0] ΔQ/ΔΦ
```

---

**Document Version:** 1.0
**Author:** ARIA Research Team
**Date:** December 2025
**References:** See main CP-MLMD research report
