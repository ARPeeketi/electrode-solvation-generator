#!/usr/bin/env python3
"""
================================================================================
Two-Exponential Chronoamperometry Fitting Script
================================================================================

Fits voltage-stepped chronoamperometry data with 1-exp and 2-exp models:

    1-exp: i(t) = A·exp(-t/τ) + B/√t + C
    2-exp: i(t) = A₁·exp(-t/τ₁) + A₂·exp(-t/τ₂) + B/√t + C

Features:
    - Robust voltage step detection with median smoothing
    - Automatic 1-exp vs 2-exp model selection via BIC
    - Comprehensive plotting and Bazant analysis
    - Supports European decimal format (comma)

Usage:
    python fit_transient_2exp.py data.csv
    python fit_transient_2exp.py data.txt falling
    python fit_transient_2exp.py data.csv both --sep=\\t --decimal=,

Arguments:
    file       : CSV/TXT file with time, voltage, current columns
    step_type  : 'rising', 'falling', or 'both' (default: 'falling')
    --sep      : Column separator (default: auto-detect)
    --decimal  : Decimal separator (default: '.')
    
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import sys
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

# Fitting parameters
T_SKIP = 0.002          # Skip first 2 ms (1/√t singularity)
T_FIT_MAX = None        # Max time to fit (None = use all)
STEP_DURATION = 4.0     # Max duration per step (seconds)

# Step detection parameters
MIN_STEP_SIZE = 0.05    # Minimum ΔV to count as step (V)
MIN_DURATION = 0.1      # Minimum time at a level (s)
SMOOTHING_WINDOW = 50   # Points for median smoothing

# Output
OUTPUT_PREFIX = 'transient_fits'


# =============================================================================
# MODELS
# =============================================================================

def model_1exp(t, A, tau, B, C):
    """Single exponential: i(t) = A·exp(-t/τ) + B/√t + C"""
    return A * np.exp(-t / tau) + B / np.sqrt(t) + C


def model_2exp(t, A1, tau1, A2, tau2, B, C):
    """Double exponential: i(t) = A₁·exp(-t/τ₁) + A₂·exp(-t/τ₂) + B/√t + C"""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + B / np.sqrt(t) + C


def calculate_bic(n, k, ss_res):
    """Calculate BIC for model comparison."""
    if ss_res <= 0:
        return np.inf
    return n * np.log(ss_res / n) + k * np.log(n)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath, sep=None, decimal='.', encoding='utf-8'):
    """
    Load CSV/TXT data file with automatic column detection.
    
    Parameters:
    -----------
    filepath : str - path to data file
    sep : str - column separator (None = auto-detect)
    decimal : str - decimal separator ('.' or ',')
    encoding : str - file encoding
    
    Returns:
    --------
    df : DataFrame with standardized column names
    t_col, v_col, i_col : column name strings
    """
    print(f"\nLoading: {filepath}")
    
    # Auto-detect separator
    if sep is None:
        with open(filepath, 'r', encoding=encoding) as f:
            first_line = f.readline()
            if '\t' in first_line:
                sep = '\t'
                print("  Detected separator: TAB")
            elif ';' in first_line:
                sep = ';'
                print("  Detected separator: semicolon")
            else:
                sep = ','
                print("  Detected separator: comma")
    
    # Load data
    try:
        df = pd.read_csv(filepath, sep=sep, decimal=decimal, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep=sep, decimal=decimal, encoding='latin-1')
        print("  Using latin-1 encoding")
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for units row (second row with units like 's', 'V', 'mA')
    first_row = df.iloc[0]
    is_units_row = False
    for val in first_row:
        if isinstance(val, str) and val.lower() in ['s', 'v', 'mv', 'ma', 'a', 'ua']:
            is_units_row = True
            break
    
    if is_units_row:
        print("  Detected units row, skipping...")
        df = df.iloc[1:].reset_index(drop=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    print(f"  After cleaning: {len(df)} rows")
    
    # Auto-detect column names
    cols_lower = [c.lower() for c in df.columns]
    
    t_col = None
    v_col = None
    i_col = None
    
    # Time column
    for i, c in enumerate(cols_lower):
        if 'time' in c or c == 't' or c == 't/s' or 'zeit' in c:
            t_col = df.columns[i]
            break
    
    # Voltage column
    for i, c in enumerate(cols_lower):
        if 'potential' in c or 'voltage' in c or 'ewe' in c or c == 'v' or c == 'e/v' or 'spannung' in c:
            v_col = df.columns[i]
            break
    
    # Current column
    for i, c in enumerate(cols_lower):
        if 'current' in c or c == 'i' or 'i/ma' in c or 'strom' in c or c == 'i/a':
            i_col = df.columns[i]
            break
    
    # Fallback to positional
    if t_col is None:
        t_col = df.columns[0]
    if v_col is None:
        v_col = df.columns[1]
    if i_col is None:
        i_col = df.columns[2]
    
    print(f"  Using columns: time='{t_col}', voltage='{v_col}', current='{i_col}'")
    
    # Basic stats
    print(f"\n  Data ranges:")
    print(f"    Time: {df[t_col].min():.4f} - {df[t_col].max():.4f} s")
    print(f"    Voltage: {df[v_col].min():.4f} - {df[v_col].max():.4f} V")
    print(f"    Current: {df[i_col].min():.4f} - {df[i_col].max():.4f} mA")
    
    return df, t_col, v_col, i_col


# =============================================================================
# VOLTAGE STEP DETECTION
# =============================================================================

def detect_voltage_steps(df, v_col, t_col, min_step_size=0.05, 
                         min_duration=0.1, smoothing_window=50):
    """
    Robustly detect voltage steps in noisy data.
    
    Algorithm:
    1. Smooth voltage with rolling median
    2. Round to voltage resolution to cluster similar values
    3. Identify distinct voltage levels with minimum duration
    4. Detect transitions between levels
    
    Returns:
    --------
    list of step dicts with V_before, V_after, dV, type, timing
    """
    voltage = df[v_col].values.copy()
    time = df[t_col].values.copy()
    
    # Smooth voltage with rolling median
    if smoothing_window > 1 and len(voltage) > smoothing_window:
        voltage_smooth = pd.Series(voltage).rolling(
            window=smoothing_window, center=True, min_periods=1
        ).median().values
    else:
        voltage_smooth = voltage
    
    # Round to cluster similar values
    voltage_resolution = min_step_size / 2
    voltage_rounded = np.round(voltage_smooth / voltage_resolution) * voltage_resolution
    
    # Identify distinct voltage levels
    levels = []
    current_level = voltage_rounded[0]
    level_start_idx = 0
    
    for i in range(1, len(voltage_rounded)):
        if abs(voltage_rounded[i] - current_level) > min_step_size / 2:
            # End of current level
            duration = time[i-1] - time[level_start_idx]
            if duration >= min_duration:
                levels.append({
                    'V_mean': np.mean(voltage[level_start_idx:i]),
                    'V_median': np.median(voltage[level_start_idx:i]),
                    'start_idx': level_start_idx,
                    'end_idx': i - 1,
                    'start_time': time[level_start_idx],
                    'end_time': time[i-1],
                    'duration': duration
                })
            current_level = voltage_rounded[i]
            level_start_idx = i
    
    # Last level
    duration = time[-1] - time[level_start_idx]
    if duration >= min_duration:
        levels.append({
            'V_mean': np.mean(voltage[level_start_idx:]),
            'V_median': np.median(voltage[level_start_idx:]),
            'start_idx': level_start_idx,
            'end_idx': len(voltage) - 1,
            'start_time': time[level_start_idx],
            'end_time': time[-1],
            'duration': duration
        })
    
    print(f"\n  Detected {len(levels)} voltage levels:")
    for i, lvl in enumerate(levels):
        print(f"    Level {i+1}: {lvl['V_mean']:.3f}V for {lvl['duration']:.2f}s "
              f"(t={lvl['start_time']:.2f}-{lvl['end_time']:.2f}s)")
    
    # Identify transitions between levels
    steps = []
    for i in range(1, len(levels)):
        prev_level = levels[i-1]
        curr_level = levels[i]
        dV = curr_level['V_mean'] - prev_level['V_mean']
        
        if abs(dV) >= min_step_size:
            steps.append({
                'step_num': len(steps) + 1,
                'V_before': prev_level['V_mean'],
                'V_after': curr_level['V_mean'],
                'dV': dV,
                'type': 'RISING' if dV > 0 else 'FALLING',
                't_start': curr_level['start_time'],
                't_end': curr_level['end_time'],
                'duration': curr_level['duration'],
                'start_idx': curr_level['start_idx'],
                'end_idx': curr_level['end_idx']
            })
    
    print(f"\n  Found {len(steps)} voltage transitions:")
    for step in steps:
        print(f"    Step {step['step_num']}: {step['V_before']:.3f}V → {step['V_after']:.3f}V "
              f"({step['type']}, ΔV={abs(step['dV']):.3f}V)")
    
    return steps


# =============================================================================
# SINGLE TRANSIENT FITTING
# =============================================================================

def fit_transient(t, i, t_skip=0.002, t_fit_max=None):
    """
    Fit a single transient with 1-exp and 2-exp models.
    
    Parameters:
    -----------
    t : array - time in seconds (starting from 0)
    i : array - current in mA
    t_skip : float - skip initial time to avoid 1/√t singularity
    t_fit_max : float - max time to fit (None = use all)
    
    Returns:
    --------
    dict with fit_1exp, fit_2exp, best_model, delta_BIC, t_fit, i_fit
    """
    t = np.asarray(t)
    i = np.asarray(i)
    
    # Filter data
    mask = t > t_skip
    if t_fit_max is not None:
        mask &= t <= t_fit_max
    
    t_fit = t[mask]
    i_fit = i[mask]
    
    if len(t_fit) < 20:
        print("    ERROR: Not enough data points")
        return None
    
    n = len(t_fit)
    ss_tot = np.sum((i_fit - np.mean(i_fit))**2)
    
    # Initial guesses
    A0 = i_fit[0] - i_fit[-1]
    B0 = 0.1 * np.sign(A0) if A0 != 0 else 0.1
    C0 = i_fit[-1]
    
    # =========================================================================
    # 1-EXPONENTIAL FIT
    # =========================================================================
    try:
        popt_1, pcov_1 = curve_fit(
            model_1exp, t_fit, i_fit,
            p0=[A0, 0.1, B0, C0],
            bounds=([-np.inf, 0.001, -10, -np.inf], [np.inf, 10, 10, np.inf]),
            maxfev=20000
        )
        perr_1 = np.sqrt(np.diag(pcov_1))
        
        i_pred_1 = model_1exp(t_fit, *popt_1)
        ss_res_1 = np.sum((i_fit - i_pred_1)**2)
        r2_1 = 1 - ss_res_1 / ss_tot
        bic_1 = calculate_bic(n, 4, ss_res_1)
        
        fit_1exp = {
            'A': popt_1[0], 'A_err': perr_1[0],
            'tau': popt_1[1], 'tau_err': perr_1[1],
            'B': popt_1[2], 'B_err': perr_1[2],
            'C': popt_1[3], 'C_err': perr_1[3],
            'R2': r2_1, 'BIC': bic_1,
            'ss_res': ss_res_1,
            'i_pred': i_pred_1
        }
    except Exception as e:
        print(f"    1-exp fit failed: {e}")
        return None
    
    # =========================================================================
    # 2-EXPONENTIAL FIT
    # =========================================================================
    fit_2exp = None
    try:
        popt_2, pcov_2 = curve_fit(
            model_2exp, t_fit, i_fit,
            p0=[A0*0.6, 0.03, A0*0.4, 0.15, fit_1exp['B'], fit_1exp['C']],
            bounds=([-np.inf, 0.001, -np.inf, 0.001, -10, -np.inf],
                   [np.inf, 10, np.inf, 10, 10, np.inf]),
            maxfev=30000
        )
        perr_2 = np.sqrt(np.diag(pcov_2))
        
        # Sort so tau1 < tau2 (fast < slow)
        A1, tau1, A2, tau2, B, C = popt_2
        A1_err, tau1_err, A2_err, tau2_err, B_err, C_err = perr_2
        
        if tau1 > tau2:
            A1, A2 = A2, A1
            tau1, tau2 = tau2, tau1
            A1_err, A2_err = A2_err, A1_err
            tau1_err, tau2_err = tau2_err, tau1_err
        
        i_pred_2 = model_2exp(t_fit, A1, tau1, A2, tau2, B, C)
        ss_res_2 = np.sum((i_fit - i_pred_2)**2)
        r2_2 = 1 - ss_res_2 / ss_tot
        bic_2 = calculate_bic(n, 6, ss_res_2)
        
        fit_2exp = {
            'A1': A1, 'A1_err': A1_err,
            'tau1': tau1, 'tau1_err': tau1_err,
            'A2': A2, 'A2_err': A2_err,
            'tau2': tau2, 'tau2_err': tau2_err,
            'B': B, 'B_err': B_err,
            'C': C, 'C_err': C_err,
            'R2': r2_2, 'BIC': bic_2,
            'ss_res': ss_res_2,
            'i_pred': i_pred_2
        }
    except Exception as e:
        print(f"    2-exp fit failed: {e}")
    
    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    if fit_2exp is not None:
        delta_bic = fit_2exp['BIC'] - fit_1exp['BIC']
        best_model = '2exp' if delta_bic < -6 else '1exp'
    else:
        delta_bic = None
        best_model = '1exp'
    
    return {
        'fit_1exp': fit_1exp,
        'fit_2exp': fit_2exp,
        'best_model': best_model,
        'delta_BIC': delta_bic,
        't_fit': t_fit,
        'i_fit': i_fit
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_all_steps(df, t_col, v_col, i_col, 
                      step_duration=4.0, t_skip=0.002, t_fit_max=None,
                      step_type_filter='falling', min_step_size=0.05):
    """
    Analyze all voltage steps in the data.
    
    Returns:
    --------
    results : list of result dicts
    """
    print("\n" + "="*70)
    print("VOLTAGE-STEPPED CHRONOAMPEROMETRY ANALYSIS")
    print("="*70)
    
    # Detect steps
    steps = detect_voltage_steps(df, v_col=v_col, t_col=t_col,
                                  min_step_size=min_step_size,
                                  min_duration=MIN_DURATION,
                                  smoothing_window=SMOOTHING_WINDOW)
    
    # Filter by type
    if step_type_filter.lower() == 'falling':
        steps = [s for s in steps if s['type'] == 'FALLING']
        print(f"\n  Filtering to FALLING only: {len(steps)} steps")
    elif step_type_filter.lower() == 'rising':
        steps = [s for s in steps if s['type'] == 'RISING']
        print(f"\n  Filtering to RISING only: {len(steps)} steps")
    else:
        print(f"\n  Using all steps: {len(steps)} steps")
    
    if len(steps) == 0:
        print("  No steps found!")
        return []
    
    results = []
    
    for step in steps:
        print(f"\n{'='*70}")
        print(f"STEP {step['step_num']}: {step['V_before']:.3f}V → {step['V_after']:.3f}V ({step['type']})")
        print(f"ΔV = {abs(step['dV']):.3f}V, Duration = {step['duration']:.2f}s")
        print(f"t = {step['t_start']:.3f}s to {step['t_end']:.3f}s")
        print("="*70)
        
        # Extract segment
        actual_duration = min(step['duration'], step_duration)
        t_end_fit = step['t_start'] + actual_duration
        
        mask = (df[t_col] >= step['t_start']) & (df[t_col] <= t_end_fit)
        df_seg = df[mask].copy()
        
        if len(df_seg) < 50:
            print("  Skipping: not enough points")
            continue
        
        # Reset time to start at 0
        t0 = df_seg[t_col].iloc[0]
        t = df_seg[t_col].values - t0
        current = df_seg[i_col].values
        
        # Fit
        fit_result = fit_transient(t, current, t_skip=t_skip, t_fit_max=t_fit_max)
        
        if fit_result is None:
            continue
        
        # Print results
        f1 = fit_result['fit_1exp']
        f2 = fit_result['fit_2exp']
        
        print(f"\n--- 1-EXPONENTIAL MODEL ---")
        print(f"  A   = {f1['A']:+.4f} ± {f1['A_err']:.4f} mA")
        print(f"  τ   = {f1['tau']*1000:.2f} ± {f1['tau_err']*1000:.2f} ms")
        print(f"  B   = {f1['B']:+.5f} ± {f1['B_err']:.5f} mA·s^0.5")
        print(f"  C   = {f1['C']:+.4f} ± {f1['C_err']:.4f} mA")
        print(f"  R²  = {f1['R2']:.6f}")
        print(f"  BIC = {f1['BIC']:.1f}")
        
        if f2 is not None:
            print(f"\n--- 2-EXPONENTIAL MODEL ---")
            print(f"  A₁  = {f2['A1']:+.4f} ± {f2['A1_err']:.4f} mA  (τ₁ = {f2['tau1']*1000:.2f} ms) [FAST]")
            print(f"  A₂  = {f2['A2']:+.4f} ± {f2['A2_err']:.4f} mA  (τ₂ = {f2['tau2']*1000:.2f} ms) [SLOW]")
            print(f"  B   = {f2['B']:+.5f} ± {f2['B_err']:.5f} mA·s^0.5")
            print(f"  C   = {f2['C']:+.4f} ± {f2['C_err']:.4f} mA")
            print(f"  R²  = {f2['R2']:.6f}")
            print(f"  BIC = {f2['BIC']:.1f}")
            print(f"\n  ΔBIC = {fit_result['delta_BIC']:.1f} (negative favors 2-exp)")
            print(f"  τ₂/τ₁ = {f2['tau2']/f2['tau1']:.1f}x separation")
        
        print(f"\n  >>> BEST MODEL: {fit_result['best_model'].upper()} <<<")
        
        # Store result
        result = {
            'step': step['step_num'],
            'type': step['type'],
            'V_initial': step['V_before'],
            'V_final': step['V_after'],
            'dV': abs(step['dV']),
            't_start': step['t_start'],
            't_end': step['t_end'],
            'fit_result': fit_result,
            't_data': t,
            'i_data': current
        }
        results.append(result)
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_all_fits(results, save_prefix='fits'):
    """Plot all fits with residuals."""
    n = len(results)
    if n == 0:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i, res in enumerate(results):
        fit = res['fit_result']
        t_ms = fit['t_fit'] * 1000
        i_data = fit['i_fit']
        f1 = fit['fit_1exp']
        f2 = fit['fit_2exp']
        
        # Main plot
        ax = axes[0, i]
        ax.plot(t_ms, i_data, 'k.', markersize=1, alpha=0.3, label='Data')
        ax.plot(t_ms, f1['i_pred'], 'b-', lw=2, label=f"1-exp R²={f1['R2']:.5f}")
        
        if f2 is not None:
            ax.plot(t_ms, f2['i_pred'], 'r--', lw=2, label=f"2-exp R²={f2['R2']:.5f}")
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (mA)')
        ax.set_title(f"Step {res['step']}: {res['V_initial']:.2f}→{res['V_final']:.2f}V\n"
                    f"Best: {fit['best_model'].upper()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(500, t_ms.max())])
        
        # Residuals
        ax = axes[1, i]
        res_1 = i_data - f1['i_pred']
        ax.plot(t_ms, res_1, 'b.', ms=1, alpha=0.5, label='1-exp')
        
        if f2 is not None:
            res_2 = i_data - f2['i_pred']
            ax.plot(t_ms, res_2, 'r.', ms=1, alpha=0.5, label='2-exp')
        
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Residual (mA)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(500, t_ms.max())])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_all_fits.png', dpi=150)
    plt.show()
    print(f"Saved: {save_prefix}_all_fits.png")


def plot_model_comparison(results, save_prefix='fits'):
    """Compare 1-exp vs 2-exp models across all steps."""
    # Filter to those with 2-exp fits
    results_2exp = [r for r in results if r['fit_result']['fit_2exp'] is not None]
    
    if len(results_2exp) < 1:
        print("No 2-exp fits to compare")
        return
    
    n = len(results_2exp)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Extract data
    steps = [r['step'] for r in results_2exp]
    r2_1exp = [r['fit_result']['fit_1exp']['R2'] for r in results_2exp]
    r2_2exp = [r['fit_result']['fit_2exp']['R2'] for r in results_2exp]
    delta_bic = [r['fit_result']['delta_BIC'] for r in results_2exp]
    tau_1exp = [r['fit_result']['fit_1exp']['tau']*1000 for r in results_2exp]
    tau1 = [r['fit_result']['fit_2exp']['tau1']*1000 for r in results_2exp]
    tau2 = [r['fit_result']['fit_2exp']['tau2']*1000 for r in results_2exp]
    B_1exp = [abs(r['fit_result']['fit_1exp']['B']) for r in results_2exp]
    B_2exp = [abs(r['fit_result']['fit_2exp']['B']) for r in results_2exp]
    
    x = np.arange(n)
    width = 0.35
    
    # R² comparison
    ax = axes[0, 0]
    ax.bar(x - width/2, r2_1exp, width, label='1-exp', color='blue', alpha=0.7)
    ax.bar(x + width/2, r2_2exp, width, label='2-exp', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel('R²')
    ax.set_title('Goodness of Fit')
    ax.legend()
    ax.set_ylim([min(0.99, min(r2_1exp + r2_2exp) - 0.005), 1.001])
    ax.grid(True, alpha=0.3)
    
    # ΔBIC
    ax = axes[0, 1]
    colors = ['green' if d < -6 else 'orange' if d < 0 else 'gray' for d in delta_bic]
    ax.bar(x, delta_bic, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(-6, color='green', ls='--', lw=2, label='Strong evidence for 2-exp')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel('ΔBIC (2exp - 1exp)')
    ax.set_title('Model Selection\n(negative = 2-exp better)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Time constants
    ax = axes[0, 2]
    ax.bar(x - width, tau_1exp, width, label='1-exp: τ', color='gray', alpha=0.7)
    ax.bar(x, tau1, width, label='2-exp: τ₁ (fast)', color='red', alpha=0.7)
    ax.bar(x + width, tau2, width, label='2-exp: τ₂ (slow)', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel('τ (ms)')
    ax.set_title('Time Constants')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # τ₂/τ₁ ratio
    ax = axes[1, 0]
    tau_ratio = np.array(tau2) / np.array(tau1)
    colors = ['red' if r['type'] == 'RISING' else 'blue' for r in results_2exp]
    ax.bar(x, tau_ratio, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(3, color='green', ls='--', label='3x separation')
    ax.axhline(5, color='orange', ls='--', label='5x separation')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel('τ₂/τ₁ ratio')
    ax.set_title('Time Scale Separation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # B comparison
    ax = axes[1, 1]
    ax.bar(x - width/2, B_1exp, width, label='1-exp: |B|', color='blue', alpha=0.7)
    ax.bar(x + width/2, B_2exp, width, label='2-exp: |B|', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel('|B| (mA·s^0.5)')
    ax.set_title('Diffusion Term')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Amplitudes
    ax = axes[1, 2]
    A_1exp = [abs(r['fit_result']['fit_1exp']['A']) for r in results_2exp]
    A1 = [abs(r['fit_result']['fit_2exp']['A1']) for r in results_2exp]
    A2 = [abs(r['fit_result']['fit_2exp']['A2']) for r in results_2exp]
    ax.bar(x - width, A_1exp, width, label='1-exp: |A|', color='gray', alpha=0.7)
    ax.bar(x, A1, width, label='2-exp: |A₁|', color='red', alpha=0.7)
    ax.bar(x + width, A2, width, label='2-exp: |A₂|', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_ylabel('|Amplitude| (mA)')
    ax.set_title('Amplitudes')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_model_comparison.png', dpi=150)
    plt.show()
    print(f"Saved: {save_prefix}_model_comparison.png")


def plot_parameters_vs_voltage(results, save_prefix='fits'):
    """Plot fit parameters vs voltage."""
    if len(results) < 2:
        print("Need at least 2 steps for parameter plots")
        return
    
    # Extract data
    data = []
    for res in results:
        f1 = res['fit_result']['fit_1exp']
        f2 = res['fit_result']['fit_2exp']
        
        # Use V_initial for falling, V_final for rising (for symmetry comparison)
        if res['type'] == 'FALLING':
            V_plot = res['V_initial']
        else:
            V_plot = res['V_final']
        
        row = {
            'V': V_plot,
            'type': res['type'],
            'dV': res['dV'],
            'A': f1['A'],
            'tau': f1['tau'] * 1000,
            'B': f1['B'],
            'C': f1['C'],
            'R2': f1['R2']
        }
        if f2 is not None:
            row['tau1'] = f2['tau1'] * 1000
            row['tau2'] = f2['tau2'] * 1000
            row['B_2exp'] = f2['B']
        data.append(row)
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    colors = ['red' if t == 'RISING' else 'blue' for t in df['type']]
    markers = ['^' if t == 'RISING' else 'v' for t in df['type']]
    
    # |A| vs V
    ax = axes[0, 0]
    for i, row in df.iterrows():
        ax.scatter(row['V'], abs(row['A']), c=colors[i], marker=markers[i], s=120, edgecolor='k')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('|A| (mA)')
    ax.set_title('Amplitude vs Voltage')
    ax.grid(True, alpha=0.3)
    
    # τ vs V
    ax = axes[0, 1]
    for i, row in df.iterrows():
        ax.scatter(row['V'], row['tau'], c=colors[i], marker=markers[i], s=120, edgecolor='k')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('τ (ms)')
    ax.set_title('Time Constant vs Voltage')
    ax.grid(True, alpha=0.3)
    
    # |B| vs V
    ax = axes[0, 2]
    for i, row in df.iterrows():
        ax.scatter(row['V'], abs(row['B']), c=colors[i], marker=markers[i], s=120, edgecolor='k')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('|B| (mA·s^0.5)')
    ax.set_title('Diffusion Term vs Voltage')
    ax.grid(True, alpha=0.3)
    
    # |B| vs ΔV (BAZANT TEST)
    ax = axes[1, 0]
    dV = df['dV'].values
    B_abs = np.abs(df['B'].values)
    
    for i, row in df.iterrows():
        ax.scatter(row['dV'], abs(row['B']), c=colors[i], marker=markers[i], s=120, edgecolor='k')
    
    if len(dV) >= 2:
        slope, intercept, r_val, p_val, _ = linregress(dV, B_abs)
        x_fit = np.linspace(0, dV.max()*1.1, 100)
        ax.plot(x_fit, slope*x_fit + intercept, 'k--', lw=2,
               label=f'|B| = {slope:.3f}·ΔV + {intercept:.4f}\nR² = {r_val**2:.4f}')
        ax.legend(fontsize=9)
    
    ax.set_xlabel('|ΔV| (V)')
    ax.set_ylabel('|B| (mA·s^0.5)')
    ax.set_title('B vs ΔV (Bazant Test)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    
    # |B|/ΔV consistency
    ax = axes[1, 1]
    B_over_dV = np.abs(df['B']) / df['dV']
    x_pos = np.arange(len(df))
    ax.bar(x_pos, B_over_dV, color=colors, edgecolor='k', alpha=0.7)
    ax.axhline(B_over_dV.mean(), color='green', ls='--', lw=2,
              label=f'Mean = {B_over_dV.mean():.3f} ± {B_over_dV.std():.3f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"S{i+1}" for i in x_pos])
    ax.set_ylabel('|B|/ΔV (mA·s^0.5/V)')
    ax.set_title('|B|/ΔV Consistency')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # C vs V (steady-state current)
    ax = axes[1, 2]
    for i, row in df.iterrows():
        ax.scatter(row['V'], row['C'], c=colors[i], marker=markers[i], s=120, edgecolor='k')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('C (mA)')
    ax.set_title('Steady-State Current vs Voltage')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=12, label='Rising'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue', markersize=12, label='Falling')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_parameters.png', dpi=150)
    plt.show()
    print(f"Saved: {save_prefix}_parameters.png")


def plot_summary_table(results, save_prefix='fits'):
    """Create and save summary table."""
    rows = []
    for res in results:
        f1 = res['fit_result']['fit_1exp']
        f2 = res['fit_result']['fit_2exp']
        
        row = {
            'Step': res['step'],
            'Type': res['type'],
            'V_i (V)': f"{res['V_initial']:.3f}",
            'V_f (V)': f"{res['V_final']:.3f}",
            'ΔV (V)': f"{res['dV']:.3f}",
            'A (mA)': f"{f1['A']:.3f}",
            'τ (ms)': f"{f1['tau']*1000:.2f}",
            'B': f"{f1['B']:.5f}",
            'C (mA)': f"{f1['C']:.3f}",
            'R²': f"{f1['R2']:.5f}",
            '|B|/ΔV': f"{abs(f1['B'])/res['dV']:.3f}"
        }
        
        if f2 is not None:
            row['τ₁ (ms)'] = f"{f2['tau1']*1000:.2f}"
            row['τ₂ (ms)'] = f"{f2['tau2']*1000:.2f}"
            row['R²_2exp'] = f"{f2['R2']:.5f}"
            row['ΔBIC'] = f"{res['fit_result']['delta_BIC']:.1f}"
            row['Best'] = res['fit_result']['best_model']
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Save to CSV
    summary_df.to_csv(f'{save_prefix}_results.csv', index=False)
    print(f"Saved: {save_prefix}_results.csv")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(3, len(rows)*0.5 + 1)))
    ax.axis('off')
    
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['lightblue']*len(summary_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color rows by type
    for i in range(len(rows)):
        color = '#ffcccc' if rows[i]['Type'] == 'RISING' else '#ccccff'
        for j in range(len(summary_df.columns)):
            table[(i+1, j)].set_facecolor(color)
    
    ax.set_title('Fit Results Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_table.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_prefix}_table.png")
    
    return summary_df


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Main entry point."""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo input file specified. Run with: python fit_transient_2exp.py data.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Step type filter
    step_type_filter = 'falling'  # default
    if len(sys.argv) >= 3:
        if sys.argv[2].lower() in ['rising', 'falling', 'both']:
            step_type_filter = sys.argv[2].lower()
    
    # Parse optional arguments
    sep = None
    decimal = '.'
    
    for arg in sys.argv[3:]:
        if arg.startswith('--sep='):
            sep = arg.split('=')[1]
            if sep == '\\t':
                sep = '\t'
        elif arg.startswith('--decimal='):
            decimal = arg.split('=')[1]
    
    print(f"\n{'='*70}")
    print(f"Two-Exponential Chronoamperometry Fitting")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Step filter: {step_type_filter}")
    print(f"Decimal separator: '{decimal}'")
    
    # Load data
    df, t_col, v_col, i_col = load_data(filepath, sep=sep, decimal=decimal)
    
    # Analyze
    results = analyze_all_steps(
        df, t_col, v_col, i_col,
        step_duration=STEP_DURATION,
        t_skip=T_SKIP,
        t_fit_max=T_FIT_MAX,
        step_type_filter=step_type_filter,
        min_step_size=MIN_STEP_SIZE
    )
    
    if len(results) == 0:
        print("\nNo steps could be fitted!")
        sys.exit(1)
    
    # Generate output prefix from filename
    output_prefix = os.path.splitext(os.path.basename(filepath))[0] + '_fits'
    
    # Plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    plot_all_fits(results, save_prefix=output_prefix)
    plot_model_comparison(results, save_prefix=output_prefix)
    plot_parameters_vs_voltage(results, save_prefix=output_prefix)
    summary_df = plot_summary_table(results, save_prefix=output_prefix)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    n_1exp = sum(1 for r in results if r['fit_result']['best_model'] == '1exp')
    n_2exp = sum(1 for r in results if r['fit_result']['best_model'] == '2exp')
    print(f"\nModel selection:")
    print(f"  1-exp preferred: {n_1exp} steps")
    print(f"  2-exp preferred: {n_2exp} steps")
    
    rising = [r for r in results if r['type'] == 'RISING']
    falling = [r for r in results if r['type'] == 'FALLING']
    
    if len(rising) > 0:
        tau_rise = [r['fit_result']['fit_1exp']['tau']*1000 for r in rising]
        print(f"\nRISING steps ({len(rising)}):")
        print(f"  τ mean: {np.mean(tau_rise):.1f} ± {np.std(tau_rise):.1f} ms")
    
    if len(falling) > 0:
        tau_fall = [r['fit_result']['fit_1exp']['tau']*1000 for r in falling]
        print(f"\nFALLING steps ({len(falling)}):")
        print(f"  τ mean: {np.mean(tau_fall):.1f} ± {np.std(tau_fall):.1f} ms")
    
    # Bazant test
    if len(results) >= 2:
        dV = np.array([r['dV'] for r in results])
        B_abs = np.array([abs(r['fit_result']['fit_1exp']['B']) for r in results])
        slope, intercept, r_val, p_val, _ = linregress(dV, B_abs)
        
        print(f"\nBAZANT TEST (B vs ΔV):")
        print(f"  |B| = {slope:.4f}·ΔV + {intercept:.4f}")
        print(f"  R² = {r_val**2:.4f}")
        print(f"  p-value = {p_val:.2e}")
        if r_val**2 > 0.9:
            print(f"  ✓ Strong linear relationship - supports Bazant bulk diffusion")
        elif r_val**2 > 0.7:
            print(f"  ~ Moderate linearity")
        else:
            print(f"  ⚠ Weak linearity - may indicate other effects")
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    results = main()
