#!/usr/bin/env python3
"""
================================================================================
Two-Exponential Chronoamperometry Fitting Script
================================================================================

Fits voltage-stepped chronoamperometry data with 1-exp and 2-exp models:

    1-exp: i(t) = A·exp(-t/τ) + B/√t + C
    2-exp: i(t) = A₁·exp(-t/τ₁) + A₂·exp(-t/τ₂) + B/√t + C

Outputs comprehensive summary sheet with all coefficients for Bazant and PCET/CER analysis.
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
# DATA LOADING (User's version)
# =============================================================================

def load_data(filepath, sep=None, decimal='.', encoding='utf-8', CBG=0.0):
    """
    Load CSV/TXT data file with automatic column detection.
    """
    print(f"\nLoading: {filepath}")
    
    # Auto-detect separator if not provided
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

    # Apply corrections (RHE conversion and current inversion)
    df['Ewe/V'] = df['Ewe/V'].values + 0.721    
    df['I/mA'] = -df['I/mA'].values - CBG
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for units row
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
    
    t_col = v_col = i_col = None
    
    for i, c in enumerate(cols_lower):
        if 'time' in c or c == 't' or c == 't/s' or 'zeit' in c:
            t_col = df.columns[i]
            break
    
    for i, c in enumerate(cols_lower):
        if 'potential' in c or 'voltage' in c or 'ewe' in c or c == 'v' or c == 'e/v':
            v_col = df.columns[i]
            break
    
    for i, c in enumerate(cols_lower):
        if 'current' in c or c == 'i' or 'i/ma' in c or c == 'i/a':
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
    """Robustly detect voltage steps in noisy data."""
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
            duration = time[i-1] - time[level_start_idx]
            if duration >= min_duration:
                levels.append({
                    'V_mean': np.mean(voltage[level_start_idx:i]),
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
            'start_idx': level_start_idx,
            'end_idx': len(voltage) - 1,
            'start_time': time[level_start_idx],
            'end_time': time[-1],
            'duration': duration
        })
    
    print(f"\n  Detected {len(levels)} voltage levels:")
    for i, lvl in enumerate(levels):
        print(f"    Level {i+1}: {lvl['V_mean']:.3f}V for {lvl['duration']:.2f}s")
    
    # Identify transitions
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
                'duration': curr_level['duration']
            })
    
    print(f"\n  Found {len(steps)} voltage transitions")
    return steps


# =============================================================================
# SINGLE TRANSIENT FITTING
# =============================================================================

def fit_transient(t, i, t_skip=0.002, t_fit_max=None):
    """Fit a single transient with 1-exp and 2-exp models."""
    t = np.asarray(t)
    i = np.asarray(i)
    
    # Filter data
    mask = t > t_skip
    if t_fit_max is not None:
        mask &= t <= t_fit_max
    
    t_fit = t[mask]
    i_fit = i[mask]
    
    if len(t_fit) < 20:
        return None
    
    n = len(t_fit)
    ss_tot = np.sum((i_fit - np.mean(i_fit))**2)
    
    # Initial guesses
    A0 = i_fit[0] - i_fit[-1]
    B0 = 0.1 * np.sign(A0) if A0 != 0 else 0.1
    C0 = i_fit[-1]
    
    # 1-EXPONENTIAL FIT
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
            'i_pred': i_pred_1
        }
    except Exception as e:
        print(f"    1-exp fit failed: {e}")
        return None
    
    # 2-EXPONENTIAL FIT
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
        
        # Sort so tau1 < tau2
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
            'i_pred': i_pred_2
        }
    except Exception as e:
        print(f"    2-exp fit failed: {e}")
    
    # Model selection
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
    """Analyze all voltage steps in the data."""
    
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
    
    if len(steps) == 0:
        print("  No steps found!")
        return []
    
    results = []
    
    for step in steps:
        print(f"\nStep {step['step_num']}: {step['V_before']:.3f}V → {step['V_after']:.3f}V ({step['type']})")
        
        # Extract segment
        actual_duration = min(step['duration'], step_duration)
        t_end_fit = step['t_start'] + actual_duration
        
        mask = (df[t_col] >= step['t_start']) & (df[t_col] <= t_end_fit)
        df_seg = df[mask].copy()
        
        if len(df_seg) < 50:
            print("  Skipping: not enough points")
            continue
        
        t0 = df_seg[t_col].iloc[0]
        t = df_seg[t_col].values - t0
        current = df_seg[i_col].values
        
        # Fit
        fit_result = fit_transient(t, current, t_skip=t_skip, t_fit_max=t_fit_max)
        
        if fit_result is None:
            continue
        
        # Store result
        result = {
            'step': step['step_num'],
            'type': step['type'],
            'V_initial': step['V_before'],
            'V_final': step['V_after'],
            'dV': abs(step['dV']),
            't_start': step['t_start'],
            'fit_result': fit_result,
            't_data': t,
            'i_data': current
        }
        results.append(result)
        
        print(f"  1-exp: R²={fit_result['fit_1exp']['R2']:.5f}, τ={fit_result['fit_1exp']['tau']*1000:.1f}ms")
        if fit_result['fit_2exp']:
            print(f"  2-exp: R²={fit_result['fit_2exp']['R2']:.5f}, τ₁={fit_result['fit_2exp']['tau1']*1000:.1f}ms, τ₂={fit_result['fit_2exp']['tau2']*1000:.1f}ms")
            print(f"  Best: {fit_result['best_model'].upper()}")
    
    return results


# =============================================================================
# COMPREHENSIVE SUMMARY OUTPUT
# =============================================================================

def create_comprehensive_summary(results, output_prefix='analysis'):
    """
    Create comprehensive summary DataFrame with all coefficients for both models.
    Includes derived quantities for Bazant and PCET/CER analysis.
    """
    
    rows = []
    
    for res in results:
        f1 = res['fit_result']['fit_1exp']
        f2 = res['fit_result']['fit_2exp']
        dV = res['dV']
        
        # Calculate derived quantities for 1-exp
        Q_exp_1 = abs(f1['A']) * f1['tau'] * 1000  # mC (charge from exp term)
        B_over_dV_1 = abs(f1['B']) / dV if dV > 0 else np.nan
        
        row = {
            # Step info
            'Step': res['step'],
            'Type': res['type'],
            'V_initial_RHE': res['V_initial'],
            'V_final_RHE': res['V_final'],
            'dV': dV,
            
            # 1-EXP MODEL PARAMETERS
            'A_1exp (mA)': f1['A'],
            'A_1exp_err': f1['A_err'],
            'tau_1exp (ms)': f1['tau'] * 1000,
            'tau_1exp_err (ms)': f1['tau_err'] * 1000,
            'B_1exp (mA·s^0.5)': f1['B'],
            'B_1exp_err': f1['B_err'],
            'C_1exp (mA)': f1['C'],
            'C_1exp_err': f1['C_err'],
            'R2_1exp': f1['R2'],
            'BIC_1exp': f1['BIC'],
            
            # 1-EXP DERIVED QUANTITIES
            '|A|_1exp': abs(f1['A']),
            '|B|_1exp': abs(f1['B']),
            '|B|/dV_1exp': B_over_dV_1,
            'Q_exp_1exp (mC)': Q_exp_1,
        }
        
        # 2-EXP MODEL PARAMETERS
        if f2 is not None:
            Q_fast = abs(f2['A1']) * f2['tau1'] * 1000  # mC
            Q_slow = abs(f2['A2']) * f2['tau2'] * 1000  # mC
            Q_total_2exp = Q_fast + Q_slow
            B_over_dV_2 = abs(f2['B']) / dV if dV > 0 else np.nan
            tau_ratio = f2['tau2'] / f2['tau1']
            A_ratio = abs(f2['A1']) / abs(f2['A2']) if abs(f2['A2']) > 0 else np.nan
            
            row.update({
                # Fast component (τ₁)
                'A1_2exp (mA)': f2['A1'],
                'A1_2exp_err': f2['A1_err'],
                'tau1_2exp (ms)': f2['tau1'] * 1000,
                'tau1_2exp_err (ms)': f2['tau1_err'] * 1000,
                
                # Slow component (τ₂)
                'A2_2exp (mA)': f2['A2'],
                'A2_2exp_err': f2['A2_err'],
                'tau2_2exp (ms)': f2['tau2'] * 1000,
                'tau2_2exp_err (ms)': f2['tau2_err'] * 1000,
                
                # Diffusion and steady-state
                'B_2exp (mA·s^0.5)': f2['B'],
                'B_2exp_err': f2['B_err'],
                'C_2exp (mA)': f2['C'],
                'C_2exp_err': f2['C_err'],
                
                # Quality metrics
                'R2_2exp': f2['R2'],
                'BIC_2exp': f2['BIC'],
                'delta_BIC': res['fit_result']['delta_BIC'],
                'Best_Model': res['fit_result']['best_model'],
                
                # 2-EXP DERIVED QUANTITIES
                '|A1|_2exp': abs(f2['A1']),
                '|A2|_2exp': abs(f2['A2']),
                '|B|_2exp': abs(f2['B']),
                '|B|/dV_2exp': B_over_dV_2,
                'tau2/tau1': tau_ratio,
                '|A1|/|A2|': A_ratio,
                'Q_fast (mC)': Q_fast,
                'Q_slow (mC)': Q_slow,
                'Q_total_2exp (mC)': Q_total_2exp,
                'Q_fast/Q_total': Q_fast / Q_total_2exp if Q_total_2exp > 0 else np.nan,
            })
        else:
            # Fill with NaN if 2-exp not available
            row.update({
                'A1_2exp (mA)': np.nan, 'A1_2exp_err': np.nan,
                'tau1_2exp (ms)': np.nan, 'tau1_2exp_err (ms)': np.nan,
                'A2_2exp (mA)': np.nan, 'A2_2exp_err': np.nan,
                'tau2_2exp (ms)': np.nan, 'tau2_2exp_err (ms)': np.nan,
                'B_2exp (mA·s^0.5)': np.nan, 'B_2exp_err': np.nan,
                'C_2exp (mA)': np.nan, 'C_2exp_err': np.nan,
                'R2_2exp': np.nan, 'BIC_2exp': np.nan,
                'delta_BIC': np.nan, 'Best_Model': '1exp',
                '|A1|_2exp': np.nan, '|A2|_2exp': np.nan,
                '|B|_2exp': np.nan, '|B|/dV_2exp': np.nan,
                'tau2/tau1': np.nan, '|A1|/|A2|': np.nan,
                'Q_fast (mC)': np.nan, 'Q_slow (mC)': np.nan,
                'Q_total_2exp (mC)': np.nan, 'Q_fast/Q_total': np.nan,
            })
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = f'{output_prefix}_comprehensive_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # Also save to Excel with formatting
    try:
        excel_path = f'{output_prefix}_comprehensive_summary.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='All_Parameters', index=False)
            
            # Create a condensed sheet for quick reference
            condensed_cols = [
                'Step', 'Type', 'V_initial_RHE', 'V_final_RHE', 'dV',
                'tau_1exp (ms)', 'R2_1exp', '|B|/dV_1exp',
                'tau1_2exp (ms)', 'tau2_2exp (ms)', 'tau2/tau1',
                'R2_2exp', '|B|/dV_2exp', 'delta_BIC', 'Best_Model'
            ]
            condensed_df = summary_df[[c for c in condensed_cols if c in summary_df.columns]]
            condensed_df.to_excel(writer, sheet_name='Quick_Reference', index=False)
        
        print(f"Saved: {excel_path}")
    except Exception as e:
        print(f"Excel save failed (openpyxl not available): {e}")
    
    return summary_df


def print_bazant_analysis(results):
    """Print Bazant analysis summary."""
    
    print("\n" + "="*70)
    print("BAZANT ANALYSIS (B vs ΔV)")
    print("="*70)
    
    if len(results) < 2:
        print("Need at least 2 steps for Bazant analysis")
        return
    
    # 1-exp B values
    dV = np.array([r['dV'] for r in results])
    B_1exp = np.array([abs(r['fit_result']['fit_1exp']['B']) for r in results])
    
    slope_1, intercept_1, r_val_1, p_val_1, _ = linregress(dV, B_1exp)
    
    print("\n1-EXP MODEL:")
    print(f"  |B| = {slope_1:.4f}·ΔV + {intercept_1:.5f}")
    print(f"  R² = {r_val_1**2:.4f}")
    print(f"  p-value = {p_val_1:.2e}")
    print(f"  Mean |B|/ΔV = {np.mean(B_1exp/dV):.4f} ± {np.std(B_1exp/dV):.4f}")
    
    # 2-exp B values
    results_2exp = [r for r in results if r['fit_result']['fit_2exp'] is not None]
    if len(results_2exp) >= 2:
        dV_2 = np.array([r['dV'] for r in results_2exp])
        B_2exp = np.array([abs(r['fit_result']['fit_2exp']['B']) for r in results_2exp])
        
        slope_2, intercept_2, r_val_2, p_val_2, _ = linregress(dV_2, B_2exp)
        
        print("\n2-EXP MODEL:")
        print(f"  |B| = {slope_2:.4f}·ΔV + {intercept_2:.5f}")
        print(f"  R² = {r_val_2**2:.4f}")
        print(f"  p-value = {p_val_2:.2e}")
        print(f"  Mean |B|/ΔV = {np.mean(B_2exp/dV_2):.4f} ± {np.std(B_2exp/dV_2):.4f}")


def print_pcet_cer_analysis(results):
    """Print PCET/CER analysis summary."""
    
    print("\n" + "="*70)
    print("PCET/CER ANALYSIS (Time Constants)")
    print("="*70)
    
    results_2exp = [r for r in results if r['fit_result']['fit_2exp'] is not None]
    
    if len(results_2exp) == 0:
        print("No 2-exp fits available")
        return
    
    print("\n2-EXP MODEL TIME CONSTANTS:")
    print("-"*60)
    print(f"{'Step':<6} {'Type':<8} {'V_i':<6} {'τ₁(ms)':<10} {'τ₂(ms)':<10} {'τ₂/τ₁':<8} {'|A₁|/|A₂|':<10}")
    print("-"*60)
    
    tau1_all = []
    tau2_all = []
    ratio_all = []
    
    for res in results_2exp:
        f2 = res['fit_result']['fit_2exp']
        tau1 = f2['tau1'] * 1000
        tau2 = f2['tau2'] * 1000
        ratio = tau2 / tau1
        A_ratio = abs(f2['A1']) / abs(f2['A2']) if abs(f2['A2']) > 0 else np.nan
        
        tau1_all.append(tau1)
        tau2_all.append(tau2)
        ratio_all.append(ratio)
        
        print(f"{res['step']:<6} {res['type']:<8} {res['V_initial']:<6.2f} {tau1:<10.2f} {tau2:<10.2f} {ratio:<8.1f} {A_ratio:<10.2f}")
    
    print("-"*60)
    print(f"{'Mean':<6} {'':<8} {'':<6} {np.mean(tau1_all):<10.2f} {np.mean(tau2_all):<10.2f} {np.mean(ratio_all):<8.1f}")
    print(f"{'Std':<6} {'':<8} {'':<6} {np.std(tau1_all):<10.2f} {np.std(tau2_all):<10.2f} {np.std(ratio_all):<8.1f}")
    
    print("\nINTERPRETATION:")
    mean_ratio = np.mean(ratio_all)
    if mean_ratio > 3:
        print(f"  τ₂/τ₁ = {mean_ratio:.1f}x → Good time-scale separation")
        print("  τ₁ (fast): Surface PCET / CER kinetics")
        print("  τ₂ (slow): Bulk oxide proton diffusion / slow PCET")
    else:
        print(f"  τ₂/τ₁ = {mean_ratio:.1f}x → Limited separation, may be single process")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Main entry point."""
    
    step_type_filter = 'falling'  # default
    
    # USER CONFIGURATION - MODIFY HERE
    filepath = 'CER_1500mM_20241120_28_pulse_590mV_N_S1_CER_02_CA_C02.txt'
    CBGa = -0.0117
    
    # Load data
    df, t_col, v_col, i_col = load_data(filepath, sep=r'\s+', decimal=',', CBG=CBGa)
    
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
    output_prefix = os.path.splitext(os.path.basename(filepath))[0]
    
    # Create comprehensive summary
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE SUMMARY")
    print("="*70)
    
    summary_df = create_comprehensive_summary(results, output_prefix=output_prefix)
    
    # Print analysis summaries
    print_bazant_analysis(results)
    print_pcet_cer_analysis(results)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    n_1exp = sum(1 for r in results if r['fit_result']['best_model'] == '1exp')
    n_2exp = sum(1 for r in results if r['fit_result']['best_model'] == '2exp')
    
    print(f"\nTotal steps analyzed: {len(results)}")
    print(f"  1-exp preferred: {n_1exp}")
    print(f"  2-exp preferred: {n_2exp}")
    
    print(f"\nOutput files:")
    print(f"  {output_prefix}_comprehensive_summary.csv")
    print(f"  {output_prefix}_comprehensive_summary.xlsx")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results, summary_df


if __name__ == '__main__':
    results, summary_df = main()
