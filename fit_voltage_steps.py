"""
Voltage-Stepped Chronoamperometry Fitting Script (2-Exponential Model)
=======================================================================

Fits transients with both 1-exp and 2-exp models:
    1-exp: i(t) = A·exp(-t/τ) + B/√t + C
    2-exp: i(t) = A₁·exp(-t/τ₁) + A₂·exp(-t/τ₂) + B/√t + C

Uses BIC for model selection.

Usage:
    python fit_transient_2exp.py data.csv
    python fit_transient_2exp.py data.xlsx SheetName
    python fit_transient_2exp.py data.xlsx SheetName falling
    
Arguments:
    file      : CSV or Excel file with electrochemistry data
    sheet     : (optional) Sheet name for Excel files
    step_type : (optional) 'rising', 'falling', or 'both' (default: 'falling')
    
Expected columns: time, voltage, current (auto-detected)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import sys

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
# STEP DETECTION
# =============================================================================

def detect_voltage_steps(df, v_col='Ewe/V', t_col='time/s', 
                         min_step_size=0.05, min_duration=0.1,
                         smoothing_window=50):
    """
    Robustly detect voltage steps in noisy data.
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
    
    print(f"  Detected {len(levels)} voltage levels:")
    for i, lvl in enumerate(levels):
        print(f"    Level {i+1}: {lvl['V_mean']:.3f}V for {lvl['duration']:.2f}s "
              f"(t={lvl['start_time']:.2f}-{lvl['end_time']:.2f}s)")
    
    # Identify transitions
    steps = []
    for i in range(1, len(levels)):
        prev_level = levels[i-1]
        curr_level = levels[i]
        dV = curr_level['V_mean'] - prev_level['V_mean']
        
        if abs(dV) >= min_step_size:
            steps.append({
                'idx': curr_level['start_idx'],
                'V_before': prev_level['V_mean'],
                'V_after': curr_level['V_mean'],
                'dV': dV,
                'type': 'RISING' if dV > 0 else 'FALLING',
                't_start': curr_level['start_time'],
                't_end': curr_level['end_time'],
                'duration': curr_level['duration']
            })
    
    return steps


# =============================================================================
# FITTING
# =============================================================================

def fit_single_step(t, i, step_type='FALLING', t_fit_max=None):
    """
    Fit a single transient with 1-exp and 2-exp models.
    
    Returns dict with both fits and model comparison.
    """
    # Remove t <= 0
    mask = t > 0.0005
    if t_fit_max is not None:
        mask &= t <= t_fit_max
    
    t_fit = t[mask]
    i_fit = i[mask]
    
    if len(t_fit) < 20:
        return None
    
    n_points = len(t_fit)
    ss_tot = np.sum((i_fit - np.mean(i_fit))**2)
    
    # =========================================================================
    # 1-EXPONENTIAL FIT
    # =========================================================================
    A0 = i_fit[0] - i_fit[-1]
    tau0 = 0.1
    B0 = 0.1 * np.sign(A0) if A0 != 0 else 0.1
    C0 = i_fit[-1]
    
    try:
        popt_1, pcov_1 = curve_fit(
            model_1exp, t_fit, i_fit,
            p0=[A0, tau0, B0, C0],
            bounds=([-np.inf, 0.001, -10, -np.inf], [np.inf, 10, 10, np.inf]),
            maxfev=20000
        )
        perr_1 = np.sqrt(np.diag(pcov_1))
        
        i_pred_1 = model_1exp(t_fit, *popt_1)
        ss_res_1 = np.sum((i_fit - i_pred_1)**2)
        r2_1 = 1 - ss_res_1 / ss_tot
        bic_1 = calculate_bic(n_points, 4, ss_res_1)
        
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
    
    # =========================================================================
    # 2-EXPONENTIAL FIT
    # =========================================================================
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
        bic_2 = calculate_bic(n_points, 6, ss_res_2)
        
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
        fit_2exp = None
    
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
# ANALYSIS
# =============================================================================

def analyze_voltage_steps(df, t_col='time/s', v_col='Ewe/V', i_col='I/mA',
                          step_duration=4.0, t_skip=0.001, t_fit_max=None,
                          step_type_filter='both', min_step_size=0.05):
    """Analyze all voltage steps in the data."""
    
    steps = detect_voltage_steps(df, v_col=v_col, t_col=t_col, 
                                  min_step_size=min_step_size)
    print(f"Found {len(steps)} voltage steps")
    
    # Filter by step type
    if step_type_filter.lower() == 'falling':
        steps = [s for s in steps if s['type'] == 'FALLING']
        print(f"  Filtering to FALLING only: {len(steps)} steps")
    elif step_type_filter.lower() == 'rising':
        steps = [s for s in steps if s['type'] == 'RISING']
        print(f"  Filtering to RISING only: {len(steps)} steps")
    
    results = []
    
    for i, step in enumerate(steps):
        V_before = step['V_before']
        V_after = step['V_after']
        step_type = step['type']
        t_start = step['t_start']
        t_end = step['t_end']
        
        print(f"\nStep {i+1}: {V_before:.3f} → {V_after:.3f} V ({step_type})")
        print(f"         t = {t_start:.3f}s to {t_end:.3f}s ({step['duration']:.2f}s)")
        
        # Extract segment
        actual_duration = min(step['duration'], step_duration)
        t_end_fit = t_start + actual_duration
        
        mask = (df[t_col] >= t_start) & (df[t_col] <= t_end_fit)
        df_seg = df[mask].copy()
        
        t0 = df_seg[t_col].iloc[0]
        t = df_seg[t_col].values - t0
        current = df_seg[i_col].values
        
        # Skip initial points
        mask = t >= t_skip
