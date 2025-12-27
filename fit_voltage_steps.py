"""
Voltage-Stepped Chronoamperometry Fitting Script
=================================================

Automatically:
1. Detects voltage step transitions
2. Fits each transient with: i(t) = A*exp(-t/τ) + B/√t + C
3. Separates RISING vs FALLING steps
4. Plots parameters vs voltage (optionally iR-corrected)

Usage:
    python fit_voltage_steps.py your_data.csv
    python fit_voltage_steps.py your_data.xlsx SheetName
    python fit_voltage_steps.py your_data.xlsx SheetName falling
    python fit_voltage_steps.py your_data.xlsx SheetName falling iR_lookup.csv
    
Arguments:
    file        : CSV or Excel file with electrochemistry data
    sheet       : (optional) Sheet name for Excel files
    step_type   : (optional) 'rising', 'falling', or 'both' (default: 'falling')
    iR_lookup   : (optional) CSV/Excel with voltage → iR-corrected voltage mapping
    
Expected columns: 'time/s', 'Ewe/V', 'I/mA' (or similar, auto-detected)

Note: Rising steps often have poor fits due to additional physics 
      (OER, bubble formation). Default is to fit only falling steps.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys

# =============================================================================
# iR CORRECTION LOOKUP
# =============================================================================

def load_iR_correction_table(filename, sheet_name=0):
    """
    Load voltage to iR-corrected voltage lookup table.
    
    Expected format: 2 columns
      Column 1: Voltage (V)
      Column 2: iR-corrected voltage (V)
    
    Returns:
    --------
    interpolator function: V -> V_iR_corrected
    """
    print(f"\nLoading iR correction table: {filename}")
    
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    else:
        df = pd.read_csv(filename, header=None)
    
    # Check if first row is header
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        print(f"  Skipping header row: {df.iloc[0].tolist()}")
        df = df.iloc[1:].reset_index(drop=True)
    
    # Convert to numeric
    df[0] = pd.to_numeric(df[0], errors='coerce')
    df[1] = pd.to_numeric(df[1], errors='coerce')
    df = df.dropna()
    
    V_raw = df.iloc[:, 0].values
    V_corrected = df.iloc[:, 1].values
    
    print(f"  Loaded {len(V_raw)} voltage points")
    print(f"  V range: {V_raw.min():.3f} - {V_raw.max():.3f} V")
    print(f"  V_iR range: {V_corrected.min():.3f} - {V_corrected.max():.3f} V")
    
    # Create interpolator (with extrapolation for edge cases)
    interpolator = interp1d(V_raw, V_corrected, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    return interpolator


def apply_iR_correction(voltage, iR_interpolator):
    """Apply iR correction to a voltage value or array."""
    if iR_interpolator is None:
        return voltage
    return float(iR_interpolator(voltage))

# =============================================================================
# MODELS
# =============================================================================

def transient_model_1exp(t, A, tau, B, C):
    """Single exponential: i(t) = A*exp(-t/τ) + B/√t + C"""
    return A * np.exp(-t / tau) + B / np.sqrt(t) + C


def transient_model_2exp(t, A1, tau1, A2, tau2, B, C):
    """Double exponential: i(t) = A1*exp(-t/τ1) + A2*exp(-t/τ2) + B/√t + C"""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + B / np.sqrt(t) + C


def calculate_aic_bic(n, k, ss_res):
    """
    Calculate AIC and BIC for model comparison.
    
    n: number of data points
    k: number of parameters
    ss_res: sum of squared residuals
    """
    # Avoid log(0)
    if ss_res <= 0:
        return np.inf, np.inf
    
    # Log-likelihood (assuming Gaussian errors)
    log_likelihood = -n/2 * np.log(2*np.pi*ss_res/n) - n/2
    
    # AIC = 2k - 2*log(L)
    aic = 2*k - 2*log_likelihood
    
    # BIC = k*log(n) - 2*log(L)
    bic = k*np.log(n) - 2*log_likelihood
    
    return aic, bic


# =============================================================================
# STEP DETECTION (ROBUST)
# =============================================================================

def detect_voltage_steps(df, v_col='Ewe/V', t_col='time/s', 
                         min_step_size=0.05, min_duration=0.1,
                         smoothing_window=50):
    """
    Robustly detect voltage steps in noisy data.
    
    Uses:
    1. Median smoothing to reduce noise
    2. Clustering to identify distinct voltage levels
    3. Duration filtering to ignore transient spikes
    
    Parameters:
    -----------
    df : DataFrame
    v_col : str - voltage column name
    t_col : str - time column name  
    min_step_size : float - minimum voltage change to count as step (V)
    min_duration : float - minimum duration at a voltage level (s)
    smoothing_window : int - window size for median smoothing
    
    Returns:
    --------
    list of step dicts with idx, V_before, V_after, dV, type
    """
    voltage = df[v_col].values.copy()
    time = df[t_col].values.copy()
    
    # 1. Smooth voltage with rolling median to reduce noise
    if smoothing_window > 1 and len(voltage) > smoothing_window:
        voltage_smooth = pd.Series(voltage).rolling(
            window=smoothing_window, center=True, min_periods=1
        ).median().values
    else:
        voltage_smooth = voltage
    
    # 2. Round to voltage resolution to cluster similar values
    voltage_resolution = min_step_size / 2  # Round to half the step size
    voltage_rounded = np.round(voltage_smooth / voltage_resolution) * voltage_resolution
    
    # 3. Identify distinct voltage levels and their time ranges
    levels = []
    current_level = voltage_rounded[0]
    level_start_idx = 0
    
    for i in range(1, len(voltage_rounded)):
        if abs(voltage_rounded[i] - current_level) > min_step_size / 2:
            # Voltage changed - record previous level
            duration = time[i-1] - time[level_start_idx]
            if duration >= min_duration:
                levels.append({
                    'V': current_level,
                    'V_mean': np.mean(voltage[level_start_idx:i]),
                    'start_idx': level_start_idx,
                    'end_idx': i - 1,
                    'start_time': time[level_start_idx],
                    'end_time': time[i-1],
                    'duration': duration
                })
            current_level = voltage_rounded[i]
            level_start_idx = i
    
    # Don't forget the last level
    duration = time[-1] - time[level_start_idx]
    if duration >= min_duration:
        levels.append({
            'V': current_level,
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
    
    # 4. Identify transitions between levels
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
# SINGLE STEP FITTING
# =============================================================================

def fit_single_step(t, i, step_type='FALLING', t_fit_max=None, try_2exp=True):
    """
    Fit a single transient segment with 1 or 2 exponentials.
    
    Parameters:
    -----------
    t : array - time (starting from 0)
    i : array - current
    step_type : str - 'RISING' or 'FALLING'
    t_fit_max : float - max time to include in fit (None = use all)
    try_2exp : bool - if True, also try 2-exponential model
    
    Returns:
    --------
    dict with fit parameters, quality metrics, and model comparison
    """
    # Remove t <= 0 (1/√t diverges)
    mask = t > 0.0005  # 0.5 ms minimum
    if t_fit_max is not None:
        mask &= t <= t_fit_max
    
    t_fit = t[mask]
    i_fit = i[mask]
    
    if len(t_fit) < 10:
        return None
    
    n_points = len(t_fit)
    
    # =========================================================================
    # FIT 1-EXPONENTIAL MODEL
    # =========================================================================
    
    # Initial guesses
    A0 = i_fit[0] - i_fit[-1]
    tau0 = 0.1
    B0 = 0.1 * np.sign(A0) if A0 != 0 else 0.1
    C0 = i_fit[-1]
    
    p0_1exp = [A0, tau0, B0, C0]
    bounds_1exp = ([-np.inf, 0.001, -10, -np.inf], [np.inf, 10, 10, np.inf])
    
    try:
        popt_1exp, pcov_1exp = curve_fit(transient_model_1exp, t_fit, i_fit, 
                                          p0=p0_1exp, bounds=bounds_1exp, maxfev=20000)
        perr_1exp = np.sqrt(np.diag(pcov_1exp))
        
        i_pred_1exp = transient_model_1exp(t_fit, *popt_1exp)
        ss_res_1exp = np.sum((i_fit - i_pred_1exp)**2)
        ss_tot = np.sum((i_fit - np.mean(i_fit))**2)
        r2_1exp = 1 - ss_res_1exp / ss_tot
        
        aic_1exp, bic_1exp = calculate_aic_bic(n_points, 4, ss_res_1exp)
        
        fit_1exp = {
            'A': popt_1exp[0], 'A_err': perr_1exp[0],
            'tau': popt_1exp[1], 'tau_err': perr_1exp[1],
            'B': popt_1exp[2], 'B_err': perr_1exp[2],
            'C': popt_1exp[3], 'C_err': perr_1exp[3],
            'R2': r2_1exp, 'AIC': aic_1exp, 'BIC': bic_1exp,
            'ss_res': ss_res_1exp,
            'i_pred': i_pred_1exp
        }
        fit_1exp_success = True
    except Exception as e:
        print(f"    1-exp fit failed: {e}")
        fit_1exp_success = False
        fit_1exp = None
    
    # =========================================================================
    # FIT 2-EXPONENTIAL MODEL
    # =========================================================================
    
    fit_2exp = None
    fit_2exp_success = False
    
    if try_2exp and fit_1exp_success:
        # Initial guesses for 2-exp: split the amplitude, two different τ
        A1_0 = A0 * 0.6
        tau1_0 = 0.03   # Fast component ~30 ms
        A2_0 = A0 * 0.4
        tau2_0 = 0.15   # Slow component ~150 ms
        B0_2 = fit_1exp['B']  # Use 1-exp result
        C0_2 = fit_1exp['C']
        
        p0_2exp = [A1_0, tau1_0, A2_0, tau2_0, B0_2, C0_2]
        
        # Bounds: ensure tau1 < tau2 by convention (will sort after)
        bounds_2exp = ([-np.inf, 0.001, -np.inf, 0.001, -10, -np.inf], 
                       [np.inf, 10, np.inf, 10, 10, np.inf])
        
        try:
            popt_2exp, pcov_2exp = curve_fit(transient_model_2exp, t_fit, i_fit,
                                              p0=p0_2exp, bounds=bounds_2exp, maxfev=30000)
            perr_2exp = np.sqrt(np.diag(pcov_2exp))
            
            # Sort so tau1 < tau2 (fast < slow)
            A1, tau1, A2, tau2, B, C = popt_2exp
            A1_err, tau1_err, A2_err, tau2_err, B_err, C_err = perr_2exp
            
            if tau1 > tau2:
                # Swap
                A1, A2 = A2, A1
                tau1, tau2 = tau2, tau1
                A1_err, A2_err = A2_err, A1_err
                tau1_err, tau2_err = tau2_err, tau1_err
            
            i_pred_2exp = transient_model_2exp(t_fit, A1, tau1, A2, tau2, B, C)
            ss_res_2exp = np.sum((i_fit - i_pred_2exp)**2)
            r2_2exp = 1 - ss_res_2exp / ss_tot
            
            aic_2exp, bic_2exp = calculate_aic_bic(n_points, 6, ss_res_2exp)
            
            fit_2exp = {
                'A1': A1, 'A1_err': A1_err,
                'tau1': tau1, 'tau1_err': tau1_err,
                'A2': A2, 'A2_err': A2_err,
                'tau2': tau2, 'tau2_err': tau2_err,
                'B': B, 'B_err': B_err,
                'C': C, 'C_err': C_err,
                'R2': r2_2exp, 'AIC': aic_2exp, 'BIC': bic_2exp,
                'ss_res': ss_res_2exp,
                'i_pred': i_pred_2exp
            }
            fit_2exp_success = True
        except Exception as e:
            print(f"    2-exp fit failed: {e}")
            fit_2exp_success = False
    
    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    
    if not fit_1exp_success:
        return None
    
    # Determine which model is better
    if fit_2exp_success:
        # Use BIC for model selection (more conservative than AIC)
        delta_bic = fit_2exp['BIC'] - fit_1exp['BIC']
        delta_aic = fit_2exp['AIC'] - fit_1exp['AIC']
        
        # Negative delta = 2-exp is better
        # BIC difference > 10: very strong evidence
        # BIC difference 6-10: strong evidence
        # BIC difference 2-6: positive evidence
        # BIC difference 0-2: weak evidence
        
        prefer_2exp = delta_bic < -6  # Strong evidence for 2-exp
        
        if prefer_2exp:
            best_model = '2exp'
        else:
            best_model = '1exp'
    else:
        best_model = '1exp'
        delta_bic = None
        delta_aic = None
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    
    return {
        'fit_1exp': fit_1exp,
        'fit_2exp': fit_2exp,
        'best_model': best_model,
        'delta_BIC': delta_bic,
        'delta_AIC': delta_aic,
        't_fit': t_fit,
        'i_fit': i_fit
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_voltage_steps(df, t_col='time/s', v_col='Ewe/V', i_col='I/mA',
                          step_duration=4.0, t_skip=0.001, t_fit_max=None,
                          step_type_filter='both', min_step_size=0.05,
                          iR_interpolator=None):
    """
    Analyze all voltage steps in the data.
    
    Parameters:
    -----------
    df : DataFrame with time, voltage, current columns
    step_duration : float - max duration to consider for fitting (seconds)
    t_skip : float - time to skip at start of each step
    t_fit_max : float - max time to fit (None = use step_duration)
    step_type_filter : str - 'rising', 'falling', or 'both'
    min_step_size : float - minimum voltage change to count as step (V)
    iR_interpolator : function - V -> V_iR_corrected (optional)
    
    Returns:
    --------
    results_df : DataFrame with fit parameters for each step
    """
    # Detect steps with robust algorithm
    steps = detect_voltage_steps(df, v_col=v_col, t_col=t_col, 
                                  min_step_size=min_step_size)
    print(f"Found {len(steps)} voltage steps")
    
    # Filter by step type if requested
    if step_type_filter.lower() == 'falling':
        steps = [s for s in steps if s['type'] == 'FALLING']
        print(f"  Filtering to FALLING only: {len(steps)} steps")
    elif step_type_filter.lower() == 'rising':
        steps = [s for s in steps if s['type'] == 'RISING']
        print(f"  Filtering to RISING only: {len(steps)} steps")
    
    results = []
    
    for i, step in enumerate(steps):
        idx = step['idx']
        V_before = step['V_before']
        V_after = step['V_after']
        step_type = step['type']
        t_start = step['t_start']
        t_end = step['t_end']
        
        # Apply iR correction if available
        V_before_iR = apply_iR_correction(V_before, iR_interpolator)
        V_after_iR = apply_iR_correction(V_after, iR_interpolator)
        
        if iR_interpolator is not None:
            print(f"\nStep {i+1}: {V_before:.3f}V → {V_after:.3f}V ({step_type})")
            print(f"         iR-corrected: {V_before_iR:.3f}V → {V_after_iR:.3f}V")
        else:
            print(f"\nStep {i+1}: {V_before:.3f} → {V_after:.3f} V ({step_type})")
        print(f"         t = {t_start:.3f}s to {t_end:.3f}s ({t_end-t_start:.2f}s duration)")
        
        # Extract segment for this step - use the detected time range
        # but limit to step_duration if specified
        actual_duration = min(step['duration'], step_duration)
        t_end_fit = t_start + actual_duration
        
        mask = (df[t_col] >= t_start) & (df[t_col] <= t_end_fit)
        df_seg = df[mask].copy()
        
        # Reset time to start at 0
        t0 = df_seg[t_col].iloc[0]
        t = df_seg[t_col].values - t0
        current = df_seg[i_col].values
        
        # Skip initial points
        mask = t >= t_skip
        t = t[mask]
        current = current[mask]
        
        if len(t) < 20:
            print(f"    Skipping: not enough data points ({len(t)})")
            continue
        
        # Fit
        fit_result = fit_single_step(t, current, step_type, t_fit_max)
        
        if fit_result is not None and fit_result['fit_1exp'] is not None:
            fit_1exp = fit_result['fit_1exp']
            fit_2exp = fit_result['fit_2exp']
            best_model = fit_result['best_model']
            
            # Build result dict with 1-exp parameters (always available)
            result_dict = {
                'step': i + 1,
                'type': step_type,
                'V_initial': V_before,
                'V_final': V_after,
                'V_initial_iR': V_before_iR,
                'V_final_iR': V_after_iR,
                'dV': abs(step['dV']),
                'dV_iR': abs(V_after_iR - V_before_iR),
                'best_model': best_model,
                # 1-exp results
                'A': fit_1exp['A'],
                'A_err': fit_1exp['A_err'],
                'tau_ms': fit_1exp['tau'] * 1000,
                'tau_err_ms': fit_1exp['tau_err'] * 1000,
                'B': fit_1exp['B'],
                'B_err': fit_1exp['B_err'],
                'C': fit_1exp['C'],
                'C_err': fit_1exp['C_err'],
                'R2_1exp': fit_1exp['R2'],
                'AIC_1exp': fit_1exp['AIC'],
                'BIC_1exp': fit_1exp['BIC'],
                't_data': t,
                'i_data': current,
                't_fit': fit_result['t_fit'],
                'i_fit': fit_result['i_fit'],
                'i_pred_1exp': fit_1exp['i_pred']
            }
            
            # Add 2-exp results if available
            if fit_2exp is not None:
                result_dict.update({
                    'A1': fit_2exp['A1'],
                    'A1_err': fit_2exp['A1_err'],
                    'tau1_ms': fit_2exp['tau1'] * 1000,
                    'tau1_err_ms': fit_2exp['tau1_err'] * 1000,
                    'A2': fit_2exp['A2'],
                    'A2_err': fit_2exp['A2_err'],
                    'tau2_ms': fit_2exp['tau2'] * 1000,
                    'tau2_err_ms': fit_2exp['tau2_err'] * 1000,
                    'B_2exp': fit_2exp['B'],
                    'B_2exp_err': fit_2exp['B_err'],
                    'C_2exp': fit_2exp['C'],
                    'C_2exp_err': fit_2exp['C_err'],
                    'R2_2exp': fit_2exp['R2'],
                    'AIC_2exp': fit_2exp['AIC'],
                    'BIC_2exp': fit_2exp['BIC'],
                    'delta_BIC': fit_result['delta_BIC'],
                    'delta_AIC': fit_result['delta_AIC'],
                    'i_pred_2exp': fit_2exp['i_pred']
                })
            else:
                # Fill with NaN if 2-exp not available
                result_dict.update({
                    'A1': np.nan, 'A1_err': np.nan,
                    'tau1_ms': np.nan, 'tau1_err_ms': np.nan,
                    'A2': np.nan, 'A2_err': np.nan,
                    'tau2_ms': np.nan, 'tau2_err_ms': np.nan,
                    'B_2exp': np.nan, 'B_2exp_err': np.nan,
                    'C_2exp': np.nan, 'C_2exp_err': np.nan,
                    'R2_2exp': np.nan, 'AIC_2exp': np.nan, 'BIC_2exp': np.nan,
                    'delta_BIC': np.nan, 'delta_AIC': np.nan,
                    'i_pred_2exp': None
                })
            
            results.append(result_dict)
            
            # Print results
            print(f"\n    === 1-EXPONENTIAL MODEL ===")
            print(f"    A = {fit_1exp['A']:.3f} ± {fit_1exp['A_err']:.3f} mA")
            print(f"    τ = {fit_1exp['tau']*1000:.1f} ± {fit_1exp['tau_err']*1000:.1f} ms")
            print(f"    B = {fit_1exp['B']:.4f} ± {fit_1exp['B_err']:.4f} mA·s^0.5")
            print(f"    C = {fit_1exp['C']:.3f} ± {fit_1exp['C_err']:.3f} mA")
            print(f"    R² = {fit_1exp['R2']:.5f}, AIC = {fit_1exp['AIC']:.1f}, BIC = {fit_1exp['BIC']:.1f}")
            
            if fit_2exp is not None:
                print(f"\n    === 2-EXPONENTIAL MODEL ===")
                print(f"    A1 = {fit_2exp['A1']:.3f} ± {fit_2exp['A1_err']:.3f} mA (τ1 = {fit_2exp['tau1']*1000:.1f} ms) [FAST]")
                print(f"    A2 = {fit_2exp['A2']:.3f} ± {fit_2exp['A2_err']:.3f} mA (τ2 = {fit_2exp['tau2']*1000:.1f} ms) [SLOW]")
                print(f"    B  = {fit_2exp['B']:.4f} ± {fit_2exp['B_err']:.4f} mA·s^0.5")
                print(f"    C  = {fit_2exp['C']:.3f} ± {fit_2exp['C_err']:.3f} mA")
                print(f"    R² = {fit_2exp['R2']:.5f}, AIC = {fit_2exp['AIC']:.1f}, BIC = {fit_2exp['BIC']:.1f}")
                print(f"\n    ΔBIC = {fit_result['delta_BIC']:.1f} (negative favors 2-exp)")
                print(f"    >>> BEST MODEL: {best_model.upper()} <<<")
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_fits(results_df, save_prefix='fits'):
    """Plot all individual fits, showing both 1-exp and 2-exp models."""
    n_steps = len(results_df)
    
    # Two rows per step: top=data+fits, bottom=residuals
    fig, axes = plt.subplots(2, n_steps, figsize=(5*n_steps, 8))
    if n_steps == 1:
        axes = axes.reshape(2, 1)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax_fit = axes[0, i]
        ax_res = axes[1, i]
        
        t_ms = row['t_data'] * 1000
        t_fit_ms = row['t_fit'] * 1000
        
        # Plot data
        ax_fit.plot(t_ms, row['i_data'], 'k.', markersize=1, alpha=0.3, label='Data')
        
        # Plot 1-exp fit
        ax_fit.plot(t_fit_ms, row['i_pred_1exp'], 'b-', linewidth=2, 
                   label=f'1-exp (R²={row["R2_1exp"]:.4f})')
        
        # Plot 2-exp fit if available
        if row['i_pred_2exp'] is not None and not (isinstance(row['R2_2exp'], float) and np.isnan(row['R2_2exp'])):
            ax_fit.plot(t_fit_ms, row['i_pred_2exp'], 'r--', linewidth=2,
                       label=f'2-exp (R²={row["R2_2exp"]:.4f})')
        
        # Title
        best = row['best_model'] if 'best_model' in row else '1exp'
        title = f"Step {row['step']}: {row['V_initial']:.2f}→{row['V_final']:.2f}V"
        ax_fit.set_title(f"{title}\nBest: {best.upper()}", fontsize=10)
        ax_fit.set_xlabel('Time (ms)')
        ax_fit.set_ylabel('Current (mA)')
        ax_fit.legend(fontsize=8)
        ax_fit.grid(True, alpha=0.3)
        ax_fit.set_xlim([0, min(500, t_ms.max())])
        
        # Residuals
        res_1exp = row['i_fit'] - row['i_pred_1exp']
        ax_res.plot(t_fit_ms, res_1exp, 'b.', markersize=1, alpha=0.5, label='1-exp residuals')
        
        if row['i_pred_2exp'] is not None and not (isinstance(row['R2_2exp'], float) and np.isnan(row['R2_2exp'])):
            res_2exp = row['i_fit'] - row['i_pred_2exp']
            ax_res.plot(t_fit_ms, res_2exp, 'r.', markersize=1, alpha=0.5, label='2-exp residuals')
        
        ax_res.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax_res.set_xlabel('Time (ms)')
        ax_res.set_ylabel('Residual (mA)')
        ax_res.legend(fontsize=8)
        ax_res.grid(True, alpha=0.3)
        ax_res.set_xlim([0, min(500, t_ms.max())])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_all_fits.png', dpi=150)
    plt.show()


def plot_model_comparison(results_df, save_prefix='fits'):
    """Plot comparison of 1-exp vs 2-exp models."""
    
    # Only include steps where 2-exp was tried
    has_2exp = results_df['R2_2exp'].notna()
    if not has_2exp.any():
        print("No 2-exp fits to compare")
        return
    
    df = results_df[has_2exp].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 1. R² comparison
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['R2_1exp'], width, label='1-exp', color='blue', alpha=0.7)
    ax.bar(x + width/2, df['R2_2exp'], width, label='2-exp', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in df['step']])
    ax.set_ylabel('R²')
    ax.set_title('Goodness of Fit')
    ax.legend()
    ax.set_ylim([min(0.95, df[['R2_1exp', 'R2_2exp']].min().min() - 0.01), 1.0])
    ax.grid(True, alpha=0.3)
    
    # 2. ΔBIC
    ax = axes[0, 1]
    colors = ['green' if d < -6 else 'orange' if d < 0 else 'gray' for d in df['delta_BIC']]
    ax.bar(x, df['delta_BIC'], color=colors, alpha=0.7)
    ax.axhline(-6, color='green', linestyle='--', label='Strong evidence for 2-exp')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in df['step']])
    ax.set_ylabel('ΔBIC (2exp - 1exp)')
    ax.set_title('Model Selection (negative = 2-exp better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Time constants comparison
    ax = axes[0, 2]
    ax.errorbar(x - 0.2, df['tau_ms'], yerr=df['tau_err_ms'], 
               fmt='bo', markersize=8, capsize=3, label='1-exp: τ')
    ax.errorbar(x, df['tau1_ms'], yerr=df['tau1_err_ms'],
               fmt='r^', markersize=8, capsize=3, label='2-exp: τ₁ (fast)')
    ax.errorbar(x + 0.2, df['tau2_ms'], yerr=df['tau2_err_ms'],
               fmt='rs', markersize=8, capsize=3, label='2-exp: τ₂ (slow)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in df['step']])
    ax.set_ylabel('τ (ms)')
    ax.set_title('Time Constants')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Amplitudes comparison
    ax = axes[1, 0]
    ax.bar(x - 0.2, np.abs(df['A']), 0.2, label='1-exp: |A|', color='blue', alpha=0.7)
    ax.bar(x, np.abs(df['A1']), 0.2, label='2-exp: |A₁|', color='red', alpha=0.7)
    ax.bar(x + 0.2, np.abs(df['A2']), 0.2, label='2-exp: |A₂|', color='darkred', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in df['step']])
    ax.set_ylabel('|Amplitude| (mA)')
    ax.set_title('Amplitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. B comparison
    ax = axes[1, 1]
    ax.errorbar(x - 0.15, np.abs(df['B']), yerr=df['B_err'],
               fmt='bo', markersize=8, capsize=3, label='1-exp')
    ax.errorbar(x + 0.15, np.abs(df['B_2exp']), yerr=df['B_2exp_err'],
               fmt='rs', markersize=8, capsize=3, label='2-exp')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in df['step']])
    ax.set_ylabel('|B| (mA·s^0.5)')
    ax.set_title('Diffusion Component')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. τ ratio (τ2/τ1) - indicates separation of processes
    ax = axes[1, 2]
    tau_ratio = df['tau2_ms'] / df['tau1_ms']
    ax.bar(x, tau_ratio, color='purple', alpha=0.7)
    ax.axhline(3, color='green', linestyle='--', label='Good separation (3x)')
    ax.axhline(10, color='red', linestyle='--', label='Strong separation (10x)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in df['step']])
    ax.set_ylabel('τ₂/τ₁ ratio')
    ax.set_title('Time Scale Separation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_model_comparison.png', dpi=150)
    plt.show()


def plot_parameters_vs_voltage(results_df, save_prefix='fits', use_iR_corrected=True):
    """Plot fit parameters vs voltage, separated by rising/falling.
    
    For proper symmetry comparison:
    - RISING: plot against V_final (target voltage)
    - FALLING: plot against V_initial (starting voltage)
    This way both directions at "the same voltage" can be compared.
    
    If iR-corrected voltages available, uses those for plotting.
    """
    
    rising = results_df[results_df['type'] == 'RISING'].copy()
    falling = results_df[results_df['type'] == 'FALLING'].copy()
    
    # Check if iR-corrected voltages are available and different from raw
    has_iR = 'V_initial_iR' in results_df.columns
    if has_iR and use_iR_corrected:
        # Check if they're actually different
        if not np.allclose(results_df['V_initial'], results_df['V_initial_iR'], atol=0.001):
            v_suffix = '_iR'
            v_label = 'iR-corrected Voltage (V)'
            print("  Using iR-corrected voltages for plots")
        else:
            v_suffix = ''
            v_label = 'Voltage (V)'
    else:
        v_suffix = ''
        v_label = 'Voltage (V)'
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Helper for plotting with error bars
    # For rising: use V_final; for falling: use V_initial
    def plot_param(ax, df_r, df_f, param, ylabel, use_abs=False, use_V_final_for_C=False):
        """
        For most parameters: rising uses V_final, falling uses V_initial
        For C (steady-state): always use V_final
        """
        # Handle special case for tau_ms
        err_col = 'tau_err_ms' if param == 'tau_ms' else f'{param}_err'
        
        # Determine which voltage columns to use
        v_final_col = f'V_final{v_suffix}' if f'V_final{v_suffix}' in df_r.columns else 'V_final'
        v_initial_col = f'V_initial{v_suffix}' if f'V_initial{v_suffix}' in df_f.columns else 'V_initial'
        
        if len(df_r) > 0:
            x_r = df_r[v_final_col]  # Rising: plot at target voltage
            y_r = np.abs(df_r[param]) if use_abs else df_r[param]
            ax.errorbar(x_r, y_r, yerr=df_r[err_col], 
                       fmt='r^', markersize=10, capsize=4, label='Rising', alpha=0.8, linewidth=2)
        if len(df_f) > 0:
            # For C, use V_final; for other params, use V_initial
            x_f = df_f[v_final_col if use_V_final_for_C else v_initial_col]
            y_f = np.abs(df_f[param]) if use_abs else df_f[param]
            ax.errorbar(x_f, y_f, yerr=df_f[err_col], 
                       fmt='bv', markersize=10, capsize=4, label='Falling', alpha=0.8, linewidth=2)
        
        ax.set_xlabel(v_label)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # |A| vs voltage (rising@V_final, falling@V_initial)
    plot_param(axes[0,0], rising, falling, 'A', '|A| (mA)', use_abs=True)
    axes[0,0].set_title('Amplitude |A| vs Voltage\n(Rising@V_final, Falling@V_initial)')
    
    # τ vs voltage
    plot_param(axes[0,1], rising, falling, 'tau_ms', 'τ (ms)')
    axes[0,1].set_title('Time Constant τ vs Voltage\n(Rising@V_final, Falling@V_initial)')
    
    # |B| vs voltage
    plot_param(axes[0,2], rising, falling, 'B', '|B| (mA·s^0.5)', use_abs=True)
    axes[0,2].set_title('Diffusion |B| vs Voltage\n(Rising@V_final, Falling@V_initial)')
    
    # C vs V_final (C should ALWAYS depend on final voltage!)
    plot_param(axes[1,0], rising, falling, 'C', 'C (mA)', use_V_final_for_C=True)
    axes[1,0].set_title('Steady-State C vs V_final\n(Both use V_final)')
    
    # |B| vs dV (should be linear if Bazant interpretation correct)
    ax = axes[1,1]
    dV_col = f'dV{v_suffix}' if f'dV{v_suffix}' in results_df.columns else 'dV'
    if len(rising) > 0:
        ax.errorbar(rising[dV_col], np.abs(rising['B']), yerr=rising['B_err'],
                   fmt='r^', markersize=10, capsize=4, label='Rising', alpha=0.8, linewidth=2)
    if len(falling) > 0:
        ax.errorbar(falling[dV_col], np.abs(falling['B']), yerr=falling['B_err'],
                   fmt='bv', markersize=10, capsize=4, label='Falling', alpha=0.8, linewidth=2)
    
    # Add linear fit using |B|
    all_dV = results_df[dV_col].values
    all_B_abs = np.abs(results_df['B'].values)
    if len(all_dV) > 2 and np.std(all_dV) > 1e-6:  # Check for variation in dV
        coeffs = np.polyfit(all_dV, all_B_abs, 1)
        dV_line = np.linspace(0, all_dV.max() * 1.1, 100)
        ax.plot(dV_line, coeffs[0]*dV_line + coeffs[1], 'k--', 
               label=f'Linear: |B| = {coeffs[0]:.3f}·ΔV + {coeffs[1]:.3f}')
    
    ax.set_xlabel('|ΔV| (V)')
    ax.set_ylabel('|B| (mA·s^0.5)')
    ax.set_title('|B| vs ΔV (Linearity Test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # τ comparison: Rising vs Falling at same voltage
    ax = axes[1,2]
    # Create paired comparison if we have matching voltages
    if len(rising) > 0 and len(falling) > 0:
        # For each rising V_final, find falling with same V_initial
        paired_data = []
        for _, r_row in rising.iterrows():
            v_target = r_row['V_final']
            # Find falling step from this voltage
            f_match = falling[np.abs(falling['V_initial'] - v_target) < 0.05]
            if len(f_match) > 0:
                paired_data.append({
                    'V': v_target,
                    'tau_rising': r_row['tau_ms'],
                    'tau_falling': f_match.iloc[0]['tau_ms']
                })
        
        if paired_data:
            paired_df = pd.DataFrame(paired_data)
            x = np.arange(len(paired_df))
            width = 0.35
            ax.bar(x - width/2, paired_df['tau_rising'], width, label='Rising', color='red', alpha=0.7)
            ax.bar(x + width/2, paired_df['tau_falling'], width, label='Falling', color='blue', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{v:.2f}V' for v in paired_df['V']])
            ax.set_xlabel('Voltage')
            ax.set_ylabel('τ (ms)')
            ax.set_title('τ Asymmetry: Rising vs Falling')
            ax.legend()
        else:
            # No paired data, just show bar chart
            ax.bar(results_df['step'], results_df['tau_ms'], 
                   color=['red' if t=='RISING' else 'blue' for t in results_df['type']], alpha=0.7)
            ax.set_xlabel('Step Number')
            ax.set_ylabel('τ (ms)')
            ax.set_title('Time Constant by Step')
    else:
        ax.bar(results_df['step'], results_df['tau_ms'], 
               color=['red' if t=='RISING' else 'blue' for t in results_df['type']], alpha=0.7)
        ax.set_xlabel('Step Number')
        ax.set_ylabel('τ (ms)')
        ax.set_title('Time Constant by Step')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_parameters.png', dpi=150)
    plt.show()


def plot_summary_table(results_df, save_prefix='fits'):
    """Create a summary table figure."""
    
    # Check if iR-corrected voltages are available
    has_iR = 'V_initial_iR' in results_df.columns
    use_iR = has_iR and not np.allclose(results_df['V_initial'], results_df['V_initial_iR'], atol=0.001)
    
    # Check if 2-exp results are available
    has_2exp = 'R2_2exp' in results_df.columns and results_df['R2_2exp'].notna().any()
    
    # Select columns for table
    if use_iR:
        base_cols = ['step', 'type', 'V_initial_iR', 'V_final_iR', 'dV_iR']
        col_names = ['Step', 'Type', 'V_i_iR (V)', 'V_f_iR (V)', '|ΔV|_iR']
    else:
        base_cols = ['step', 'type', 'V_initial', 'V_final', 'dV']
        col_names = ['Step', 'Type', 'V_i (V)', 'V_f (V)', '|ΔV|']
    
    if has_2exp:
        # Show both models
        table_cols = base_cols + ['A', 'tau_ms', 'R2_1exp', 'tau1_ms', 'tau2_ms', 'R2_2exp', 'best_model']
        col_names += ['A (mA)', 'τ_1exp (ms)', 'R²_1exp', 'τ₁ (ms)', 'τ₂ (ms)', 'R²_2exp', 'Best']
    else:
        table_cols = base_cols + ['A', 'tau_ms', 'B', 'C', 'R2_1exp']
        col_names += ['A (mA)', 'τ (ms)', 'B', 'C (mA)', 'R²']
    
    table_df = results_df[table_cols].copy()
    table_df.columns = col_names
    
    # Format numbers
    for col in table_df.columns:
        if 'V_' in col or 'ΔV' in col:
            table_df[col] = table_df[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else '')
        elif col in ['A (mA)', 'C (mA)']:
            table_df[col] = table_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')
        elif 'τ' in col or 'tau' in col.lower():
            table_df[col] = table_df[col].apply(lambda x: f'{x:.1f}' if pd.notna(x) else '')
        elif col == 'B':
            table_df[col] = table_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
        elif 'R²' in col:
            table_df[col] = table_df[col].apply(lambda x: f'{x:.5f}' if pd.notna(x) else '')
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(table_df)*0.4)))
    ax.axis('off')
    
    table = ax.table(cellText=table_df.values,
                     colLabels=table_df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['lightblue']*len(table_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color rows by step type
    for i, (_, row) in enumerate(results_df.iterrows()):
        color = '#ffcccc' if row['type'] == 'RISING' else '#ccccff'
        for j in range(len(table_df.columns)):
            table[(i+1, j)].set_facecolor(color)
    
    plt.title('Fit Parameters Summary\n(Red=Rising, Blue=Falling)', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_table.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return table_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    
    # =========================================================================
    # ANALYSIS PARAMETERS (defaults - may be overridden by file loading)
    # =========================================================================
    
    # Column names (adjust if your data uses different names)
    T_COL = 'time/s'
    V_COL = 'Ewe/V'
    I_COL = 'I/mA'
    
    # Fitting parameters
    STEP_DURATION = 4.0    # Max duration to consider for each step (s)
    T_SKIP = 0.002         # Time to skip at start of step (s) - avoid t=0 singularity
    T_FIT_MAX = None       # Max time to include in fit (None = use all data)
    STEP_TYPE_FILTER = 'falling'  # 'rising', 'falling', or 'both'
    
    # Output prefix
    OUTPUT_PREFIX = 'voltage_step_fits'
    
    # =========================================================================
    # LOAD YOUR DATA HERE
    # =========================================================================
    
    # Option 1: From command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"Loading: {filename}")
        
        # Check for sheet name argument
        sheet_name = sys.argv[2] if len(sys.argv) > 2 else 0
        
        # Check for step type filter argument
        if len(sys.argv) > 3:
            STEP_TYPE_FILTER = sys.argv[3].lower()
            print(f"  Step type filter: {STEP_TYPE_FILTER}")
        
        # Check for iR correction lookup file
        iR_interpolator = None
        if len(sys.argv) > 4:
            iR_file = sys.argv[4]
            iR_interpolator = load_iR_correction_table(iR_file)
        
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            # For Excel, try to load with sheet name
            try:
                df = pd.read_excel(filename, sheet_name=sheet_name)
                original_cols = df.columns.tolist()
                
                # Check if first row contains units (common in electrochemistry data)
                first_row = df.iloc[0].astype(str).tolist()
                if any(val in ['s', 'V', 'mA', 'µA', 'A', 'ms'] for val in first_row):
                    print(f"  Detected units row: {first_row}")
                    # Keep original column names, just skip the units row
                    df = df.iloc[1:].reset_index(drop=True)
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"  Using original headers: {original_cols}")
            except Exception as e:
                print(f"  Error loading sheet '{sheet_name}': {e}")
                print(f"  Available sheets: {pd.ExcelFile(filename).sheet_names}")
                sys.exit(1)
        else:
            df = pd.read_csv(filename)
        
        # Try to identify columns automatically
        cols = df.columns.tolist()
        print(f"  Columns found: {cols}")
        
        # Common column name patterns
        time_patterns = ['time', 't/', 'time/s', 'Time']
        voltage_patterns = ['ewe', 'potential', 'voltage', 'Ewe/V', 'E/V', 'Non-iR', 'corrected potential']
        current_patterns = ['current', 'I/mA', 'i/mA', 'I/A', 'Current']
        
        def find_column(patterns, columns):
            for col in columns:
                col_lower = str(col).lower()
                for pat in patterns:
                    if pat.lower() in col_lower:
                        return col
            return None
        
        t_col = find_column(time_patterns, cols)
        v_col = find_column(voltage_patterns, cols)
        i_col = find_column(current_patterns, cols)
        
        if t_col and v_col and i_col:
            print(f"  Auto-detected: time='{t_col}', voltage='{v_col}', current='{i_col}'")
            T_COL, V_COL, I_COL = t_col, v_col, i_col
        else:
            # Fallback: assume first 3 columns are time, voltage, current
            if len(cols) >= 3:
                T_COL, V_COL, I_COL = cols[0], cols[1], cols[2]
                print(f"  Using positional columns: time='{T_COL}', voltage='{V_COL}', current='{I_COL}'")
    
    # Option 2: Generate demo data (comment out when using real data)
    else:
        print("No file provided. Generating demo data...")
        
        # Simulate stepped voltage data
        dt = 0.001  # 1 ms sampling
        step_duration = 2.0  # 2 seconds per step
        
        # Voltage levels (rising then falling)
        voltages = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
        
        t_all, V_all, I_all = [], [], []
        t_current = 0
        
        for i, V in enumerate(voltages):
            n_points = int(step_duration / dt)
            t_step = np.arange(n_points) * dt
            
            # Generate transient for this step
            if i == 0:
                # First step: no transient
                I_step = np.zeros(n_points) + 0.01 * np.random.randn(n_points)
            else:
                dV = V - voltages[i-1]
                # Transient parameters scale with dV
                A = 22.4 * abs(dV)
                tau = 0.095
                B = 0.60 * abs(dV)
                C = 0.5 * (V - 1.3)**2  # OER current increases with potential
                
                # Sign convention: positive for both rising and falling in this demo
                I_step = A * np.exp(-t_step / tau) + B / np.sqrt(t_step + 0.0001) + C
                I_step += 0.02 * np.random.randn(n_points)  # Add noise
            
            t_all.extend(t_current + t_step)
            V_all.extend([V] * n_points)
            I_all.extend(I_step)
            t_current += step_duration
        
        df = pd.DataFrame({
            'time/s': t_all,
            'Ewe/V': V_all,
            'I/mA': I_all
        })
        
        print(f"Demo data: {len(df)} points, {len(voltages)} voltage levels")
        iR_interpolator = None  # No iR correction for demo data
    
    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    
    print("\n" + "="*60)
    print("VOLTAGE-STEPPED CHRONOAMPEROMETRY ANALYSIS")
    print("="*60)
    
    # Check columns exist
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Run analysis
    results_df = analyze_voltage_steps(
        df, 
        t_col=T_COL, 
        v_col=V_COL, 
        i_col=I_COL,
        step_duration=STEP_DURATION,
        t_skip=T_SKIP,
        t_fit_max=T_FIT_MAX,
        step_type_filter=STEP_TYPE_FILTER,
        iR_interpolator=iR_interpolator
    )
    
    if len(results_df) == 0:
        print("\nNo steps could be fitted!")
        sys.exit(1)
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Plot all individual fits
    plot_all_fits(results_df, save_prefix=OUTPUT_PREFIX)
    
    # Plot model comparison (1-exp vs 2-exp)
    plot_model_comparison(results_df, save_prefix=OUTPUT_PREFIX)
    
    # Plot parameters vs voltage
    plot_parameters_vs_voltage(results_df, save_prefix=OUTPUT_PREFIX)
    
    # Summary table
    summary_table = plot_summary_table(results_df, save_prefix=OUTPUT_PREFIX)
    
    # Save results to CSV - drop array columns
    cols_to_drop = ['t_data', 'i_data', 't_fit', 'i_fit', 'i_pred_1exp', 'i_pred_2exp']
    cols_to_drop = [c for c in cols_to_drop if c in results_df.columns]
    results_export = results_df.drop(columns=cols_to_drop)
    results_export.to_csv(f'{OUTPUT_PREFIX}_results.csv', index=False)
    print(f"\nResults saved to: {OUTPUT_PREFIX}_results.csv")
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    rising = results_df[results_df['type'] == 'RISING']
    falling = results_df[results_df['type'] == 'FALLING']
    
    # Check if 2-exp is preferred
    has_2exp = 'best_model' in results_df.columns
    if has_2exp:
        n_2exp = (results_df['best_model'] == '2exp').sum()
        n_1exp = (results_df['best_model'] == '1exp').sum()
        print(f"\nMODEL SELECTION:")
        print(f"  1-exp preferred: {n_1exp} steps")
        print(f"  2-exp preferred: {n_2exp} steps")
    
    print(f"\nRISING steps: {len(rising)}")
    if len(rising) > 0:
        print(f"  1-exp τ mean: {rising['tau_ms'].mean():.1f} ± {rising['tau_ms'].std():.1f} ms")
        print(f"  |B|/ΔV mean: {(np.abs(rising['B'])/rising['dV']).mean():.3f} mA·s^0.5/V")
        if has_2exp and rising['tau1_ms'].notna().any():
            print(f"  2-exp τ₁ mean: {rising['tau1_ms'].mean():.1f} ms (fast)")
            print(f"  2-exp τ₂ mean: {rising['tau2_ms'].mean():.1f} ms (slow)")
    
    print(f"\nFALLING steps: {len(falling)}")
    if len(falling) > 0:
        print(f"  1-exp τ mean: {falling['tau_ms'].mean():.1f} ± {falling['tau_ms'].std():.1f} ms")
        print(f"  |B|/ΔV mean: {(np.abs(falling['B'])/falling['dV']).mean():.3f} mA·s^0.5/V")
        if has_2exp and falling['tau1_ms'].notna().any():
            print(f"  2-exp τ₁ mean: {falling['tau1_ms'].mean():.1f} ms (fast)")
            print(f"  2-exp τ₂ mean: {falling['tau2_ms'].mean():.1f} ms (slow)")
            print(f"  τ₂/τ₁ ratio: {(falling['tau2_ms']/falling['tau1_ms']).mean():.1f}x")
    
    # Linearity test
    if len(results_df) > 2:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            results_df['dV'], np.abs(results_df['B'])
        )
        print(f"\nLINEARITY TEST (B vs ΔV):")
        print(f"  Slope: {slope:.4f} mA·s^0.5/V")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.2e}")
        if r_value**2 > 0.9:
            print("  ✓ Strong linear relationship - supports Bazant interpretation")
        else:
            print("  ⚠ Weak linearity - may indicate nonlinear effects")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
