"""
Voltage-Stepped Chronoamperometry Fitting Script
=================================================

Automatically:
1. Detects voltage step transitions
2. Fits each transient with: i(t) = A*exp(-t/τ) + B/√t + C
3. Separates RISING vs FALLING steps
4. Plots parameters vs voltage

Usage:
    python fit_voltage_steps.py your_data.csv
    python fit_voltage_steps.py your_data.xlsx SheetName
    python fit_voltage_steps.py your_data.xlsx SheetName falling
    
Arguments:
    file      : CSV or Excel file with electrochemistry data
    sheet     : (optional) Sheet name for Excel files
    step_type : (optional) 'rising', 'falling', or 'both' (default: 'falling')
    
Expected columns: 'time/s', 'Ewe/V', 'I/mA' (or similar, auto-detected)

Note: Rising steps often have poor fits due to additional physics 
      (OER, bubble formation). Default is to fit only falling steps.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys

# =============================================================================
# MODEL
# =============================================================================

def transient_model(t, A, tau, B, C):
    """i(t) = A*exp(-t/τ) + B/√t + C"""
    return A * np.exp(-t / tau) + B / np.sqrt(t) + C


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

def fit_single_step(t, i, step_type='FALLING', t_fit_max=None):
    """
    Fit a single transient segment.
    
    Parameters:
    -----------
    t : array - time (starting from 0)
    i : array - current
    step_type : str - 'RISING' or 'FALLING'
    t_fit_max : float - max time to include in fit (None = use all)
    
    Returns:
    --------
    dict with fit parameters and quality metrics
    """
    # Remove t <= 0 (1/√t diverges)
    mask = t > 0.0005  # 0.5 ms minimum
    if t_fit_max is not None:
        mask &= t <= t_fit_max
    
    t_fit = t[mask]
    i_fit = i[mask]
    
    if len(t_fit) < 10:
        return None
    
    # Initial guesses - account for sign based on step type
    A0 = i_fit[0] - i_fit[-1]  # Amplitude from first to last (preserves sign)
    tau0 = 0.1                  # 100 ms guess
    B0 = 0.1 * np.sign(A0) if A0 != 0 else 0.1  # B typically has same sign as A
    C0 = i_fit[-1]              # Steady state from last points
    
    p0 = [A0, tau0, B0, C0]
    
    # Bounds - allow negative A and B for falling steps
    if step_type == 'FALLING':
        # Falling: A < 0, B < 0 (current decreases, diffusion opposes)
        bounds = ([-np.inf, 0.001, -10, -np.inf], [np.inf, 10, 10, np.inf])
    else:
        # Rising: A > 0, B > 0 typically
        bounds = ([-np.inf, 0.001, -10, -np.inf], [np.inf, 10, 10, np.inf])
    
    try:
        popt, pcov = curve_fit(transient_model, t_fit, i_fit, p0=p0, 
                               bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate R²
        i_pred = transient_model(t_fit, *popt)
        ss_res = np.sum((i_fit - i_pred)**2)
        ss_tot = np.sum((i_fit - np.mean(i_fit))**2)
        r2 = 1 - ss_res / ss_tot
        
        return {
            'A': popt[0],
            'tau': popt[1],
            'B': popt[2],
            'C': popt[3],
            'A_err': perr[0],
            'tau_err': perr[1],
            'B_err': perr[2],
            'C_err': perr[3],
            'R2': r2,
            't_fit': t_fit,
            'i_fit': i_fit,
            'i_pred': i_pred
        }
    except Exception as e:
        print(f"    Fitting failed: {e}")
        return None


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_voltage_steps(df, t_col='time/s', v_col='Ewe/V', i_col='I/mA',
                          step_duration=4.0, t_skip=0.001, t_fit_max=None,
                          step_type_filter='both', min_step_size=0.05):
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
        
        if fit_result is not None:
            results.append({
                'step': i + 1,
                'type': step_type,
                'V_initial': V_before,
                'V_final': V_after,
                'dV': abs(step['dV']),
                'A': fit_result['A'],
                'A_err': fit_result['A_err'],
                'tau_ms': fit_result['tau'] * 1000,
                'tau_err_ms': fit_result['tau_err'] * 1000,
                'B': fit_result['B'],
                'B_err': fit_result['B_err'],
                'C': fit_result['C'],
                'C_err': fit_result['C_err'],
                'R2': fit_result['R2'],
                't_data': t,
                'i_data': current,
                't_fit': fit_result['t_fit'],
                'i_fit': fit_result['i_fit'],
                'i_pred': fit_result['i_pred']
            })
            
            print(f"    A = {fit_result['A']:.3f} ± {fit_result['A_err']:.3f} mA")
            print(f"    τ = {fit_result['tau']*1000:.1f} ± {fit_result['tau_err']*1000:.1f} ms")
            print(f"    B = {fit_result['B']:.4f} ± {fit_result['B_err']:.4f} mA·s^0.5")
            print(f"    C = {fit_result['C']:.3f} ± {fit_result['C_err']:.3f} mA")
            print(f"    R² = {fit_result['R2']:.5f}")
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_fits(results_df, save_prefix='fits'):
    """Plot all individual fits."""
    n_steps = len(results_df)
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_steps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax = axes[i]
        
        # Plot data
        ax.plot(row['t_data']*1000, row['i_data'], 'b.', markersize=1, alpha=0.3)
        # Plot fit
        ax.plot(row['t_fit']*1000, row['i_pred'], 'r-', linewidth=1.5)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('I (mA)')
        title = f"{row['V_initial']:.2f}→{row['V_final']:.2f}V"
        ax.set_title(f"{title}\nR²={row['R2']:.4f}", fontsize=9)
        ax.set_xlim([0, row['t_data'].max()*1000])
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_all_fits.png', dpi=150)
    plt.show()


def plot_parameters_vs_voltage(results_df, save_prefix='fits'):
    """Plot fit parameters vs voltage, separated by rising/falling.
    
    For proper symmetry comparison:
    - RISING: plot against V_final (target voltage)
    - FALLING: plot against V_initial (starting voltage)
    This way both directions at "the same voltage" can be compared.
    """
    
    rising = results_df[results_df['type'] == 'RISING'].copy()
    falling = results_df[results_df['type'] == 'FALLING'].copy()
    
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
        
        if len(df_r) > 0:
            x_r = df_r['V_final']  # Rising: plot at target voltage
            y_r = np.abs(df_r[param]) if use_abs else df_r[param]
            ax.errorbar(x_r, y_r, yerr=df_r[err_col], 
                       fmt='r^', markersize=10, capsize=4, label='Rising', alpha=0.8, linewidth=2)
        if len(df_f) > 0:
            # For C, use V_final; for other params, use V_initial
            x_f = df_f['V_final'] if use_V_final_for_C else df_f['V_initial']
            y_f = np.abs(df_f[param]) if use_abs else df_f[param]
            ax.errorbar(x_f, y_f, yerr=df_f[err_col], 
                       fmt='bv', markersize=10, capsize=4, label='Falling', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Voltage (V)')
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
    if len(rising) > 0:
        ax.errorbar(rising['dV'], np.abs(rising['B']), yerr=rising['B_err'],
                   fmt='r^', markersize=10, capsize=4, label='Rising', alpha=0.8, linewidth=2)
    if len(falling) > 0:
        ax.errorbar(falling['dV'], np.abs(falling['B']), yerr=falling['B_err'],
                   fmt='bv', markersize=10, capsize=4, label='Falling', alpha=0.8, linewidth=2)
    
    # Add linear fit using |B|
    all_dV = results_df['dV'].values
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
    
    # Select columns for table
    table_df = results_df[['step', 'type', 'V_initial', 'V_final', 'dV', 
                           'A', 'tau_ms', 'B', 'C', 'R2']].copy()
    
    # Format numbers
    table_df['V_initial'] = table_df['V_initial'].apply(lambda x: f'{x:.3f}')
    table_df['V_final'] = table_df['V_final'].apply(lambda x: f'{x:.3f}')
    table_df['dV'] = table_df['dV'].apply(lambda x: f'{x:.3f}')
    table_df['A'] = table_df['A'].apply(lambda x: f'{x:.2f}')
    table_df['tau_ms'] = table_df['tau_ms'].apply(lambda x: f'{x:.1f}')
    table_df['B'] = table_df['B'].apply(lambda x: f'{x:.4f}')
    table_df['C'] = table_df['C'].apply(lambda x: f'{x:.3f}')
    table_df['R2'] = table_df['R2'].apply(lambda x: f'{x:.5f}')
    
    # Rename columns for display
    table_df.columns = ['Step', 'Type', 'V_i (V)', 'V_f (V)', '|ΔV|', 
                        'A (mA)', 'τ (ms)', 'B', 'C (mA)', 'R²']
    
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
        step_type_filter=STEP_TYPE_FILTER
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
    
    # Plot parameters vs voltage
    plot_parameters_vs_voltage(results_df, save_prefix=OUTPUT_PREFIX)
    
    # Summary table
    summary_table = plot_summary_table(results_df, save_prefix=OUTPUT_PREFIX)
    
    # Save results to CSV
    results_export = results_df.drop(columns=['t_data', 'i_data', 't_fit', 'i_fit', 'i_pred'])
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
    
    print(f"\nRISING steps: {len(rising)}")
    if len(rising) > 0:
        print(f"  τ mean: {rising['tau_ms'].mean():.1f} ± {rising['tau_ms'].std():.1f} ms")
        print(f"  |B|/ΔV mean: {(np.abs(rising['B'])/rising['dV']).mean():.3f} mA·s^0.5/V")
    
    print(f"\nFALLING steps: {len(falling)}")
    if len(falling) > 0:
        print(f"  τ mean: {falling['tau_ms'].mean():.1f} ± {falling['tau_ms'].std():.1f} ms")
        print(f"  |B|/ΔV mean: {(np.abs(falling['B'])/falling['dV']).mean():.3f} mA·s^0.5/V")
    
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
