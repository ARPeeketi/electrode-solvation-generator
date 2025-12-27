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
    
Expected columns: 'time/s', 'Ewe/V', 'I/mA'
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
# STEP DETECTION
# =============================================================================

def detect_voltage_steps(df, v_col='Ewe/V', threshold=0.01):
    """
    Detect indices where voltage changes by more than threshold.
    Returns list of (start_idx, V_before, V_after) tuples.
    """
    voltage = df[v_col].values
    steps = []
    
    for i in range(1, len(voltage)):
        dV = voltage[i] - voltage[i-1]
        if abs(dV) > threshold:
            steps.append({
                'idx': i,
                'V_before': voltage[i-1],
                'V_after': voltage[i],
                'dV': dV,
                'type': 'RISING' if dV > 0 else 'FALLING'
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
    
    # Initial guesses
    A0 = i_fit[0] - i_fit[-1]  # Amplitude from first to last
    tau0 = 0.1                  # 100 ms guess
    B0 = 0.1                    # Small diffusion term
    C0 = i_fit[-1]              # Steady state from last points
    
    p0 = [A0, tau0, B0, C0]
    
    # Bounds
    if step_type == 'FALLING':
        # Falling: A > 0, B > 0 typically
        bounds = ([0, 0.001, -1, -np.inf], [np.inf, 10, 5, np.inf])
    else:
        # Rising: A > 0, B can be positive or negative
        bounds = ([0, 0.001, -5, -np.inf], [np.inf, 10, 5, np.inf])
    
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
                          step_duration=4.0, t_skip=0.001, t_fit_max=None):
    """
    Analyze all voltage steps in the data.
    
    Parameters:
    -----------
    df : DataFrame with time, voltage, current columns
    step_duration : float - expected duration of each step (seconds)
    t_skip : float - time to skip at start of each step
    t_fit_max : float - max time to fit (None = use step_duration)
    
    Returns:
    --------
    results_df : DataFrame with fit parameters for each step
    """
    # Detect steps
    steps = detect_voltage_steps(df, v_col=v_col)
    print(f"Found {len(steps)} voltage steps")
    
    results = []
    
    for i, step in enumerate(steps):
        idx = step['idx']
        V_before = step['V_before']
        V_after = step['V_after']
        step_type = step['type']
        
        print(f"\nStep {i+1}: {V_before:.3f} → {V_after:.3f} V ({step_type})")
        
        # Extract segment for this step
        # From step index to either next step or step_duration
        if i < len(steps) - 1:
            idx_end = steps[i+1]['idx']
        else:
            # Last step: use time-based cutoff
            t_step_start = df[t_col].iloc[idx]
            idx_end = df[df[t_col] <= t_step_start + step_duration].index[-1] + 1
        
        df_seg = df.iloc[idx:idx_end].copy()
        
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
    """Plot fit parameters vs voltage, separated by rising/falling."""
    
    rising = results_df[results_df['type'] == 'RISING']
    falling = results_df[results_df['type'] == 'FALLING']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Helper for plotting with error bars
    def plot_param(ax, df_r, df_f, param, ylabel, use_V_final=False):
        x_col = 'V_final' if use_V_final else 'V_initial'
        
        # Handle special case for tau_ms
        err_col = 'tau_err_ms' if param == 'tau_ms' else f'{param}_err'
        
        if len(df_r) > 0:
            ax.errorbar(df_r[x_col], df_r[param], yerr=df_r[err_col], 
                       fmt='r^', markersize=8, capsize=3, label='Rising', alpha=0.8)
        if len(df_f) > 0:
            ax.errorbar(df_f[x_col], df_f[param], yerr=df_f[err_col], 
                       fmt='bv', markersize=8, capsize=3, label='Falling', alpha=0.8)
        
        ax.set_xlabel('V_final (V)' if use_V_final else 'V_initial (V)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # A vs V_final
    plot_param(axes[0,0], rising, falling, 'A', 'A (mA)', use_V_final=True)
    axes[0,0].set_title('Amplitude A vs Final Voltage')
    
    # τ vs V_final
    plot_param(axes[0,1], rising, falling, 'tau_ms', 'τ (ms)', use_V_final=True)
    axes[0,1].set_title('Time Constant τ vs Final Voltage')
    
    # B vs V_final
    plot_param(axes[0,2], rising, falling, 'B', 'B (mA·s^0.5)', use_V_final=True)
    axes[0,2].set_title('Diffusion B vs Final Voltage')
    
    # C vs V_final (should depend only on V_final!)
    plot_param(axes[1,0], rising, falling, 'C', 'C (mA)', use_V_final=True)
    axes[1,0].set_title('Steady-State C vs Final Voltage')
    
    # B vs dV (should be linear if Bazant interpretation correct)
    ax = axes[1,1]
    if len(rising) > 0:
        ax.errorbar(rising['dV'], rising['B'], yerr=rising['B_err'],
                   fmt='r^', markersize=8, capsize=3, label='Rising', alpha=0.8)
    if len(falling) > 0:
        ax.errorbar(falling['dV'], falling['B'], yerr=falling['B_err'],
                   fmt='bv', markersize=8, capsize=3, label='Falling', alpha=0.8)
    
    # Add linear fit
    all_dV = results_df['dV'].values
    all_B = results_df['B'].values
    if len(all_dV) > 2:
        coeffs = np.polyfit(all_dV, all_B, 1)
        dV_line = np.linspace(0, all_dV.max(), 100)
        ax.plot(dV_line, coeffs[0]*dV_line + coeffs[1], 'k--', 
               label=f'Linear: B = {coeffs[0]:.3f}·ΔV + {coeffs[1]:.3f}')
    
    ax.set_xlabel('|ΔV| (V)')
    ax.set_ylabel('B (mA·s^0.5)')
    ax.set_title('B vs ΔV (Linearity Test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R² vs step number
    ax = axes[1,2]
    ax.bar(results_df['step'], results_df['R2'], color=['red' if t=='RISING' else 'blue' 
                                                         for t in results_df['type']], alpha=0.7)
    ax.axhline(0.99, color='green', linestyle='--', label='R²=0.99')
    ax.set_xlabel('Step Number')
    ax.set_ylabel('R²')
    ax.set_title('Fit Quality')
    ax.set_ylim([min(0.95, results_df['R2'].min()-0.01), 1.0])
    ax.legend()
    
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
    # LOAD YOUR DATA HERE
    # =========================================================================
    
    # Option 1: From command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"Loading: {filename}")
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(filename)
        else:
            df = pd.read_csv(filename)
    
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
    # ANALYSIS PARAMETERS
    # =========================================================================
    
    # Column names (adjust if your data uses different names)
    T_COL = 'time/s'
    V_COL = 'Ewe/V'
    I_COL = 'I/mA'
    
    # Fitting parameters
    STEP_DURATION = 4.0    # Max duration to consider for each step (s)
    T_SKIP = 0.002         # Time to skip at start of step (s) - avoid t=0 singularity
    T_FIT_MAX = None       # Max time to include in fit (None = use all data)
    
    # Output prefix
    OUTPUT_PREFIX = 'voltage_step_fits'
    
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
        t_fit_max=T_FIT_MAX
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
        print(f"  B/ΔV mean: {(rising['B']/rising['dV']).mean():.3f} mA·s^0.5/V")
    
    print(f"\nFALLING steps: {len(falling)}")
    if len(falling) > 0:
        print(f"  τ mean: {falling['tau_ms'].mean():.1f} ± {falling['tau_ms'].std():.1f} ms")
        print(f"  B/ΔV mean: {(falling['B']/falling['dV']).mean():.3f} mA·s^0.5/V")
    
    # Linearity test
    if len(results_df) > 2:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            results_df['dV'], results_df['B']
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
