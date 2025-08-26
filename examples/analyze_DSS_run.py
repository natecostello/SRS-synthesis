#!/usr/bin/env python3
"""
Generalized analysis script for Damped Sine Synthesis (DSS) results.
Takes a pickled result file and generates comprehensive analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srs_conversion import convert_srs

def analyze_dss_run(results_file):
    """
    Analyze DSS synthesis results from a pickled file.
    
    Parameters:
    -----------
    results_file : str
        Path to the pickled results file
    """
    
    print(f"Loading and analyzing DSS results from: {results_file}")
    print("=" * 60)
    
    # Load results
    try:
        with open(results_file, 'rb') as f:
            result = pickle.load(f)
    except FileNotFoundError:
        print(f"Results file '{results_file}' not found!")
        return None
    except Exception as e:
        print(f"Error loading results file: {e}")
        return None
    
    # Extract data
    time = result['time']
    acceleration = result['acceleration']
    velocity = result['velocity'] 
    displacement = result['displacement']
    synthesis_error = result['synthesis_error']
    srs_freq = result['srs_freq']
    srs_pos = result['srs_pos']
    srs_neg = result['srs_neg']
    
    # Get original specification if available
    if 'freq_spec' in result and 'accel_spec' in result:
        freq_spec = result['freq_spec']
        accel_spec = result['accel_spec']
        has_spec = True
    else:
        has_spec = False
    
    # Analysis Results
    peak_accel = np.max(np.abs(acceleration))
    rms_accel = np.sqrt(np.mean(acceleration**2))
    duration = time[-1] - time[0]
    sample_rate = len(time) / duration
    
    print("Analysis Results:")
    print(f"  Synthesis error: {synthesis_error:.3f} dB")
    print(f"  Peak acceleration: {peak_accel:.1f} G")
    print(f"  RMS acceleration: {rms_accel:.1f} G") 
    print(f"  Duration: {duration:.6f} seconds")
    print(f"  Sample rate: {sample_rate:.1f} Hz")
    print(f"  Number of samples: {len(time)}")
    
    # Baseline control analysis
    print(f"\\nBaseline Control:")
    print(f"  Final velocity: {velocity[-1]:.6f} in/sec")
    print(f"  Final displacement: {displacement[-1]:.6f} inches")
    print(f"  Max velocity: {np.max(np.abs(velocity)):.3f} in/sec")
    print(f"  Max displacement: {np.max(np.abs(displacement)):.3f} inches")
    
    # Parameters used (if available)
    params_to_show = ['duration', 'max_iterations', 'wavelet_reconstruction', 
                     'wavelet_trials', 'wavelet_frequencies']
    print(f"\\nParameters Used:")
    for param in params_to_show:
        if param in result:
            print(f"  {param}: {result[param]}")
    
    # Accuracy at specification points (if available)
    if has_spec:
        print(f"\\nAccuracy at Specification Points:")
        for i, (f, target) in enumerate(zip(freq_spec, accel_spec)):
            # Find closest frequency
            idx = np.argmin(np.abs(srs_freq - f))
            actual = srs_pos[idx]
            error_db = 20 * np.log10(actual / target)
            error_pct = (actual - target) / target * 100
            print(f"  {f:6.0f} Hz: Target={target:6.0f}G, Actual={actual:6.1f}G, "
                  f"Error={error_db:+5.2f}dB ({error_pct:+5.1f}%)")
    
    # Create comprehensive analysis plots
    create_analysis_plots(result, results_file)
    
    # Create multi-unit SRS comparison if specification available
    if has_spec:
        create_srs_unit_comparison_plots(result, results_file)
    
    return result

def create_analysis_plots(result, results_file):
    """Create comprehensive time history and SRS analysis plots."""
    
    time = result['time']
    acceleration = result['acceleration']
    velocity = result['velocity']
    displacement = result['displacement']
    srs_freq = result['srs_freq']
    srs_pos = result['srs_pos']
    srs_neg = result['srs_neg']
    
    # Get specification if available
    has_spec = 'freq_spec' in result and 'accel_spec' in result
    if has_spec:
        freq_spec = result['freq_spec']
        accel_spec = result['accel_spec']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'DSS Analysis Results: {os.path.basename(results_file)}', 
                 fontsize=16, fontweight='bold')
    
    # Time history plots
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(time, acceleration, 'b-', linewidth=1)
    plt.xlabel('Time (sec)')
    plt.ylabel('Acceleration (G)')
    plt.title('Acceleration Time History')
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(time, velocity, 'g-', linewidth=1)
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (in/sec)')
    plt.title('Velocity Time History')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(time, displacement, 'r-', linewidth=1)
    plt.xlabel('Time (sec)')
    plt.ylabel('Displacement (inches)')
    plt.title('Displacement Time History')
    plt.grid(True, alpha=0.3)
    
    # SRS plot
    ax4 = plt.subplot(2, 3, (4, 6))
    plt.loglog(srs_freq, srs_pos, 'b-', linewidth=2, label='Positive SRS')
    plt.loglog(srs_freq, srs_neg, 'b--', linewidth=2, label='Negative SRS')
    
    if has_spec:
        plt.loglog(freq_spec, accel_spec, 'ro-', linewidth=3, markersize=8, 
                  label='Target Specification', zorder=10)
        
        # Add 3dB tolerance bands
        accel_spec_interp = np.interp(srs_freq, freq_spec, accel_spec)
        plt.loglog(srs_freq, accel_spec_interp * 10**(3/20), 'r:', alpha=0.7, 
                  linewidth=1, label='+3dB tolerance')
        plt.loglog(srs_freq, accel_spec_interp * 10**(-3/20), 'r:', alpha=0.7, 
                  linewidth=1, label='-3dB tolerance')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SRS Acceleration (G)')
    plt.title('Shock Response Spectrum')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([10, 20000])
    
    plt.tight_layout()
    
    # Save plots
    base_name = os.path.splitext(results_file)[0]
    plot_filename = f'{base_name}_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\\nAnalysis plots saved as: {plot_filename}")
    
    plt.show()

def create_srs_unit_comparison_plots(result, results_file):
    """Create multi-unit SRS comparison plots."""
    
    # Extract data
    srs_frequencies = result['srs_freq']
    srs_positive = result['srs_pos']
    srs_negative = result['srs_neg']
    freq_spec = result['freq_spec']
    accel_spec = result['accel_spec']
    
    # Convert specifications to velocity and displacement
    vel_spec = convert_srs(accel_spec, freq_spec, "acceleration", "velocity", "g", "in/sec")
    disp_spec = convert_srs(accel_spec, freq_spec, "acceleration", "displacement", "g", "inches")
    
    # Convert synthesized SRS to velocity and displacement
    srs_vel_pos = convert_srs(srs_positive, srs_frequencies, "acceleration", "velocity", "g", "in/sec")
    srs_vel_neg = convert_srs(srs_negative, srs_frequencies, "acceleration", "velocity", "g", "in/sec")
    srs_disp_pos = convert_srs(srs_positive, srs_frequencies, "acceleration", "displacement", "g", "inches")
    srs_disp_neg = convert_srs(srs_negative, srs_frequencies, "acceleration", "displacement", "g", "inches")
    
    # Interpolate specs to SRS frequencies for tolerance bands
    accel_spec_interp = np.interp(srs_frequencies, freq_spec, accel_spec)
    vel_spec_interp = convert_srs(accel_spec_interp, srs_frequencies, "acceleration", "velocity", "g", "in/sec")
    disp_spec_interp = convert_srs(accel_spec_interp, srs_frequencies, "acceleration", "displacement", "g", "inches")
    
    # Create comprehensive SRS comparison figure
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'Multi-Unit SRS Comparison: {os.path.basename(results_file)}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Acceleration SRS
    ax1 = plt.subplot(2, 3, 1)
    plt.loglog(srs_frequencies, accel_spec_interp * 10**(3/20), 'r:', alpha=0.7, linewidth=1, label='±3dB tolerance')
    plt.loglog(srs_frequencies, accel_spec_interp * 10**(-3/20), 'r:', alpha=0.7, linewidth=1)
    plt.loglog(freq_spec, accel_spec, 'ro-', linewidth=3, markersize=8, label='Target Spec', zorder=10)
    plt.loglog(srs_frequencies, srs_positive, 'b-', linewidth=2, label='Synthesized (+)')
    plt.loglog(srs_frequencies, srs_negative, 'b--', linewidth=2, label='Synthesized (-)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Acceleration (G)')
    plt.title('Acceleration SRS')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim([10, 20000])
    
    # Plot 2: Velocity SRS  
    ax2 = plt.subplot(2, 3, 2)
    plt.loglog(srs_frequencies, vel_spec_interp * 10**(3/20), 'r:', alpha=0.7, linewidth=1, label='±3dB tolerance')
    plt.loglog(srs_frequencies, vel_spec_interp * 10**(-3/20), 'r:', alpha=0.7, linewidth=1)
    plt.loglog(freq_spec, vel_spec, 'ro-', linewidth=3, markersize=8, label='Target Spec', zorder=10)
    plt.loglog(srs_frequencies, srs_vel_pos, 'g-', linewidth=2, label='Synthesized (+)')
    plt.loglog(srs_frequencies, srs_vel_neg, 'g--', linewidth=2, label='Synthesized (-)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Velocity (in/sec)')
    plt.title('Velocity SRS')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim([10, 20000])
    
    # Plot 3: Displacement SRS
    ax3 = plt.subplot(2, 3, 3)
    plt.loglog(srs_frequencies, disp_spec_interp * 10**(3/20), 'r:', alpha=0.7, linewidth=1, label='±3dB tolerance')
    plt.loglog(srs_frequencies, disp_spec_interp * 10**(-3/20), 'r:', alpha=0.7, linewidth=1)
    plt.loglog(freq_spec, disp_spec, 'ro-', linewidth=3, markersize=8, label='Target Spec', zorder=10)
    plt.loglog(srs_frequencies, srs_disp_pos, 'm-', linewidth=2, label='Synthesized (+)')
    plt.loglog(srs_frequencies, srs_disp_neg, 'm--', linewidth=2, label='Synthesized (-)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Displacement (inches)')
    plt.title('Displacement SRS')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim([10, 20000])
    
    # Error analysis plots
    error_pos = 20 * np.log10(srs_positive / accel_spec_interp)
    vel_error_pos = 20 * np.log10(srs_vel_pos / vel_spec_interp)  
    disp_error_pos = 20 * np.log10(srs_disp_pos / disp_spec_interp)
    
    for i, (errors, title, color) in enumerate([
        (error_pos, 'Acceleration SRS Error', 'b'),
        (vel_error_pos, 'Velocity SRS Error', 'g'),
        (disp_error_pos, 'Displacement SRS Error', 'm')
    ]):
        ax = plt.subplot(2, 3, 4 + i)
        plt.semilogx(srs_frequencies, errors, f'{color}-', linewidth=2, label='Error')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        plt.axhline(y=3, color='r', linestyle=':', alpha=0.7, label='±3dB tolerance')
        plt.axhline(y=-3, color='r', linestyle=':', alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Error (dB)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.ylim([-6, 6])
        plt.xlim([10, 20000])
    
    plt.tight_layout()
    
    # Save the comprehensive SRS plots
    base_name = os.path.splitext(results_file)[0]
    srs_plot_filename = f'{base_name}_multi_unit_srs.png'
    plt.savefig(srs_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Multi-unit SRS comparison plots saved as: {srs_plot_filename}")
    
    plt.show()

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze Damped Sine Synthesis (DSS) results from pickled file'
    )
    parser.add_argument('results_file', 
                       help='Path to pickled results file (*.pkl)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found!")
        sys.exit(1)
    
    result = analyze_dss_run(args.results_file)
    
    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
