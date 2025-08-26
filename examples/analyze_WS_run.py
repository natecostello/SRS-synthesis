#!/usr/bin/env python3
"""
Analysis script for Wavelet Synthesis (WS) results.
Takes a pickled result file and generates comprehensive analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_ws_run(results_file):
    """
    Analyze WS synthesis results from a pickled file.
    
    Parameters:
    -----------
    results_file : str
        Path to the pickled results file
    """
    
    print(f"Loading and analyzing Wavelet Synthesis results from: {results_file}")
    print("=" * 70)
    
    # Load results
    try:
        with open(results_file, 'rb') as f:
            result = pickle.load(f)
    except FileNotFoundError:
        print(f"Results file '{results_file}' not found!")
        print("Run 'python example_WS_run_tom_irvine_case.py' first to generate results.")
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
    
    # Extract specification and parameters
    freq_spec = result.get('freq_spec', [])
    accel_spec = result.get('accel_spec', [])
    parameters = result.get('parameters', {})
    
    # Print summary
    print("Synthesis Summary:")
    print(f"  Synthesis Error: {synthesis_error:.3f} dB")
    print(f"  Peak Acceleration: {np.max(np.abs(acceleration)):.1f} G")
    print(f"  Final Velocity: {velocity[-1]:.6f} in/sec")
    print(f"  Final Displacement: {displacement[-1]:.6f} inches")
    print(f"  Duration: {time[-1]:.3f} seconds")
    print(f"  Sample Rate: {1.0/(time[1]-time[0]):.1f} Hz")
    
    if parameters:
        print(f"\nSynthesis Parameters:")
        print(f"  Strategy: {parameters.get('strategy', 'Unknown')}")
        print(f"  Q Factor: {parameters.get('Q_factor', 'Unknown')}")
        print(f"  Damping Ratio: {parameters.get('damping_ratio', 'Unknown')}")
        print(f"  Octave Spacing: {parameters.get('octave_spacing', 'Unknown')}")
        print(f"  Number of Trials: {parameters.get('ntrials', 'Unknown')}")
    
    # Wavelet-specific information
    if 'wavelet_table' in result:
        wavelet_table = result['wavelet_table']
        print(f"\nWavelet Information:")
        print(f"  Number of wavelets: {len(wavelet_table)}")
        print(f"  Frequency range: {np.min(wavelet_table[:, 2]):.1f} - {np.max(wavelet_table[:, 2]):.1f} Hz")
        
    if 'ranking_metrics' in result:
        metrics = result['ranking_metrics']
        print(f"\nWinning Solution Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
    
    # Calculate additional statistics
    print(f"\nSignal Statistics:")
    print(f"  RMS Acceleration: {np.sqrt(np.mean(acceleration**2)):.2f} G")
    print(f"  Crest Factor: {np.max(np.abs(acceleration))/np.sqrt(np.mean(acceleration**2)):.2f}")
    print(f"  Velocity Range: {np.min(velocity):.3f} to {np.max(velocity):.3f} in/sec")
    print(f"  Displacement Range: {np.min(displacement):.4f} to {np.max(displacement):.4f} inches")
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'Wavelet Synthesis Analysis - Error: {synthesis_error:.2f} dB', fontsize=14, fontweight='bold')
    
    # 1. Time history plots
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(time, acceleration, 'b-', linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (sec)')
    plt.ylabel('Acceleration (G)')
    plt.title('Acceleration Time History')
    
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(time, velocity, 'g-', linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (in/sec)')
    plt.title('Velocity Time History')
    
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(time, displacement, 'r-', linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (sec)')
    plt.ylabel('Displacement (in)')
    plt.title('Displacement Time History')
    
    # 2. SRS Comparison
    ax4 = plt.subplot(3, 3, 4)
    plt.loglog(srs_freq, srs_pos, 'b-', linewidth=2, label='Synthesized SRS+')
    plt.loglog(srs_freq, srs_neg, 'b--', linewidth=2, label='Synthesized SRS-')
    if len(freq_spec) > 0 and len(accel_spec) > 0:
        plt.loglog(freq_spec, accel_spec, 'ro-', linewidth=2, markersize=8, label='Target Spec')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Acceleration (G)')
    plt.title('SRS Comparison')
    plt.legend()
    
    # 3. SRS Error vs Frequency
    if len(freq_spec) > 0 and len(accel_spec) > 0:
        ax5 = plt.subplot(3, 3, 5)
        # Interpolate synthesized SRS at specification frequencies
        srs_interp_pos = np.interp(freq_spec, srs_freq, srs_pos)
        srs_interp_neg = np.interp(freq_spec, srs_freq, srs_neg)
        srs_envelope = np.maximum(srs_interp_pos, srs_interp_neg)
        
        error_db = 20 * np.log10(srs_envelope / accel_spec)
        plt.semilogx(freq_spec, error_db, 'ro-', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Error (dB)')
        plt.title('SRS Error at Spec Points')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. FFT of acceleration
    ax6 = plt.subplot(3, 3, 6)
    dt = time[1] - time[0]
    fft_freq = np.fft.rfftfreq(len(acceleration), dt)
    fft_mag = np.abs(np.fft.rfft(acceleration))
    plt.loglog(fft_freq[1:], fft_mag[1:], 'b-', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Acceleration FFT')
    
    # 5. Wavelet distribution (if available)
    if 'wavelet_table' in result:
        ax7 = plt.subplot(3, 3, 7)
        wavelet_table = result['wavelet_table']
        frequencies = wavelet_table[:, 2]  # frequency column
        amplitudes = wavelet_table[:, 1]   # amplitude column
        plt.scatter(frequencies, np.abs(amplitudes), alpha=0.6, c='red', s=20)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Wavelet Distribution')
        plt.grid(True, alpha=0.3)
        if len(frequencies) > 0:
            plt.xlim(left=min(frequencies)*0.5)
    
    # 6. Histogram of acceleration values
    ax8 = plt.subplot(3, 3, 8)
    plt.hist(acceleration, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Acceleration (G)')
    plt.ylabel('Count')
    plt.title('Acceleration Distribution')
    plt.grid(True, alpha=0.3)
    
    # 7. Cumulative velocity and displacement
    ax9 = plt.subplot(3, 3, 9)
    plt.plot(time, np.cumsum(np.abs(velocity)) * (time[1]-time[0]), 'g-', linewidth=2, label='Cum. |Velocity|')
    plt.plot(time, np.cumsum(np.abs(displacement)) * (time[1]-time[0]), 'r-', linewidth=2, label='Cum. |Displacement|')
    plt.xlabel('Time (sec)')
    plt.ylabel('Cumulative Value')
    plt.title('Cumulative Integrals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = results_file.replace('.pkl', '_analysis.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    # Generate detailed report
    generate_detailed_report(result, results_file)
    
    return result

def generate_detailed_report(result, results_file):
    """Generate a detailed text report of the synthesis results."""
    
    report_filename = results_file.replace('.pkl', '_report.txt')
    
    with open(report_filename, 'w') as f:
        f.write("WAVELET SYNTHESIS DETAILED REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic information
        f.write("SYNTHESIS RESULTS:\n")
        f.write(f"  Synthesis Error: {result['synthesis_error']:.3f} dB\n")
        f.write(f"  Peak Acceleration: {np.max(np.abs(result['acceleration'])):.2f} G\n")
        f.write(f"  Final Velocity: {result['velocity'][-1]:.6f} in/sec\n")
        f.write(f"  Final Displacement: {result['displacement'][-1]:.6f} inches\n")
        f.write(f"  Signal Duration: {result['time'][-1]:.3f} seconds\n")
        f.write(f"  Sample Rate: {1.0/(result['time'][1]-result['time'][0]):.1f} Hz\n\n")
        
        # Parameters
        if 'parameters' in result:
            params = result['parameters']
            f.write("SYNTHESIS PARAMETERS:\n")
            for key, value in params.items():
                if key != 'weights':
                    f.write(f"  {key}: {value}\n")
                else:
                    f.write("  Optimization weights:\n")
                    for wkey, wval in value.items():
                        f.write(f"    {wkey}: {wval}\n")
            f.write("\n")
        
        # Wavelet information
        if 'wavelet_table' in result:
            wavelet_table = result['wavelet_table']
            f.write("WAVELET INFORMATION:\n")
            f.write(f"  Number of wavelets: {len(wavelet_table)}\n")
            f.write(f"  Frequency range: {np.min(wavelet_table[:, 2]):.1f} - {np.max(wavelet_table[:, 2]):.1f} Hz\n")
            f.write("  Individual wavelets:\n")
            f.write("    Index  Amplitude   Frequency   Half-cycles  Delay\n")
            for i, row in enumerate(wavelet_table[:10]):  # First 10 wavelets
                f.write(f"    {i:5.0f}  {row[1]:8.3f}   {row[2]:8.1f}   {row[3]:8.0f}     {row[4]:6.4f}\n")
            if len(wavelet_table) > 10:
                f.write(f"    ... and {len(wavelet_table)-10} more wavelets\n")
            f.write("\n")
        
        # Statistics
        acc = result['acceleration']
        vel = result['velocity']
        disp = result['displacement']
        
        f.write("SIGNAL STATISTICS:\n")
        f.write(f"  RMS Acceleration: {np.sqrt(np.mean(acc**2)):.3f} G\n")
        f.write(f"  Crest Factor: {np.max(np.abs(acc))/np.sqrt(np.mean(acc**2)):.3f}\n")
        f.write(f"  Velocity range: {np.min(vel):.4f} to {np.max(vel):.4f} in/sec\n")
        f.write(f"  Displacement range: {np.min(disp):.6f} to {np.max(disp):.6f} inches\n")
        f.write(f"  Mean acceleration: {np.mean(acc):.6f} G\n")
        f.write(f"  Standard deviation: {np.std(acc):.3f} G\n")
        
        # Ranking metrics
        if 'ranking_metrics' in result:
            f.write("\nWINNING SOLUTION METRICS:\n")
            for key, value in result['ranking_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.6f}\n")
    
    print(f"Detailed report saved as: {report_filename}")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Analyze Wavelet Synthesis results')
    parser.add_argument('results_file', nargs='?', default='tom_irvine_ws_case_results.pkl',
                       help='Path to results pickle file (default: tom_irvine_ws_case_results.pkl)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Results file '{args.results_file}' not found!")
        print("Available result files in current directory:")
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            for f in pkl_files:
                print(f"  {f}")
        else:
            print("  No .pkl files found")
        return
    
    analyze_ws_run(args.results_file)

if __name__ == "__main__":
    main()
