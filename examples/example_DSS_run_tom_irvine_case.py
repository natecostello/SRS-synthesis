#!/usr/bin/env python3

import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srs_synthesis.damped_sine import DSSynthesizer

def run_tom_irvine_case():
    """
    Run Tom Irvine's challenging SRS synthesis case and save results to file.
    
    This is a demanding test case with wide frequency range and high G levels.
    Tom Irvine reported in this YouTube video that this case takes him about 
    20 minutes to run in MATLAB: https://www.youtube.com/watch?v=nJwQdlcB42s&t=110s
    
    Our optimized Python implementation typically completes this in under 4 minutes.
    """
    
    print("Tom Irvine's Challenging SRS Case - Synthesis Only")
    print("=" * 50)
    
    # EXACT specification: srs_spec=[20 20; 2000 2000; 10000 2000]
    freq_spec = np.array([20.0, 2000.0, 10000.0])
    accel_spec = np.array([20.0, 2000.0, 2000.0])
    
    # EXACT parameters as specified
    duration = 0.15                   # duration 0.15
    max_iterations = 300             # number of trials 300
    wavelet_reconstruction = True     # use waveform reconstruction
    wavelet_trials = 5000            # trials per frequency = 5000
    wavelet_frequencies = 500        # number of frequencies = 500
    
    print(f"EXACT Parameters:")
    print(f"  SRS spec: [20 20; 2000 2000; 10000 2000]")
    print(f"  Duration: {duration}")
    print(f"  Number of trials: {max_iterations}")
    print(f"  Use waveform reconstruction: {wavelet_reconstruction}")
    print(f"  Trials per frequency: {wavelet_trials}")
    print(f"  Number of frequencies: {wavelet_frequencies}")
    
    print(f"\nRunning synthesis... (this may take several minutes)")
    
    # Create synthesizer and run synthesis
    synthesizer = DSSynthesizer()
    result = synthesizer.synthesize_srs(
        freq_spec=freq_spec,
        accel_spec=accel_spec,
        duration=duration,
        max_iterations=max_iterations,
        wavelet_reconstruction=wavelet_reconstruction,
        wavelet_trials=wavelet_trials,
        wavelet_frequencies=wavelet_frequencies,
        fast_mode=True
    )
    
    # Add specification to results for analysis
    result['freq_spec'] = freq_spec
    result['accel_spec'] = accel_spec
    result['parameters'] = {
        'duration': duration,
        'max_iterations': max_iterations,
        'wavelet_reconstruction': wavelet_reconstruction,
        'wavelet_trials': wavelet_trials,
        'wavelet_frequencies': wavelet_frequencies
    }
    
    # Save results to file
    output_file = 'tom_irvine_case_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\nSynthesis Complete!")
    print(f"Results saved to: {output_file}")
    print(f"Synthesis error: {result['synthesis_error']:.3f} dB")
    print(f"Peak acceleration: {np.max(np.abs(result['acceleration'])):.1f} G")
    print(f"Final velocity: {result['velocity'][-1]:.6f} in/sec")
    print(f"Final displacement: {result['displacement'][-1]:.6f} inches")
    print(f"\nRun 'python analyze_user_case.py' to generate plots and detailed analysis.")
    
    return result

if __name__ == "__main__":
    run_tom_irvine_case()
