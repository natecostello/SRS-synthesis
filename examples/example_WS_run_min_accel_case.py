#!/usr/bin/env python3

import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srs_synthesis.wavelet import WSynthesizer

def run_min_accel_case():
    """
    Run a minimum acceleration wavelet synthesis case and save results to file.
    
    This demonstrates wavelet synthesis with the specified SRS requirements:
    - SRS spec: [10 9.4; 80 75; 150 200; 300 200; 2000 75]
    - Octave spacing: 1/12
    - Duration: 0.24 seconds
    - Strategy: Reverse sine sweep
    - Q: 10 (damping ratio = 0.05)
    - All optimization weights: 1.0
    - Infinite retries: True (MATLAB-equivalent behavior)
    """
    
    print("Minimum Acceleration Wavelet Synthesis Case")
    print("=" * 40)

    # SRS specification: srs_spec=[10 9.4; 80 75; 150 200; 300 200; 2000 75]
    freq_spec = np.array([10.0, 80.0, 150.0, 300.0, 2000.0])
    accel_spec = np.array([9.4, 75.0, 200.0, 200.0, 100.0])

    # Parameters as specified
    duration = 0.24                   # duration 0.24 seconds
    sample_rate = 20000               # sample rate 20000 Hz
    octave_spacing = 3                # 1/12 octave spacing
    strategy = 3                      # reverse sine sweep
    damping_ratio = 0.05              # Q=10 -> damping = 0.5/Q = 0.05
    ntrials = 300                      # reduced for quick testing

    # All optimization weights = 1.0 (default values)
    weights = {
        'iw': 0.0,      # SRS error weight
        'ew': 0.0,      # Total error weight  
        'dw': 0.0,      # Displacement weight
        'vw': 0.0,      # Velocity weight
        'aw': 1.0,      # Acceleration weight
        'cw': 0.0,      # Crest factor weight
        'kw': 0.0,      # Kurtosis weight
        'dskw': 0.0     # Displacement skewness weight
    }
    
    print(f"Parameters:")
    print(f"  SRS spec: [10 9.4; 80 75; 2000 75]")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Octave spacing: 1/12")
    print(f"  Strategy: {strategy} (Reverse sine sweep)")
    print(f"  Q factor: 10 (damping ratio = {damping_ratio})")
    print(f"  Number of trials: {ntrials}")
    print(f"  All optimization weights: 1.0")
    print(f"  Infinite retries: True (MATLAB-equivalent)")
    
    print(f"\nRunning wavelet synthesis...")
    
    # Create synthesizer and run
    synthesizer = WSynthesizer()
    
    result = synthesizer.synthesize_srs(
        freq_spec=freq_spec,
        accel_spec=accel_spec,
        duration=duration,
        sample_rate=sample_rate,
        damping_ratio=damping_ratio,
        ntrials=ntrials,
        octave_spacing=octave_spacing,
        strategy=strategy,
        units='english',
        random_seed=42,  # For reproducibility
        allow_infinite_retries=True,  # Enable MATLAB-equivalent infinite retry behavior
        **weights  # Unpack all weight parameters
    )
    
    # Add specification to results for analysis
    result['freq_spec'] = freq_spec
    result['accel_spec'] = accel_spec
    result['parameters'] = {
        'duration': duration,
        'sample_rate': sample_rate,
        'octave_spacing': octave_spacing,
        'strategy': strategy,
        'damping_ratio': damping_ratio,
        'Q_factor': 10,
        'ntrials': ntrials,
        'weights': weights
    }
    
    # Save results to file
    output_file = 'min_accel_case_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\nWavelet Synthesis Complete!")
    print(f"Results saved to: {output_file}")
    print(f"Synthesis error: {result['synthesis_error']:.3f} dB")
    print(f"Peak acceleration: {np.max(np.abs(result['acceleration'])):.1f} G")
    
    # Check final conditions (should be near zero for wavelets)
    if 'velocity' in result:
        final_velocity = result['velocity'][-1] if hasattr(result['velocity'], '__getitem__') else 0
        print(f"Final velocity: {final_velocity:.6f} in/sec")
    
    if 'displacement' in result:
        final_displacement = result['displacement'][-1] if hasattr(result['displacement'], '__getitem__') else 0
        print(f"Final displacement: {final_displacement:.6f} inches")
    
    # Display wavelet synthesis specific results
    if 'wavelet_table' in result:
        print(f"Number of wavelets used: {len(result['wavelet_table'])}")
    
    if 'ranking_metrics' in result:
        metrics = result['ranking_metrics']
        print(f"Winning solution metrics:")
        print(f"  Crest factor: {metrics.get('crest_factor', 'N/A'):.3f}")
        print(f"  Kurtosis: {metrics.get('kurtosis', 'N/A'):.3f}")
    
    print(f"\nRun 'python analyze_WS_run.py {output_file}' to generate plots and detailed analysis.")
    
    return result

if __name__ == "__main__":
    run_min_accel_case()
