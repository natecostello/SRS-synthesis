# Shock Response Spectrum (SRS) Synthesis

Python implementations of SRS synthesis algorithms for generating acceleration time histories that match target shock response spectra.

## Overview

This repository provides two SRS synthesis methods:

- **Damped Sine Synthesis (DSS)**: Classical method producing shock-like transient signals suitable for analysis
- **Wavelet Synthesis (WSS)**: Optimizes time series for shaker test limitations and constraints

Both methods are based on Tom Irvine's MATLAB vibrationdata algorithms.

## Methods

### Damped Sine Synthesis
- Uses multiple damped sinusoids to build shock-like transients
- Produces realistic shock waveforms for analysis purposes
- Good for understanding shock physics and component response

### Wavelet Synthesis  
- Uses wavelets with multi-criteria optimization
- Optimizes for shaker limitations (displacement, velocity, acceleration constraints)
- Ranking system balances spectral accuracy against physical test constraints
- Better for actual shaker testing where equipment limits matter

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0  
- Matplotlib >= 3.3.0

```bash
pip install -r requirements.txt
```

### Repository Structure

```
├── srs_damped_sine_synthesis.py       # Damped sine synthesis
├── srs_wavelet_synthesis.py          # Wavelet synthesis
├── srs_conversion.py                 # SRS unit conversion
├── test_dss.py                       # DSS tests
├── test_ws.py                        # WSS tests  
├── examples/                         # Example cases
│   ├── example_DSS_run_tom_irvine_case.py        
│   ├── example_WS_run_tom_irvine_case.py         
│   ├── example_WS_run_min_accel_case.py          
│   ├── analyze_DSS_run.py                        
│   └── analyze_WS_run.py                         
└── requirements.txt                  
```

## References

- **Original MATLAB**: Tom Irvine, Vibration Data (tom@vibrationdata.com)
- **SRS Algorithm**: Smallwood, D.O. "An Improved Recursive Formula for Calculating Shock Response Spectra" (1981)
