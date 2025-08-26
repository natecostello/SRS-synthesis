# SRS Synthesis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for Shock Response Spectrum (SRS) synthesis - generating acceleration time histories that match target shock response spectra.  This package is a port with optimizations of Matlab code written by Tom Irvine, (tom@vibrationdata.com).

## Installation

### From GitHub (Recommended)
```bash
pip install git+https://github.com/natecostello/SRS-synthesis.git
```

### Development Installation
```bash
git clone https://github.com/natecostello/SRS-synthesis.git
cd SRS-synthesis
pip install -e .
```

## Quick Start

```python
import numpy as np
from srs_synthesis import DSSynthesizer, WSynthesizer

# Define target SRS specification
freq_spec = np.array([20, 2000, 10000])    # Hz
accel_spec = np.array([20, 2000, 2000])    # G

# Damped Sine Synthesis (DSS) - for shock analysis
synthesizer = DSSynthesizer()
result = synthesizer.synthesize_srs(
    freq_spec=freq_spec,
    accel_spec=accel_spec,
    duration=0.15,                          # seconds
    fast_mode=True                          # 44x speed improvement
)

# Wavelet Synthesis (WSS) - for shaker testing  
wss = WSynthesizer()
result = wss.synthesize_wavelet_srs(
    freq_spec=freq_spec,
    accel_spec=accel_spec,
    duration=0.15,
    strategy='reverse_sine_sweep'
)

# Access results
time = result['time']
acceleration = result['acceleration']      # G
velocity = result['velocity']             # in/sec or m/sec  
displacement = result['displacement']      # inches or mm
```

## Methods

### Damped Sine Synthesis (DSS)
- **Purpose**: Generate realistic shock-like transients for component analysis
- **Approach**: Combines multiple damped sinusoids with iterative optimization
- **Best for**: Component response analysis

### Wavelet Synthesis (WSS)  
- **Purpose**: Generate optimal waveforms for shaker testing within equipment constraints
- **Approach**: Multi-criteria optimization to balance accuracy against physical limits
- **Best for**: Laboratory testing where shaker limits matter

## Performance

The `examples/` directory contains complete working examples:

```bash
# After installation, download examples separately
git clone https://github.com/natecostello/SRS-synthesis.git
cd SRS-synthesis/examples

# Run Tom Irvine's test cases
python example_DSS_run_tom_irvine_case.py
python example_WS_run_tom_irvine_case.py

# Analyze results
python analyze_DSS_run.py results.pkl
python analyze_WS_run.py results.pkl
```

## Package Structure

```
srs_synthesis/
├── __init__.py           # Public API exports
├── damped_sine.py        # DSS synthesis (DSSynthesizer)
└── wavelet.py           # Wavelet synthesis (WSynthesizer)  
```

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0 (for examples and plotting)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- **Original MATLAB Implementation**: Tom Irvine, Vibration Data (tom@vibrationdata.com)
- **SRS Algorithm**: Smallwood, D.O. "An Improved Recursive Formula for Calculating Shock Response Spectra" (1981)
- **Python Implementation**: Ported and optimized by Claude Sonnet 4 (2025)
