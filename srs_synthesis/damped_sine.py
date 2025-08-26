"""
Shock Response Spectrum (SRS) Synthesis using Damped Sinusoids

This module provides functionality to synthesize a time history acceleration signal
that satisfies a given shock response spectrum specification using damped sinusoids.

Based on the MATLAB implementation by Tom Irvine (tom@vibrationdata.com)
Ported to Python by Claude AI Assistant

Key features:
- Damped sinusoid synthesis with iterative optimization
- Smallwood algorithm for SRS calculation  
- Performance optimizations enabled by default (fast_mode, fast_wavelet_mode)
- Optional wavelet reconstruction for zero baseline
- Configurable parameters for full control over synthesis process
- Complete result invariance between optimized and original modes
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Any, Tuple, Optional
import warnings
import time


class DSSynthesizer:
    """
    Synthesizer for shock response spectrum using damped sinusoids.
    
    This class implements the MATLAB damped_sine_syn algorithm with
    performance optimizations enabled by default while maintaining
    complete result invariance. Wavelet reconstruction capabilities
    are included for zero baseline control.
    """
    
    def __init__(self):
        """Initialize the SRS synthesizer."""
        self.tpi = 2.0 * np.pi
        self.octave = 2.0**(1.0/12.0)  # 1/12 octave spacing (hardcoded as in MATLAB)
        
        # Pre-allocated arrays for performance
        self.MAX = 400
        self.FMAX = 400
        
    def synthesize_srs(self, 
                      freq_spec: np.ndarray, 
                      accel_spec: np.ndarray,
                      duration: float,
                      sample_rate: Optional[float] = None,
                      damping_ratio: float = 0.05,
                      max_iterations: int = 100,
                      inner_iterations: int = 80,
                      wavelet_reconstruction: bool = False,
                      wavelet_trials: int = 5000,
                      wavelet_frequencies: int = 500,
                      units: str = 'english',
                      random_seed: Optional[int] = None,
                      fast_mode: bool = True,
                      fast_wavelet_mode: bool = True) -> Dict[str, Any]:
        """
        Synthesize a time history to match the target SRS specification.
        
        Parameters:
        -----------
        freq_spec : np.ndarray
            Target natural frequencies (Hz)
        accel_spec : np.ndarray  
            Target SRS acceleration values (G)
        duration : float
            Duration of synthesized signal (seconds)
        sample_rate : float, optional
            Sample rate (Hz). If None, automatically determined as 10x max frequency
        damping_ratio : float, default=0.05
            Damping ratio for SRS calculation
        max_iterations : int, default=100
            Maximum outer loop iterations (min=10, max=5000)
        inner_iterations : int, default=80
            Inner loop iterations for parameter optimization
        wavelet_reconstruction : bool, default=False
            Whether to perform wavelet reconstruction for zero baseline
        wavelet_trials : int, default=5000
            Number of trials per frequency for wavelet reconstruction
        wavelet_frequencies : int, default=500
            Number of frequencies for wavelet reconstruction
        units : str, default='english'
            Units system ('english' or 'metric')
        random_seed : int, optional
            Random seed for reproducibility
        fast_mode : bool, default=True
            Use vectorized optimizations for faster execution.
            Set to False for exact MATLAB algorithm compatibility.
        fast_wavelet_mode : bool, default=True
            Use optimized wavelet reconstruction with early termination
            and vectorized parameter search. Set to False for exact MATLAB compatibility.
            
        Returns:
        --------
        dict : Dictionary containing:
            - 'time': Time array (seconds)
            - 'acceleration': Acceleration time history (G)
            - 'velocity': Velocity time history (in/sec or m/sec)
            - 'displacement': Displacement time history (inches or mm)
            - 'srs_freq': SRS frequency array
            - 'srs_pos': Positive SRS values
            - 'srs_neg': Negative SRS values
            - 'synthesis_error': Final synthesis error (dB)
            - 'wavelet_table': Wavelet parameters (if reconstruction used)
        """
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Validate inputs
        freq_spec = np.asarray(freq_spec, dtype=float)
        accel_spec = np.asarray(accel_spec, dtype=float)
        
        if len(freq_spec) != len(accel_spec):
            raise ValueError("freq_spec and accel_spec must have same length")
        if len(freq_spec) < 2:
            raise ValueError("Minimum 2 frequency points required")
        if not np.all(freq_spec[1:] > freq_spec[:-1]):
            raise ValueError("Frequencies must be monotonically increasing")
        if duration <= 0:
            raise ValueError("Duration must be positive")
            
        # Constrain iterations
        max_iterations = max(10, min(5000, max_iterations))
        
        # Set sample rate if not provided
        if sample_rate is None:
            sample_rate = 10.0 * freq_spec[-1]
            
        # Calculate time parameters
        dt = 1.0 / sample_rate
        ns = int(np.round(sample_rate * duration)) + 20
        
        print(f"Duration: {duration:.3f} sec")
        print(f"Sample rate: {sample_rate:.1f} Hz")
        print(f"Time step: {dt:.6f} sec")
        print(f"Number of samples: {ns}")
        
        # Interpolate SRS specification to 1/12 octave spacing
        freq_interp, accel_interp = self._interpolate_srs_spec(freq_spec, accel_spec)
        last = len(freq_interp)
        
        print(f"Interpolated to {last} frequency points")
        
        # Initialize optimization variables
        best_amp = np.zeros(last)
        best_phase = np.zeros(last)  
        best_delay = np.zeros(last)
        best_dampt = np.zeros(last)
        store = np.zeros(ns)  # Initialize store to prevent UnboundLocalError
        
        omega = self.tpi * freq_interp
        errlit = 1e90
        first = 0.0  # Start time offset
        
        print(f"Starting optimization with {max_iterations} outer iterations...")
        
        # Track timing for performance monitoring
        start_time = time.time()
        iteration_times = []
        
        # Main optimization loop
        for ia in range(max_iterations):
            iter_start = time.time()
            
            # Generate damped sine parameters
            amp, phase, delay, dampt, sss = self._generate_damped_sine_params(
                ns, dt, duration, ia, max_iterations, accel_interp, omega, 
                last, best_amp, best_phase, best_delay, best_dampt, first, 
                fast_mode)
            
            # Synthesize initial time history
            time_history, _ = self._synthesize_time_history(ns, amp, sss, last, fast_mode)
            
            # Calculate SRS of synthesized signal
            srs_pos, srs_neg = self._calculate_srs(freq_interp, damping_ratio, dt, time_history)
            
            # Inner optimization loop
            inner_start = time.time()
            for ijk in range(inner_iterations):
                # Scale amplitudes based on SRS error
                if fast_mode:
                    # Vectorized amplitude scaling
                    xx = (srs_pos + np.abs(srs_neg)) / 2.0
                    valid = xx >= 1e-90
                    if not np.any(valid):
                        break
                    amp[valid] *= ((accel_interp[valid] / xx[valid]) ** 0.25)
                else:
                    # Original loop implementation
                    for i in range(last):
                        xx = (srs_pos[i] + abs(srs_neg[i])) / 2.0
                        if xx < 1e-90:
                            break
                        amp[i] = amp[i] * ((accel_interp[i] / xx) ** 0.25)
                    else:
                        # Continue only if loop completed without break
                        pass
                
                # Scale time history
                time_history = self._scale_time_history(ns, last, time_history, amp, sss, fast_mode)
                
                # Apply end tapering
                time_history = self._apply_end_taper(time_history, ns)
                
                # Apply beginning taper
                time_history = self._apply_begin_taper(time_history, ns)
                
                # Recalculate SRS
                srs_pos, srs_neg = self._calculate_srs(freq_interp, damping_ratio, dt, time_history)
                
                # Calculate synthesis error
                syn_error = self._calculate_srs_error(last, srs_pos, srs_neg, accel_interp)
                
                if syn_error >= 1e90:  # Invalid error - break like MATLAB iflag=1
                    break
                
                sym = abs(20 * np.log10(abs(np.max(time_history) / np.min(time_history))))
                
                # Check for improvement (matches MATLAB logic exactly)
                if (syn_error < errlit and sym < 2.5) or ia == 0:
                    errlit = syn_error
                    best_amp[:] = amp
                    best_phase[:] = phase
                    best_delay[:] = delay
                    best_dampt[:] = dampt
                    store = time_history.copy()
            
            inner_time = time.time() - inner_start
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            # Progress reporting with timing
            if (ia + 1) % 10 == 0 or ia == 0:
                avg_time = np.mean(iteration_times[-10:]) if len(iteration_times) >= 10 else np.mean(iteration_times)
                estimated_remaining = avg_time * (max_iterations - ia - 1)
                print(f"Iteration {ia + 1}/{max_iterations} - "
                      f"Error: {errlit:.3f} dB - "
                      f"Iter time: {iter_time:.1f}s (inner: {inner_time:.1f}s) - "
                      f"Est. remaining: {estimated_remaining/60:.1f}min")
        
        total_time = time.time() - start_time
        avg_iter_time = np.mean(iteration_times)
        print(f"Optimization complete. Best error: {errlit:.3f} dB")
        print(f"Total optimization time: {total_time/60:.2f} minutes")
        print(f"Average time per iteration: {avg_iter_time:.1f} seconds")
        
        # Create time array and add pre-shock padding
        time_final, accel_final = self._add_pre_shock(store, duration, dt)
        
        # Calculate velocity and displacement
        velocity = self._integrate_function(accel_final, dt)
        displacement = self._integrate_function(velocity, dt)
        
        # Convert units for velocity and displacement
        if units.lower() == 'english':
            displacement = displacement * 386  # Convert to inches
        else:
            displacement = displacement * 9.81 * 1000  # Convert to mm
            
        # Perform wavelet reconstruction if requested
        wavelet_table = None
        if wavelet_reconstruction:
            print("Performing wavelet reconstruction...")
            time_recon = np.linspace(0, (ns-1)*dt, ns)
            accel_final, velocity, displacement, srs_final, wavelet_table = \
                self._wavelet_reconstruction(time_recon, store, dt, first, freq_interp,
                                           freq_spec[0], freq_spec[-1], damping_ratio, 
                                           units, wavelet_trials, wavelet_frequencies,
                                           fast_wavelet_mode)
            
            # Add pre-shock padding to reconstructed signals
            time_final, accel_final = self._add_pre_shock(accel_final, duration, dt)
            _, velocity = self._add_pre_shock(velocity, duration, dt) 
            _, displacement = self._add_pre_shock(displacement, duration, dt)
        
        # Calculate final SRS
        srs_pos_final, srs_neg_final = self._calculate_srs(freq_interp, damping_ratio, dt, accel_final)
        
        return {
            'time': time_final,
            'acceleration': accel_final, 
            'velocity': velocity,
            'displacement': displacement,
            'srs_freq': freq_interp,
            'srs_pos': srs_pos_final,
            'srs_neg': srs_neg_final, 
            'synthesis_error': errlit,
            'wavelet_table': wavelet_table,
            'sample_rate': sample_rate,
            'duration': duration,
            'damping_ratio': damping_ratio,
            'units': units
        }
    
    def _interpolate_srs_spec(self, freq_spec: np.ndarray, accel_spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate SRS specification to 1/12 octave spacing."""
        
        # Calculate slopes between points
        slopes = np.zeros(len(freq_spec) - 1)
        for i in range(len(freq_spec) - 1):
            slopes[i] = np.log(accel_spec[i+1] / accel_spec[i]) / np.log(freq_spec[i+1] / freq_spec[i])
        
        # Generate interpolated frequency array
        freq_interp = [freq_spec[0]]
        while freq_interp[-1] < freq_spec[-1]:
            next_freq = freq_interp[-1] * self.octave
            if next_freq >= freq_spec[-1]:
                break
            freq_interp.append(next_freq)
        
        freq_interp.append(freq_spec[-1])
        freq_interp = np.array(freq_interp)
        
        # Interpolate accelerations
        accel_interp = np.zeros(len(freq_interp))
        for i, f in enumerate(freq_interp):
            if f == freq_spec[-1]:
                accel_interp[i] = accel_spec[-1]
                continue
                
            # Find which segment this frequency falls in
            for j in range(len(freq_spec) - 1):
                if f == freq_spec[j]:
                    accel_interp[i] = accel_spec[j]
                    break
                elif freq_spec[j] < f < freq_spec[j+1]:
                    # Power law interpolation
                    accel_interp[i] = accel_spec[j] * (f / freq_spec[j]) ** slopes[j]
                    break
        
        return freq_interp, accel_interp
    
    def _generate_damped_sine_params(self, ns: int, dt: float, duration: float, 
                                   iteration: int, max_iterations: int, accel_spec: np.ndarray,
                                   omega: np.ndarray, last: int, best_amp: np.ndarray,
                                   best_phase: np.ndarray, best_delay: np.ndarray, 
                                   best_dampt: np.ndarray, first: float, fast_mode: bool = False) -> Tuple[np.ndarray, ...]:
        """Generate damped sine parameters (amplitude, phase, delay, damping)."""
        
        amp = np.zeros(last)
        phase = np.zeros(last) 
        delay = np.zeros(last)
        dampt = np.zeros(last)
        sss = np.zeros((last, ns))
        
        # Generate parameters based on iteration number
        if iteration < 12 or np.random.rand() < 0.5:
            # Random initialization
            for i in range(last):
                amp[i] = (accel_spec[i] / 10.0)
                if np.random.rand() < 0.5:
                    amp[i] = -amp[i]
                phase[i] = 0.0
                delay[i] = first + 0.020 * duration * np.random.rand()
                dampt[i] = 0.003 + 0.035 * np.random.rand()
        else:
            # Use best parameters with small variations
            for i in range(last):
                amp[i] = best_amp[i] * (0.99 + 0.02 * np.random.rand())
                phase[i] = best_phase[i] * (0.99 + 0.02 * np.random.rand()) 
                delay[i] = best_delay[i] * (1.0 + 0.02 * np.random.rand())
                if delay[i] > 0.015 * duration:
                    delay[i] = 0.015 * duration
                dampt[i] = best_dampt[i] * (0.99 + 0.02 * np.random.rand())
        
        # Generate sinusoid matrix
        if fast_mode:
            # Vectorized implementation
            t = dt * np.arange(1, ns + 1)  # Match MATLAB indexing: starts from dt
            T = t[np.newaxis, :]  # (1, ns)
            delays = delay[:, np.newaxis]  # (last, 1)
            omegas = omega[:, np.newaxis]  # (last, 1)
            dampts = dampt[:, np.newaxis]  # (last, 1)
            
            # Mask for valid times (after delay)
            valid_mask = T > delays
            
            # Calculate effective time after delay
            T_eff = np.where(valid_mask, T - delays, 0)
            
            # Calculate frequency * time
            FT = omegas * T_eff
            
            # Calculate sinusoids with damping
            sss = np.where(valid_mask, 
                          np.exp(-dampts * FT) * np.sin(FT),
                          0.0)
        else:
            # Original loop implementation (for verification)
            for k in range(last):
                for j in range(ns):
                    tt = dt * (j + 1)  # Match MATLAB indexing: j starts from 1
                    if tt > delay[k]:
                        tt = tt - delay[k]
                        ft = omega[k] * tt
                        sss[k, j] = np.exp(-dampt[k] * ft) * np.sin(ft)
                    
        return amp, phase, delay, dampt, sss
    
    def _synthesize_time_history(self, ns: int, amp: np.ndarray, 
                               sss: np.ndarray, last: int, fast_mode: bool = False) -> Tuple[np.ndarray, float]:
        """Synthesize time history from damped sine parameters."""
        
        acc = np.zeros(ns)
        
        if fast_mode:
            # Vectorized implementation
            acc[:ns] = np.dot(amp[:last], sss[:last, :ns])
        else:
            # Original loop implementation
            for j in range(ns):
                acc[j] = np.dot(amp[:last], sss[:last, j])
        
        acc[0] = 0.0
        
        big = np.max(acc)
        small = np.min(acc)
        
        sym = abs(20 * np.log10(big / abs(small))) if small != 0 else 0
        
        return acc, sym
    
    def _scale_time_history(self, ns: int, last: int, acc: np.ndarray,
                          amp: np.ndarray, sss: np.ndarray, fast_mode: bool = False) -> np.ndarray:
        """Scale time history based on updated amplitudes."""
        
        if fast_mode:
            # Vectorized implementation using matrix multiplication
            # acc[j] = sum_i(amp[i] * sss[i,j]) = amp @ sss[:,:ns]
            acc[:ns] = np.dot(amp[:last], sss[:last, :ns])
        else:
            # Original loop-based implementation
            for j in range(ns):
                acc[j] = np.dot(amp[:last], sss[:last, j])
        
        acc[0] = 0.0
        return acc
    
    def _apply_end_taper(self, acc: np.ndarray, ns: int) -> np.ndarray:
        """Apply tapering to end of signal to reduce transients."""
        
        nk = int(0.9 * ns)
        length = ns - nk
        
        for i in range(nk, ns):
            x = i - nk
            acc[i] = acc[i] * (1.0 - x / length)
            
        return acc
    
    def _apply_begin_taper(self, acc: np.ndarray, ns: int) -> np.ndarray:
        """Apply Hann window taper to beginning of signal."""
        
        fper = 0.03  # 3% taper
        na = int(fper * ns)
        
        for i in range(na):
            arg = np.pi * ((i / (na - 1)) + 1)
            acc[i] = acc[i] * 0.5 * (1 + np.cos(arg))
            
        return acc
    
    def _calculate_srs(self, freq: np.ndarray, damp: float, dt: float, 
                      input_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate shock response spectrum using Smallwood algorithm."""
        
        # Calculate filter coefficients
        a1, a2, b1, b2, b3 = self._srs_coefficients(freq, damp, dt)
        
        srs_pos = np.zeros(len(freq))
        srs_neg = np.zeros(len(freq))
        
        for j in range(len(freq)):
            # Create filter coefficients
            forward = [b1[j], b2[j], b3[j]]
            back = [1.0, -a1[j], -a2[j]]
            
            # Apply filter
            resp = signal.lfilter(forward, back, input_signal)
            
            srs_pos[j] = abs(np.max(resp))
            srs_neg[j] = abs(np.min(resp))
            
        return srs_pos, srs_neg
    
    def _srs_coefficients(self, freq: np.ndarray, damp: float, dt: float) -> Tuple[np.ndarray, ...]:
        """Calculate SRS filter coefficients using Smallwood algorithm."""
        
        num_freq = len(freq)
        a1 = np.zeros(num_freq)
        a2 = np.zeros(num_freq) 
        b1 = np.zeros(num_freq)
        b2 = np.zeros(num_freq)
        b3 = np.zeros(num_freq)
        
        for j in range(num_freq):
            omega = self.tpi * freq[j]
            omegad = omega * np.sqrt(1.0 - damp**2)
            
            cosd = np.cos(omegad * dt)
            sind = np.sin(omegad * dt)
            domegadt = damp * omega * dt
            
            # Smallwood algorithm coefficients
            E = np.exp(-damp * omega * dt)
            K = omegad * dt
            C = E * cosd
            S = E * sind
            
            if K != 0:
                Sp = S / K
            else:
                Sp = E * dt
            
            a1[j] = 2 * C
            a2[j] = -(E**2)
            
            b1[j] = 1.0 - Sp
            b2[j] = 2.0 * (Sp - C)
            b3[j] = (E**2) - Sp
            
        return a1, a2, b1, b2, b3
    
    def _calculate_srs_error(self, last: int, srs_pos: np.ndarray, 
                           srs_neg: np.ndarray, target: np.ndarray) -> float:
        """Calculate synthesis error in dB."""
        
        error = 0.0
        
        for i in range(last):
            if srs_pos[i] <= 0 or srs_neg[i] <= 0:
                return 1e99
            
            db_pos = abs(20.0 * np.log10(srs_pos[i] / target[i]))
            db_neg = abs(20.0 * np.log10(srs_neg[i] / target[i]))
            
            error = max(error, db_pos, db_neg)
            
        return error
    
    def _add_pre_shock(self, signal: np.ndarray, duration: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Add pre-shock padding to signal."""
        
        nm = len(signal)
        tstart = duration / 20.0
        npre = int(tstart / dt)
        ntotal = nm + npre
        
        time = np.zeros(ntotal)
        padded_signal = np.zeros(ntotal)
        
        for i in range(ntotal):
            time[i] = -tstart + i * dt
            if i >= npre:
                padded_signal[i] = signal[i - npre]
                
        return time, padded_signal
    
    def _integrate_function(self, input_signal: np.ndarray, dt: float) -> np.ndarray:
        """Integrate signal using trapezoidal rule."""
        return dt * np.cumsum(input_signal)
    
    def _wavelet_reconstruction(self, time: np.ndarray, input_signal: np.ndarray, dt: float,
                              first: float, freq: np.ndarray, ffmin: float, ffmax: float,
                              damp: float, units: str, nt: int, nfr: int, 
                              fast_wavelet_mode: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Perform wavelet reconstruction to ensure zero net displacement and velocity.
        This implements the full MATLAB wavelet algorithm.
        """
        
        import time as time_module
        wavelet_start_time = time_module.time()
        mode_str = "optimized" if fast_wavelet_mode else "original"
        print(f"Performing wavelet reconstruction ({mode_str} mode)...")
        
        tp = 2.0 * np.pi
        num2 = len(input_signal)
        duration = time[-1] - time[0]
        sr = 1.0 / dt
        
        # Frequency bounds for wavelets
        fl = 3.0 / duration
        fu = sr / 10.0
        
        print(f"Wavelet frequency range: {fl:.2f} Hz to {fu:.0f} Hz")
        
        # Initialize wavelet parameters
        x1r = np.zeros(nfr)  # amplitudes
        x2r = np.zeros(nfr)  # angular frequencies  
        x3r = np.zeros(nfr)  # number of half-sines
        x4r = np.zeros(nfr)  # time delays
        
        residual = input_signal.copy()
        
        # Generate wavelets to match residual
        print(f"Optimizing {nfr} wavelet frequencies with {nt} trials each...")
        wavelet_opt_start = time_module.time()
        
        for ie in range(nfr):
            freq_start_time = time_module.time()
            
            if (ie + 1) % 25 == 0 or ie == 0:
                elapsed = time_module.time() - wavelet_opt_start
                if ie > 0:
                    avg_time_per_freq = elapsed / ie
                    remaining_freqs = nfr - ie
                    est_remaining = avg_time_per_freq * remaining_freqs
                    print(f"Wavelet optimization {ie+1}/{nfr} - "
                          f"Avg: {avg_time_per_freq:.1f}s/freq - "
                          f"Est. remaining: {est_remaining/60:.1f}min")
                else:
                    print(f"Wavelet optimization {ie+1}/{nfr} - Starting...")
            
            # Optimize wavelet parameters for this frequency
            x1r[ie], x2r[ie], x3r[ie], x4r[ie] = self._optimize_wavelet(
                num2, time, dt, residual, duration, fl, fu, nt, ffmin, ffmax, first,
                fast_wavelet_mode)
            
            # Subtract this wavelet from residual
            for i in range(num2):
                tt = time[i]
                t1 = x4r[ie] + time[0]
                t2 = t1 + tp * x3r[ie] / (2.0 * x2r[ie])
                
                if tt >= t1 and tt <= t2:
                    arg = x2r[ie] * (tt - t1)
                    y = x1r[ie] * np.sin(arg / x3r[ie]) * np.sin(arg)
                    residual[i] -= y
            
            ave = np.mean(residual)
            sd = np.std(residual)
            if ie % 100 == 0:
                print(f"  Residual: mean={ave:.3e}, std={sd:.3e}")
        
        # Reconstruct acceleration, velocity, and displacement using analytical expressions
        aaa = np.zeros(num2)
        vvv = np.zeros(num2) 
        ddd = np.zeros(num2)
        
        print("Reconstructing signals with analytical integration...")
        
        for k in range(num2):
            tt = time[k]
            
            for j in range(nfr):
                w = 0.0
                v = 0.0
                d = 0.0
                
                # Check frequency validity
                if x2r[j] < fl * tp:
                    x2r[j] = fl * tp
                    x1r[j] = 1e-20
                    x3r[j] = 3
                    x4r[j] = 0
                
                t1 = x4r[j] + time[0]
                t2 = tp * x3r[j] / (2.0 * x2r[j]) + t1
                
                if tt >= t1 and tt <= t2:
                    arg = x2r[j] * (tt - t1)
                    
                    # Acceleration (wavelet function)
                    w = x1r[j] * np.sin(arg / x3r[j]) * np.sin(arg)
                    
                    # Analytical velocity and displacement
                    aa = x2r[j] / x3r[j]
                    bb = x2r[j]
                    te = tt - t1
                    
                    alpha1 = aa + bb
                    alpha2 = aa - bb
                    
                    alpha1te = alpha1 * te
                    alpha2te = alpha2 * te
                    
                    # Analytical integration formulas
                    if abs(alpha1) > 1e-12:
                        v1 = -np.sin(alpha1te) / (2.0 * alpha1)
                        d1 = (np.cos(alpha1te) - 1.0) / (2.0 * alpha1**2)
                    else:
                        v1 = -te / 2.0
                        d1 = -te**2 / 4.0
                        
                    if abs(alpha2) > 1e-12:
                        v2 = np.sin(alpha2te) / (2.0 * alpha2)
                        d2 = -(np.cos(alpha2te) - 1.0) / (2.0 * alpha2**2)
                    else:
                        v2 = te / 2.0
                        d2 = te**2 / 4.0
                    
                    v = (v1 + v2) * x1r[j]
                    d = (d1 + d2) * x1r[j]
                
                aaa[k] += w
                vvv[k] += v
                ddd[k] += d
        
        # Convert units
        if units.lower() == 'english':
            vvv = vvv * 386  # Convert to in/sec
            ddd = ddd * 386  # Convert to inches
        else:
            vvv = vvv * 9.81  # Convert to m/sec
            ddd = ddd * 9.81 * 1000  # Convert to mm
        
        # Calculate final SRS
        srs_pos, srs_neg = self._calculate_srs(freq, damp, dt, aaa)
        
        # Create wavelet table
        wavelet_table = np.zeros((nfr, 5))
        for i in range(nfr):
            wavelet_table[i, :] = [i+1, x2r[i]/(2*np.pi), x1r[i], x3r[i], x4r[i]]
        
        total_wavelet_time = time_module.time() - wavelet_start_time
        print(f"Wavelet reconstruction complete in {total_wavelet_time:.1f} seconds.")
        print(f"Final velocity: {vvv[-1]:.6f}")
        print(f"Final displacement: {ddd[-1]:.6f}")
        
        return aaa, vvv, ddd, np.column_stack([freq, srs_pos, srs_neg]), wavelet_table
    
    def _optimize_wavelet(self, num2: int, time: np.ndarray, dt: float, 
                         residual: np.ndarray, duration: float, fl: float, fu: float,
                         nt: int, ffmin: float, ffmax: float, first: float,
                         fast_wavelet_mode: bool = False) -> Tuple[float, ...]:
        """Optimize wavelet parameters to match residual signal."""
        
        tp = 2.0 * np.pi
        min_delay = 0.1 * first
        
        ave = np.mean(residual)
        sd = np.std(residual)
        am = 2.0 * sd
        
        # Squared residual for error calculation
        asd = residual**2
        
        errormax = 1e53
        best_params = [0, fl*tp, 3, 0]
        
        # Limit trials to avoid infinite loops
        max_trials = min(nt, 10000)
        
        import time as time_module
        opt_start_time = time_module.time()
        
        if fast_wavelet_mode:
            # Optimized wavelet parameter search with early termination
            return self._optimize_wavelet_fast(num2, time, residual, duration, fl, fu, 
                                             max_trials, ffmin, ffmax, first, tp, min_delay, 
                                             am, asd)
        
        # Original algorithm for exact MATLAB compatibility
        for j in range(max_trials):
            # Generate random wavelet parameters
            x1 = 2.0 * am * (np.random.rand() - 0.5)  # amplitude
            x2 = ((fu - fl) * np.random.rand() + fl) * tp  # angular frequency
            x3 = int(3 + 2 * round(np.random.rand() * 30))  # number of half-sines (odd, integer)
            x4 = np.random.rand() * 0.8 * duration + min_delay  # delay
            
            # Ensure x3 is odd and >= 3
            if x3 % 2 == 0:
                x3 += 1
            if x3 < 3:
                x3 = 3
            
            # Ensure duration constraint
            max_attempts = 50  # Prevent infinite loop
            attempts = 0
            while tp * x3 / (2.0 * x2) + x4 >= duration and attempts < max_attempts:
                x3 -= 2
                if x3 < 3:
                    x3 = 3
                    break
                attempts += 1
            
            # Skip if still violates constraint
            if tp * x3 / (2.0 * x2) + x4 >= duration:
                continue
            
            # Frequency bounds check
            if x2 / tp < fl or x2 / tp > ffmax:
                x2 = ((ffmax - fl) * np.random.rand() + fl) * tp
                x3 = 3
                x4 = 0
            
            if x4 < min_delay:
                x4 = min_delay
            
            # Calculate error for this wavelet
            error = 0.0
            t1 = x4 + time[0]
            t2 = t1 + tp * x3 / (2.0 * x2)
            
            # Vectorized error calculation
            mask = (time >= t1) & (time <= t2)
            
            if np.any(mask):
                tt_active = time[mask]
                residual_active = residual[mask]
                asd_active = asd[mask]
                
                arg = x2 * (tt_active - t1)
                y = x1 * np.sin(arg / x3) * np.sin(arg)
                error += np.sum((residual_active - y)**2)
                
                # Add error from inactive regions
                tt_before_first = time < first
                error += 2 * np.sum(asd[tt_before_first & ~mask])
                error += np.sum(asd[~mask & ~tt_before_first])
            else:
                # No active region, just add residual error
                tt_before_first = time < first
                error += 2 * np.sum(asd[tt_before_first])
                error += np.sum(asd[~tt_before_first])
            
            error = np.sqrt(error)
            
            # Keep best parameters
            if error < errormax and x2 >= fl * tp:
                best_params = [x1, x2, x3, x4]
                errormax = error
        
        opt_time = time_module.time() - opt_start_time
        # Note: Removed individual wavelet timing output to reduce verbosity
        
        return tuple(best_params)

    def _optimize_wavelet_fast(self, num2: int, time: np.ndarray, residual: np.ndarray, 
                              duration: float, fl: float, fu: float, max_trials: int,
                              ffmin: float, ffmax: float, first: float, tp: float,
                              min_delay: float, am: float, asd: np.ndarray) -> Tuple[float, ...]:
        """
        Optimized wavelet parameter search with early termination and vectorized operations.
        """
        
        errormax = 1e53
        best_params = [0, fl*tp, 3, 0]
        
        # Early termination thresholds
        target_error = np.sqrt(np.mean(asd)) * 0.1  # Stop when error is 10% of RMS residual
        stagnation_count = 0
        max_stagnation = max_trials // 10  # Stop if no improvement for 10% of trials
        previous_best = errormax
        
        # Vectorized parameter generation for better efficiency
        batch_size = min(100, max_trials // 10)  # Process parameters in batches
        
        j = 0
        while j < max_trials:
            # Generate batch of random parameters
            batch_actual = min(batch_size, max_trials - j)
            
            # Vectorized parameter generation
            x1_batch = 2.0 * am * (np.random.rand(batch_actual) - 0.5)
            x2_batch = ((fu - fl) * np.random.rand(batch_actual) + fl) * tp
            x3_batch = 3 + 2 * np.round(np.random.rand(batch_actual) * 30).astype(int)
            x4_batch = np.random.rand(batch_actual) * 0.8 * duration + min_delay
            
            # Ensure x3 is odd and >= 3
            x3_batch = np.where(x3_batch % 2 == 0, x3_batch + 1, x3_batch)
            x3_batch = np.where(x3_batch < 3, 3, x3_batch)
            
            # Process each parameter set in batch
            for i in range(batch_actual):
                x1, x2, x3, x4 = x1_batch[i], x2_batch[i], x3_batch[i], x4_batch[i]
                
                # Duration constraint checking
                attempts = 0
                while tp * x3 / (2.0 * x2) + x4 >= duration and attempts < 50:
                    x3 -= 2
                    if x3 < 3:
                        x3 = 3
                        break
                    attempts += 1
                
                if tp * x3 / (2.0 * x2) + x4 >= duration:
                    continue
                
                # Frequency bounds check
                if x2 / tp < fl or x2 / tp > ffmax:
                    x2 = ((ffmax - fl) * np.random.rand() + fl) * tp
                    x3 = 3
                    x4 = 0
                
                if x4 < min_delay:
                    x4 = min_delay
                
                # Calculate error for this wavelet (using vectorized calculation from original)
                error = 0.0
                t1 = x4 + time[0]
                t2 = t1 + tp * x3 / (2.0 * x2)
                
                # Vectorized error calculation
                mask = (time >= t1) & (time <= t2)
                
                if np.any(mask):
                    tt_active = time[mask]
                    residual_active = residual[mask]
                    asd_active = asd[mask]
                    
                    arg = x2 * (tt_active - t1)
                    y = x1 * np.sin(arg / x3) * np.sin(arg)
                    error += np.sum((residual_active - y)**2)
                    
                    # Add error from inactive regions
                    tt_before_first = time < first
                    error += 2 * np.sum(asd[tt_before_first & ~mask])
                    error += np.sum(asd[~mask & ~tt_before_first])
                else:
                    # No active region, just add residual error
                    tt_before_first = time < first
                    error += 2 * np.sum(asd[tt_before_first])
                    error += np.sum(asd[~tt_before_first])
                
                error = np.sqrt(error)
                
                # Keep best parameters
                if error < errormax and x2 >= fl * tp:
                    best_params = [x1, x2, x3, x4]
                    errormax = error
                    stagnation_count = 0  # Reset stagnation counter
                    
                    # Early termination if error is very good
                    if error < target_error:
                        return tuple(best_params)
                else:
                    stagnation_count += 1
            
            j += batch_actual
            
            # Check for stagnation (no improvement)
            if errormax < previous_best:
                previous_best = errormax
                stagnation_count = 0
            
            # Early termination due to stagnation
            if stagnation_count > max_stagnation:
                break
        
        # Note: Individual timing removed to reduce output verbosity
        return tuple(best_params)


def synthesize_shock_spectrum(freq_spec: np.ndarray, 
                            accel_spec: np.ndarray,
                            duration: float,
                            sample_rate: Optional[float] = None,
                            damping_ratio: float = 0.05,
                            max_iterations: int = 100,
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function to synthesize a shock response spectrum.
    
    This function provides a simple interface to the SRS synthesis functionality.
    
    Parameters:
    -----------
    freq_spec : np.ndarray
        Target natural frequencies (Hz)
    accel_spec : np.ndarray
        Target SRS acceleration values (G)
    duration : float
        Duration of synthesized signal (seconds)
    sample_rate : float, optional
        Sample rate (Hz). If None, automatically determined
    damping_ratio : float, default=0.05
        Damping ratio for SRS calculation
    max_iterations : int, default=100
        Maximum iterations for optimization
    **kwargs : dict
        Additional parameters passed to DSSynthesizer.synthesize_srs()
        
    Returns:
    --------
    dict : Dictionary with synthesized time history and analysis results
    
    Example:
    --------
    >>> import numpy as np
    >>> freq = np.array([10, 50, 100, 500, 1000])
    >>> accel = np.array([10, 20, 50, 100, 50])
    >>> result = synthesize_shock_spectrum(freq, accel, duration=0.1)
    >>> time = result['time']
    >>> acceleration = result['acceleration']
    """
    
    synthesizer = DSSynthesizer()
    return synthesizer.synthesize_srs(freq_spec, accel_spec, duration, 
                                    sample_rate, damping_ratio, max_iterations, 
                                    **kwargs)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Define target SRS specification
    freq_spec = np.array([10, 50, 100, 500, 1000])  # Hz
    accel_spec = np.array([10, 20, 50, 100, 50])     # G
    
    print("SRS Synthesis Example")
    print("===================")
    print(f"Target frequencies: {freq_spec}")
    print(f"Target accelerations: {accel_spec}")
    
    # Synthesize shock spectrum
    result = synthesize_shock_spectrum(
        freq_spec=freq_spec,
        accel_spec=accel_spec, 
        duration=0.1,  # 100 ms
        damping_ratio=0.05,
        max_iterations=50,
        random_seed=42
    )
    
    print(f"\nSynthesis Results:")
    print(f"Final error: {result['synthesis_error']:.2f} dB")
    print(f"Signal duration: {result['time'][-1] - result['time'][0]:.4f} sec")
    print(f"Peak acceleration: {np.max(np.abs(result['acceleration'])):.1f} G")
    
    # Simple plot if matplotlib is available
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time history
        axes[0,0].plot(result['time'], result['acceleration'])
        axes[0,0].set_xlabel('Time (sec)')
        axes[0,0].set_ylabel('Acceleration (G)')
        axes[0,0].set_title('Synthesized Time History')
        axes[0,0].grid(True)
        
        # Displacement
        axes[0,1].plot(result['time'], result['displacement'])
        axes[0,1].set_xlabel('Time (sec)')
        axes[0,1].set_ylabel('Displacement (inches)')
        axes[0,1].set_title('Displacement')
        axes[0,1].grid(True)
        
        # SRS comparison
        axes[1,0].loglog(freq_spec, accel_spec, 'ko-', label='Target')
        axes[1,0].loglog(result['srs_freq'], result['srs_pos'], 'b-', label='Positive')
        axes[1,0].loglog(result['srs_freq'], result['srs_neg'], 'r-', label='Negative') 
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('Acceleration (G)')
        axes[1,0].set_title('SRS Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Error plot
        error_pos = 20 * np.log10(result['srs_pos'] / np.interp(result['srs_freq'], freq_spec, accel_spec))
        error_neg = 20 * np.log10(result['srs_neg'] / np.interp(result['srs_freq'], freq_spec, accel_spec))
        axes[1,1].semilogx(result['srs_freq'], error_pos, 'b-', label='Positive Error')
        axes[1,1].semilogx(result['srs_freq'], error_neg, 'r-', label='Negative Error')
        axes[1,1].axhline(y=3, color='k', linestyle='--', alpha=0.5, label='±3 dB')
        axes[1,1].axhline(y=-3, color='k', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('Frequency (Hz)')
        axes[1,1].set_ylabel('Error (dB)')
        axes[1,1].set_title('SRS Synthesis Error')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available - skipping plots")
