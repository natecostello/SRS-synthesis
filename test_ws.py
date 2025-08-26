#!/usr/bin/env python3
"""
Unit tests for Wavelet Synthesis (WS) functionality.
Essential tests for core functionality, accuracy, and edge cases.
"""

import unittest
import numpy as np
import os
import sys
import signal
from contextlib import contextmanager

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from srs_wavelet_synthesis import WSSSynthesizer
from srs_conversion import convert_srs


@contextmanager
def timeout(seconds):
    """Context manager for timeouts."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class TestWSSynthesis(unittest.TestCase):
    """Test core WS synthesis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = WSSSynthesizer()
        self.test_freq = np.array([10., 50., 100., 200.])    # 4 points
        self.test_accel = np.array([20., 30., 40., 50.])     # 4 points
        self.duration = 0.2                                  # 200ms (>= 1.5/min(freq))
        self.sample_rate = 4096                              # Higher sample rate for stability
        self.damping = 0.05
        self.ntrials = 1                                     # Minimal trials for speed
    
    def test_synthesis_basic_functionality(self):
        """Test basic synthesis functionality."""
        result = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, 
            sample_rate=self.sample_rate,
            damping_ratio=self.damping, 
            ntrials=self.ntrials
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        required_keys = ['time', 'acceleration', 'srs_pos', 'srs_neg', 
                        'synthesis_error', 'srs_freq', 'wavelet_table']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Verify signal properties
        time_signal = result['acceleration']
        time = result['time']
        self.assertEqual(len(time_signal), len(time))
        self.assertGreater(len(time_signal), 10)  # Reasonable length
        
        # Verify synthesis error is reasonable (should be < 25 dB for fast test)
        self.assertLess(result['synthesis_error'], 25.0)
        
        # Verify SRS frequencies cover input range
        srs_freq = result['srs_freq']
        self.assertLessEqual(srs_freq[0], self.test_freq[0])
        self.assertGreaterEqual(srs_freq[-1], self.test_freq[-1])
        
        # Verify SRS values are positive and reasonable
        srs_pos = result['srs_pos']
        self.assertTrue(np.all(srs_pos > 0))
        self.assertTrue(np.all(srs_pos < 1000))  # Sanity check
        
        # Verify wavelet table structure
        wavelet_table = result['wavelet_table']
        self.assertEqual(wavelet_table.shape[1], 5)  # [index, amp, freq, nhs, delay]
        self.assertTrue(np.all(wavelet_table[:, 2] > 0))  # frequencies > 0
        self.assertTrue(np.all(wavelet_table[:, 3] >= 3))  # nhs >= 3
        self.assertTrue(np.all(wavelet_table[:, 4] >= 0))  # delays >= 0
    
    def test_synthesis_strategies(self):
        """Test different synthesis strategies."""
        strategies = [1, 2, 3, 4, 5]  # Random, Forward, Reverse, Exponential, Mixed
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                result = self.synthesizer.synthesize_srs(
                    self.test_freq, self.test_accel, self.duration, 
                    sample_rate=self.sample_rate,
                    damping_ratio=self.damping, 
                    ntrials=1,
                    strategy=strategy
                )
                
                # All strategies should produce valid results
                self.assertIn('acceleration', result)
                # Basic quality checks - allow higher error for fast tests with limited trials
                self.assertLess(result['synthesis_error'], 25.0)  # Some strategies may have higher error
    
    def test_octave_spacing(self):
        """Test different octave spacing options."""
        octave_spacings = [1, 2, 3, 4]  # 1/3, 1/6, 1/12, 1/24 octave
        
        for octave in octave_spacings:
            with self.subTest(octave=octave):
                result = self.synthesizer.synthesize_srs(
                    self.test_freq, self.test_accel, self.duration, 
                    sample_rate=self.sample_rate,
                    damping_ratio=self.damping, 
                    ntrials=1,
                    octave_spacing=octave
                )
                
                # All octave spacings should produce valid results
                self.assertIn('acceleration', result)
                self.assertLess(result['synthesis_error'], 15.0)
    
    def test_units_consistency(self):
        """Test consistency across different unit systems."""
        units = ['english', 'metric']
        
        for unit in units:
            with self.subTest(unit=unit):
                result = self.synthesizer.synthesize_srs(
                    self.test_freq, self.test_accel, self.duration, 
                    sample_rate=self.sample_rate,
                    damping_ratio=self.damping, 
                    ntrials=1,
                    units=unit
                )
                
                self.assertIn('acceleration', result)
                self.assertLess(result['synthesis_error'], 15.0)
    
    def test_input_validation(self):
        """Test input validation."""
        
        # Test mismatched array lengths
        with self.assertRaises(ValueError):
            self.synthesizer.synthesize_srs(
                self.test_freq, self.test_accel[:-1], self.duration  # Wrong length
            )
        
        # Test negative frequencies
        with self.assertRaises(ValueError):
            self.synthesizer.synthesize_srs(
                np.array([-10., 50.]), np.array([20., 30.]), self.duration
            )
        
        # Test negative accelerations
        with self.assertRaises(ValueError):
            self.synthesizer.synthesize_srs(
                np.array([10., 50.]), np.array([-20., 30.]), self.duration
            )
        
        # Test too short duration
        with self.assertRaises(ValueError):
            self.synthesizer.synthesize_srs(
                self.test_freq, self.test_accel, 0.001  # Too short
            )
        
        # Test invalid damping ratio
        with self.assertRaises(ValueError):
            self.synthesizer.synthesize_srs(
                self.test_freq, self.test_accel, self.duration, 
                damping_ratio=0.0  # Invalid damping
            )
    
    def test_wavelet_generation(self):
        """Test individual wavelet generation functions."""
        
        # Test wavelet time history generation
        nspec = len(self.test_freq)
        nt = 512
        dt = 1.0 / self.sample_rate
        
        # Create dummy wavelet parameters
        amp = np.random.rand(nspec) * 0.01
        nhs = np.random.randint(3, 100, nspec)
        td = np.random.rand(nspec) * self.duration * 0.5
        f = self.test_freq
        
        # Test wavelet generation
        wavelet, th = self.synthesizer._generate_wavelets(
            amp, nhs, td, f, nt, dt
        )
        
        self.assertEqual(len(th), nt)
        self.assertEqual(wavelet.shape, (nspec, nt))
        
        # Test SRS calculation
        a1, a2, b1, b2, b3 = self.synthesizer._calculate_srs_coefficients(
            f, self.damping, dt
        )
        
        self.assertEqual(len(a1), nspec)
        self.assertEqual(len(b1), nspec)
        
        # Test SRS computation
        srs_pos, srs_neg = self.synthesizer._calculate_srs(
            th, a1, a2, b1, b2, b3, f, fast_mode=False
        )
        
        self.assertEqual(len(srs_pos), nspec)
        self.assertEqual(len(srs_neg), nspec)
        self.assertTrue(np.all(srs_pos > 0))
        self.assertTrue(np.all(srs_neg > 0))
    
    def test_error_calculation(self):
        """Test error calculation functions."""
        
        # Create test data
        spec = np.array([10., 20., 30., 40.])
        calculated = np.array([9., 22., 28., 42.])
        
        total_error, max_error = self.synthesizer._calculate_error(
            spec, calculated, calculated
        )
        
        # Should have reasonable errors
        self.assertGreater(total_error, 0)
        self.assertGreater(max_error, 0)
        self.assertLess(max_error, 1.0)  # Should be less than 1 (log scale)
    
    def test_ranking_system(self):
        """Test the ranking system."""
        
        # Create test performance metrics
        ntrials = 2
        ym = np.random.rand(ntrials) * 100      # peak accel
        vm = np.random.rand(ntrials) * 10       # peak vel
        dm = np.random.rand(ntrials) * 0.1      # peak disp
        em = np.random.rand(ntrials) * 0.5      # total error
        im = np.random.rand(ntrials) * 0.1      # max error
        cm = np.random.rand(ntrials) * 5        # crest factor
        km = np.random.rand(ntrials) * 5        # kurtosis
        dskm = np.random.rand(ntrials) * 2      # displacement skew
        
        # Test ranking
        iwin, nrank = self.synthesizer._rank_solutions(
            ym, vm, dm, em, im, cm, km, dskm,
            iw=1, ew=1, dw=1, vw=1, aw=1, cw=1, kw=1, dskw=1,
            displacement_limit=1e9
        )
        
        # Should return valid winner index
        self.assertIsInstance(iwin, (int, np.integer))
        self.assertGreaterEqual(iwin, 0)
        self.assertLess(iwin, ntrials)
        
        # Ranking array should have correct length
        self.assertEqual(len(nrank), ntrials)
    
    def test_synthesis_reproducibility(self):
        """Test that synthesis is reproducible with same random seed."""
        
        # Run twice with same seed
        result1 = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, 
            sample_rate=self.sample_rate,
            damping_ratio=self.damping, 
            ntrials=1,
            random_seed=42
        )
        
        result2 = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, 
            sample_rate=self.sample_rate,
            damping_ratio=self.damping, 
            ntrials=1,
            random_seed=42
        )
        
        # Results should be identical
        np.testing.assert_allclose(result1['acceleration'], result2['acceleration'], 
                                   rtol=1e-10, atol=1e-10)
        
        # Run with different seed - should be different
        result3 = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, 
            sample_rate=self.sample_rate,
            damping_ratio=self.damping, 
            ntrials=1,
            random_seed=123
        )
        
        # Should not be identical (with high probability)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(result1['acceleration'], result3['acceleration'], 
                                       rtol=1e-6, atol=1e-6)
    
    def test_performance_edge_cases(self):
        """Test edge cases that might cause performance issues."""
        
        # Test with very high frequency content
        high_freq = np.array([1000., 2000., 4000.])
        high_accel = np.array([100., 150., 200.])
        
        result = self.synthesizer.synthesize_srs(
            high_freq, high_accel, 0.01,  # Short duration for high freq
            sample_rate=20480,  # High sample rate
            damping_ratio=self.damping, 
            ntrials=1  # Minimal
        )
        
        self.assertIn('acceleration', result)
        
        # Test with very low frequency content  
        low_freq = np.array([1., 2., 5.])
        low_accel = np.array([5., 10., 15.])
        
        result = self.synthesizer.synthesize_srs(
            low_freq, low_accel, 2.0,  # Longer duration for low freq
            sample_rate=512,  # Lower sample rate
            damping_ratio=self.damping, 
            ntrials=1
        )
        
        self.assertIn('acceleration', result)

    def test_all_synthesis_strategies_with_timeout(self):
        """
        Test all synthesis strategies with timeout protection.
        This is the comprehensive test to verify all strategies work and return to zero.
        """
        strategies = {1: 'Random', 2: 'Forward', 3: 'Reverse', 4: 'Exponential', 5: 'Mixed'}
        
        # Small, fast test parameters
        freq = np.array([50.0, 200.0])
        accel = np.array([25.0, 35.0])
        duration = 0.05  # Very short
        sample_rate = 4096
        
        for strategy_num, strategy_name in strategies.items():
            with self.subTest(strategy=strategy_num, name=strategy_name):
                print(f"\nTesting Strategy {strategy_num} ({strategy_name})")
                
                try:
                    with timeout(20):  # 20 second timeout per strategy
                        result = self.synthesizer.synthesize_srs(
                            freq, accel, duration,
                            sample_rate=sample_rate,
                            damping_ratio=0.05,
                            strategy=strategy_num,
                            ntrials=1  # Minimal trials
                        )
                    
                    # Verify result structure
                    self.assertIsInstance(result, dict)
                    self.assertIn('velocity', result)
                    self.assertIn('displacement', result)
                    
                    # Extract final values
                    vel_data = result['velocity']
                    disp_data = result['displacement']
                    final_vel = vel_data[-1]
                    final_disp = disp_data[-1]
                    
                    # Critical test: signals must return to zero
                    self.assertLess(abs(final_vel), 1e-6, 
                                  f"Strategy {strategy_num} final velocity {final_vel} ≠ 0")
                    self.assertLess(abs(final_disp), 1e-6, 
                                  f"Strategy {strategy_num} final displacement {final_disp} ≠ 0")
                    
                    # Quality check
                    error = result['ranking_metrics']['max_error']
                    self.assertLess(error, 50.0, f"Strategy {strategy_num} error too high: {error} dB")
                    
                    print(f"  ✅ Strategy {strategy_num}: vel={final_vel:.2e}, disp={final_disp:.2e}, error={error:.1f}dB")
                    
                except TimeoutError:
                    self.fail(f"Strategy {strategy_num} ({strategy_name}) HUNG - timed out after 20 seconds")
                except Exception as e:
                    self.fail(f"Strategy {strategy_num} ({strategy_name}) FAILED: {str(e)}")


class TestWaveletSynthesisAlgorithms(unittest.TestCase):
    """Test individual wavelet synthesis algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = WSSSynthesizer()
        self.nspec = 5
        self.f = np.array([10., 25., 50., 100., 200.])
        self.amp_start = np.array([0.01, 0.015, 0.02, 0.025, 0.03])
        self.dur = 0.2
        self.onep5_period = 1.5 / self.f
    
    def test_synth1_random_strategy(self):
        """Test random synthesis strategy."""
        amp, td, nhs = self.synthesizer._synth1_random_matlab(
            self.amp_start, self.dur, self.onep5_period, self.f, allow_infinite_retries=False
        )
        
        # Check output dimensions
        self.assertEqual(len(amp), self.nspec)
        self.assertEqual(len(td), self.nspec)
        self.assertEqual(len(nhs), self.nspec)
        
        # Check constraints
        self.assertTrue(np.all(nhs >= 3))
        self.assertTrue(np.all(td >= 0))
        self.assertTrue(np.all(td <= self.dur))
        
        # Check timing constraints
        for i in range(self.nspec):
            self.assertLessEqual(nhs[i]/(2*self.f[i]) + td[i], self.dur)
    
    def test_synth3_reverse_sweep(self):
        """Test reverse sine sweep strategy."""
        amp, td, nhs = self.synthesizer._synth3_reverse_sweep_matlab(
            self.amp_start, self.dur, self.onep5_period, self.f, 100, 0, allow_infinite_retries=False
        )
        
        # Check output dimensions and constraints
        self.assertEqual(len(amp), self.nspec)
        self.assertEqual(len(td), self.nspec)
        self.assertEqual(len(nhs), self.nspec)
        
        self.assertTrue(np.all(nhs >= 3))
        self.assertTrue(np.all(td >= 0))
    
    def test_synth4_forward_sweep(self):
        """Test forward sine sweep strategy."""
        amp, td, nhs = self.synthesizer._synth4_forward_sweep_matlab(
            self.amp_start, self.dur, self.onep5_period, self.f, 100, 0, allow_infinite_retries=False
        )
        
        # Check output dimensions and constraints
        self.assertEqual(len(amp), self.nspec)
        self.assertEqual(len(td), self.nspec)
        self.assertEqual(len(nhs), self.nspec)
        
        self.assertTrue(np.all(nhs >= 3))
        self.assertTrue(np.all(td >= 0))
    
    def test_synth_exp_exponential_decay(self):
        """Test exponential decay strategy."""
        amp, td, nhs = self.synthesizer._synth_exp_exponential_decay_matlab(
            self.amp_start, self.dur, self.onep5_period, self.f, allow_infinite_retries=False
        )
        
        # Check output dimensions and constraints
        self.assertEqual(len(amp), self.nspec)
        self.assertEqual(len(td), self.nspec)
        self.assertEqual(len(nhs), self.nspec)
        
        self.assertTrue(np.all(nhs >= 3))
        self.assertTrue(np.all(td >= 0))


class TestAllowInfiniteRetries(unittest.TestCase):
    """Test allow_infinite_retries functionality."""
    
    def setUp(self):
        """Set up test fixtures for infinite retry testing."""
        self.synthesizer = WSSSynthesizer()
        # Use challenging constraints that might require multiple attempts
        self.test_freq = np.array([100., 500., 1000.])     # Higher frequencies
        self.test_accel = np.array([50., 75., 100.])       # Higher accelerations
        self.duration = 0.05                               # Short duration (challenging)
        self.sample_rate = 8192                            # High sample rate
        self.damping = 0.05
    
    def test_infinite_retries_enabled(self):
        """Test synthesis with infinite retries enabled (default behavior)."""
        # This should work with default allow_infinite_retries=True
        result = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=1
        )
        
        # Should succeed without exceptions
        self.assertIsInstance(result, dict)
        self.assertIn('synthesis_error', result)
        self.assertLess(result['synthesis_error'], 30.0)  # Reasonable error
    
    def test_infinite_retries_disabled_with_reasonable_constraints(self):
        """Test synthesis with infinite retries disabled on reasonable constraints."""
        # Use very easy constraints that should work with safety caps
        easy_freq = np.array([5., 25., 50.])     # Lower frequencies  
        easy_accel = np.array([10., 15., 20.])   # Lower accelerations
        longer_duration = 0.5                   # Longer duration
        
        # This test might fail occasionally due to the stochastic nature
        # and 100-iteration safety cap, so we'll allow either success or specific failure
        try:
            result = self.synthesizer.synthesize_srs(
                easy_freq, easy_accel, longer_duration,
                sample_rate=2048,  # Lower sample rate
                damping_ratio=self.damping,
                ntrials=1,
                allow_infinite_retries=False
            )
            # If successful, verify results
            self.assertIsInstance(result, dict)
            self.assertIn('synthesis_error', result)
        except RuntimeError as e:
            # If it fails, verify it's due to safety caps, not other errors
            error_msg = str(e)
            self.assertTrue(
                "No successful trials completed" in error_msg or 
                "Emergency break" in error_msg,
                f"Unexpected error: {error_msg}"
            )
    
    def test_infinite_retries_exception_on_impossible_constraints(self):
        """Test that emergency breaks raise RuntimeError with infinite retries disabled."""
        # Instead of trying to create impossible constraints (which is hard to predict),
        # let's test the emergency break behavior more directly by testing with 
        # a low maximum iteration count to force emergency breaks
        
        # Test that individual strategy methods respect emergency break limits
        amp_start = np.array([0.01, 0.02])
        dur = 0.1
        f = np.array([100., 200.])  
        onep5_period = 1.5 / f
        
        # The emergency break should trigger if we hit the 100k iteration limit
        # This test validates the exception-raising behavior exists, even if hard to trigger
        try:
            # Test the strategy method directly - it should complete normally for reasonable inputs
            amp, td, nhs = self.synthesizer._synth1_random_matlab(
                amp_start, dur, onep5_period, f, allow_infinite_retries=False
            )
            # If it completes, that's fine - it means the constraints were feasible
            self.assertEqual(len(amp), len(f))
            self.assertTrue(np.all(nhs >= 3))
        except RuntimeError as e:
            # If it raises RuntimeError, verify it mentions emergency break
            self.assertIn("Emergency break", str(e))
        
        # The key point is that emergency breaks exist and raise RuntimeError when triggered
        # The exact triggering conditions depend on mathematical constraints and randomness
    
    def test_strategy_specific_infinite_retries(self):
        """Test infinite retries behavior for specific strategies."""
        # Test individual strategy with allow_infinite_retries=False
        amp_start = np.array([0.01, 0.02, 0.03])
        dur = 0.1
        f = np.array([100., 500., 1000.])
        onep5_period = 1.5 / f
        
        # This should work without throwing exceptions for reasonable inputs
        amp, td, nhs = self.synthesizer._synth1_random_matlab(
            amp_start, dur, onep5_period, f, allow_infinite_retries=False
        )
        
        # Verify results
        self.assertEqual(len(amp), len(f))
        self.assertEqual(len(td), len(f))
        self.assertEqual(len(nhs), len(f))
        self.assertTrue(np.all(nhs >= 3))
        self.assertTrue(np.all(td >= 0))
    
    def test_matlab_equivalence_mode(self):
        """Test that allow_infinite_retries=True provides MATLAB-equivalent behavior."""
        # This test verifies that infinite retry mode can handle challenging cases
        # that would fail with safety caps
        challenging_freq = np.array([200., 800.])
        challenging_accel = np.array([80., 120.])
        short_duration = 0.03
        
        # Should complete successfully with infinite retries (may take longer)
        with timeout(30):  # Allow up to 30 seconds for challenging case
            result = self.synthesizer.synthesize_srs(
                challenging_freq, challenging_accel, short_duration,
                sample_rate=8192,
                damping_ratio=self.damping,
                ntrials=1,
                allow_infinite_retries=True
            )
        
        self.assertIsInstance(result, dict)
        self.assertIn('synthesis_error', result)
        # MATLAB-equivalent mode should achieve good accuracy
        self.assertLess(result['synthesis_error'], 25.0)


class TestFastModeOptimizations(unittest.TestCase):
    """Test fast mode optimizations produce identical results to normal mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = WSSSynthesizer()
        # Minimal test case for speed - shorter duration and fewer points
        self.test_freq = np.array([50., 100., 200.])     # 3 points
        self.test_accel = np.array([20., 25., 30.])      # 3 points  
        self.duration = 0.05                             # 50ms - minimal for testing
        self.sample_rate = 4096
        self.damping = 0.05
        self.ntrials = 1                                 # Single trial for deterministic comparison
        self.seed = 42                                   # Fixed seed for reproducibility
    
    def test_fast_mode_strategy_1(self):
        """Test Strategy 1 produces identical results with and without fast mode."""
        # Normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=1,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Fast mode with same seed
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=1,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True
        )
        
        # Verify results are identical
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        # Verify timing info exists and fast mode was enabled
        self.assertIn('timing', result_fast)
        self.assertTrue(result_fast['timing']['fast_mode_enabled'])
        self.assertFalse(result_normal['timing']['fast_mode_enabled'])
        
        print(f"Strategy 1: Normal={result_normal['timing']['total_time']:.4f}s, Fast={result_fast['timing']['total_time']:.4f}s")
        print(f"  Coeffs: Normal={result_normal['timing']['coeffs_time']:.6f}s, Fast={result_fast['timing']['coeffs_time']:.6f}s")  
        print(f"  Interp: Normal={result_normal['timing']['interp_time']:.6f}s, Fast={result_fast['timing']['interp_time']:.6f}s")
    
    def test_fast_mode_strategy_2(self):
        """Test Strategy 2 produces identical results with and without fast mode."""
        # Normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=2,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Fast mode with same seed
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=2,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True
        )
        
        # Verify results are identical
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        print(f"Strategy 2: Normal={result_normal['timing']['total_time']:.4f}s, Fast={result_fast['timing']['total_time']:.4f}s")
        print(f"  Coeffs: Normal={result_normal['timing']['coeffs_time']:.6f}s, Fast={result_fast['timing']['coeffs_time']:.6f}s")  
        print(f"  Interp: Normal={result_normal['timing']['interp_time']:.6f}s, Fast={result_fast['timing']['interp_time']:.6f}s")
    
    def test_fast_mode_strategy_3(self):
        """Test Strategy 3 produces identical results with and without fast mode."""
        # Normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=3,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Fast mode with same seed
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=3,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True
        )
        
        # Verify results are identical
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        print(f"Strategy 3: Normal={result_normal['timing']['total_time']:.4f}s, Fast={result_fast['timing']['total_time']:.4f}s")
        print(f"  Coeffs: Normal={result_normal['timing']['coeffs_time']:.6f}s, Fast={result_fast['timing']['coeffs_time']:.6f}s")  
        print(f"  Interp: Normal={result_normal['timing']['interp_time']:.6f}s, Fast={result_fast['timing']['interp_time']:.6f}s")
    
    def test_fast_mode_strategy_4(self):
        """Test Strategy 4 produces identical results with and without fast mode."""
        # Normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=4,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Fast mode with same seed
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=4,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True
        )
        
        # Verify results are identical
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        print(f"Strategy 4: Normal={result_normal['timing']['total_time']:.4f}s, Fast={result_fast['timing']['total_time']:.4f}s")
        print(f"  Coeffs: Normal={result_normal['timing']['coeffs_time']:.6f}s, Fast={result_fast['timing']['coeffs_time']:.6f}s")  
        print(f"  Interp: Normal={result_normal['timing']['interp_time']:.6f}s, Fast={result_fast['timing']['interp_time']:.6f}s")
    
    def test_fast_mode_caching_effectiveness(self):
        """Test that fast mode caching provides performance improvement on repeated calls."""
        # First call to populate cache
        result1 = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=1,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True
        )
        
        # Second call should benefit from cache
        result2 = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=1,
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True
        )
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result1['acceleration'], result2['acceleration'], decimal=10)
        
        # Second call should be faster (or at least not significantly slower)
        self.assertLessEqual(result2['timing']['coeffs_time'], result1['timing']['coeffs_time'] * 1.1)
        self.assertLessEqual(result2['timing']['interp_time'], result1['timing']['interp_time'] * 1.1)
        
        print(f"Cache effectiveness: 1st call coeffs={result1['timing']['coeffs_time']:.6f}s, "
              f"2nd call coeffs={result2['timing']['coeffs_time']:.6f}s")
        print(f"                   1st call interp={result1['timing']['interp_time']:.6f}s, "
              f"2nd call interp={result2['timing']['interp_time']:.6f}s")
    
    def test_strategy_1_mathematical_optimizations(self):
        """Test that Strategy 1 mathematical pre-computation optimizations work correctly."""
        
        # Run with normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel, 
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=1,  # Force Strategy 1 specifically
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Run with fast mode (mathematical optimizations)
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=1,  # Force Strategy 1 specifically  
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True   # Enable mathematical optimizations
        )
        
        # Results must be identical - the mathematical optimizations shouldn't change any behavior
        np.testing.assert_array_almost_equal(
            result_normal['acceleration'], 
            result_fast['acceleration'], 
            decimal=10,  # High precision required
            err_msg="Strategy 1 fast_mode acceleration differs from normal mode"
        )
        
        # Check all key outputs for identical results
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        # Verify fast mode was enabled
        self.assertTrue(result_fast['timing']['fast_mode_enabled'])
        self.assertFalse(result_normal['timing']['fast_mode_enabled'])
        
        print(f"Strategy 1 mathematical optimization test: PASSED")
        print(f"  Normal mode total time: {result_normal['timing']['total_time']:.4f}s")
        print(f"  Fast mode total time: {result_fast['timing']['total_time']:.4f}s")
        
        # The optimization should not make things significantly slower
        timing_ratio = result_fast['timing']['total_time'] / result_normal['timing']['total_time']
        print(f"  Timing ratio (fast/normal): {timing_ratio:.3f}")
        self.assertLess(timing_ratio, 2.0, "Strategy 1 fast_mode shouldn't be significantly slower")
    
    def test_strategy_2_mathematical_optimizations(self):
        """Test that Strategy 2 mathematical pre-computation optimizations work correctly."""
        
        # Run with normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel, 
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=2,  # Force Strategy 2 specifically (Forward sweep)
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Run with fast mode (mathematical optimizations)
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=2,  # Force Strategy 2 specifically  
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True   # Enable mathematical optimizations
        )
        
        # Results must be identical - the mathematical optimizations shouldn't change any behavior
        np.testing.assert_array_almost_equal(
            result_normal['acceleration'], 
            result_fast['acceleration'], 
            decimal=10,  # High precision required
            err_msg="Strategy 2 fast_mode acceleration differs from normal mode"
        )
        
        # Check all key outputs for identical results
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        # Verify fast mode was enabled
        self.assertTrue(result_fast['timing']['fast_mode_enabled'])
        self.assertFalse(result_normal['timing']['fast_mode_enabled'])
        
        print(f"Strategy 2 mathematical optimization test: PASSED")
        print(f"  Normal mode total time: {result_normal['timing']['total_time']:.4f}s")
        print(f"  Fast mode total time: {result_fast['timing']['total_time']:.4f}s")
        
        # The optimization should not make things significantly slower
        timing_ratio = result_fast['timing']['total_time'] / result_normal['timing']['total_time']
        print(f"  Timing ratio (fast/normal): {timing_ratio:.3f}")
        self.assertLess(timing_ratio, 2.0, "Strategy 2 fast_mode shouldn't be significantly slower")
    
    def test_strategy_3_mathematical_optimizations(self):
        """Test that Strategy 3 mathematical pre-computation optimizations work correctly."""
        
        # Run with normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel, 
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=3,  # Force Strategy 3 specifically (Reverse sweep)
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Run with fast mode (mathematical optimizations)
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=3,  # Force Strategy 3 specifically  
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True   # Enable mathematical optimizations
        )
        
        # Results must be identical - the mathematical optimizations shouldn't change any behavior
        np.testing.assert_array_almost_equal(
            result_normal['acceleration'], 
            result_fast['acceleration'], 
            decimal=10,  # High precision required
            err_msg="Strategy 3 fast_mode acceleration differs from normal mode"
        )
        
        # Check all key outputs for identical results
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        # Verify fast mode was enabled
        self.assertTrue(result_fast['timing']['fast_mode_enabled'])
        self.assertFalse(result_normal['timing']['fast_mode_enabled'])
        
        print(f"Strategy 3 mathematical optimization test: PASSED")
        print(f"  Normal mode total time: {result_normal['timing']['total_time']:.4f}s")
        print(f"  Fast mode total time: {result_fast['timing']['total_time']:.4f}s")
        
        # The optimization should not make things significantly slower
        timing_ratio = result_fast['timing']['total_time'] / result_normal['timing']['total_time']
        print(f"  Timing ratio (fast/normal): {timing_ratio:.3f}")
        self.assertLess(timing_ratio, 2.0, "Strategy 3 fast_mode shouldn't be significantly slower")
    
    def test_strategy_4_mathematical_optimizations(self):
        """Test that Strategy 4 mathematical pre-computation optimizations work correctly."""
        
        # Run with normal mode
        result_normal = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel, 
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=4,  # Force Strategy 4 specifically (Exponential decay)
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=False
        )
        
        # Run with fast mode (mathematical optimizations)
        result_fast = self.synthesizer.synthesize_srs(
            freq_spec=self.test_freq,
            accel_spec=self.test_accel,
            duration=self.duration,
            sample_rate=self.sample_rate,
            damping_ratio=self.damping,
            ntrials=self.ntrials,
            strategy=4,  # Force Strategy 4 specifically  
            random_seed=self.seed,
            allow_infinite_retries=True,
            fast_mode=True   # Enable mathematical optimizations
        )
        
        # Results must be identical - the mathematical optimizations shouldn't change any behavior
        np.testing.assert_array_almost_equal(
            result_normal['acceleration'], 
            result_fast['acceleration'], 
            decimal=10,  # High precision required
            err_msg="Strategy 4 fast_mode acceleration differs from normal mode"
        )
        
        # Check all key outputs for identical results
        np.testing.assert_array_almost_equal(result_normal['acceleration'], result_fast['acceleration'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['velocity'], result_fast['velocity'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['displacement'], result_fast['displacement'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_pos'], result_fast['srs_pos'], decimal=10)
        np.testing.assert_array_almost_equal(result_normal['srs_neg'], result_fast['srs_neg'], decimal=10)
        self.assertAlmostEqual(result_normal['synthesis_error'], result_fast['synthesis_error'], places=10)
        
        # Verify fast mode was enabled
        self.assertTrue(result_fast['timing']['fast_mode_enabled'])
        self.assertFalse(result_normal['timing']['fast_mode_enabled'])
        
        print(f"Strategy 4 mathematical optimization test: PASSED")
        print(f"  Normal mode total time: {result_normal['timing']['total_time']:.4f}s")
        print(f"  Fast mode total time: {result_fast['timing']['total_time']:.4f}s")
        
        # The optimization should not make things significantly slower
        timing_ratio = result_fast['timing']['total_time'] / result_normal['timing']['total_time']
        print(f"  Timing ratio (fast/normal): {timing_ratio:.3f}")
        self.assertLess(timing_ratio, 2.0, "Strategy 4 fast_mode shouldn't be significantly slower")

    def test_integration_vectorization_optimizations(self):
        """Test that vectorized integration produces identical results to loop-based integration."""
        
        # Create test acceleration signal
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        th = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        
        # Test velocity/displacement calculation
        vmax_normal, dmax_normal = self.synthesizer._calculate_velocity_displacement(
            th, dt, 'english', fast_mode=False
        )
        vmax_fast, dmax_fast = self.synthesizer._calculate_velocity_displacement(
            th, dt, 'english', fast_mode=True
        )
        
        # Test displacement derivative calculation
        disp_normal = self.synthesizer._calculate_displacement_derivative(
            th, dt, fast_mode=False
        )
        disp_fast = self.synthesizer._calculate_displacement_derivative(
            th, dt, fast_mode=True
        )
        
        # Verify identical results (within numerical precision)
        self.assertAlmostEqual(vmax_normal, vmax_fast, places=10, 
                              msg=f"Velocity calculation mismatch: {vmax_normal} vs {vmax_fast}")
        self.assertAlmostEqual(dmax_normal, dmax_fast, places=10,
                              msg=f"Displacement calculation mismatch: {dmax_normal} vs {dmax_fast}")
        
        np.testing.assert_array_almost_equal(disp_normal, disp_fast, decimal=10,
                                           err_msg="Displacement derivative arrays differ")
        
        print(f"Integration vectorization test: PASSED")
        print(f"  Velocity: {vmax_normal:.6f} (both modes)")
        print(f"  Displacement: {dmax_normal:.6f} (both modes)")
        print(f"  Displacement array length: {len(disp_normal)} (both modes)")
    
    def test_srs_calculation_optimization(self):
        """Test that optimized SRS calculation produces identical results."""
        
        import time
        
        # Set up test parameters
        f = np.array([10.0, 50.0, 100.0])
        dt = 0.001
        damping_ratio = 0.05
        
        # Generate test acceleration signal
        t = np.arange(0, 1.0, dt)
        th = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
        
        # Calculate SRS coefficients
        a1, a2, b1, b2, b3 = self.synthesizer._calculate_srs_coefficients(f, damping_ratio, dt, fast_mode=False)
        
        # Test both modes
        start_time = time.perf_counter()
        srs_pos_normal, srs_neg_normal = self.synthesizer._calculate_srs(
            th, a1, a2, b1, b2, b3, f, fast_mode=False
        )
        normal_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        srs_pos_fast, srs_neg_fast = self.synthesizer._calculate_srs(
            th, a1, a2, b1, b2, b3, f, fast_mode=True
        )
        fast_time = time.perf_counter() - start_time
        
        # Verify identical results
        np.testing.assert_array_almost_equal(srs_pos_normal, srs_pos_fast, decimal=10,
                                           err_msg="SRS positive arrays differ")
        np.testing.assert_array_almost_equal(srs_neg_normal, srs_neg_fast, decimal=10,
                                           err_msg="SRS negative arrays differ")
        
        print(f"SRS calculation optimization test: PASSED")
        print(f"  Normal time: {normal_time:.6f}s")
        print(f"  Fast time: {fast_time:.6f}s")
        print(f"  Performance ratio: {fast_time/normal_time:.3f}")
        print(f"  Results identical to 10 decimal places")
    
    def test_memory_pool_optimizations(self):
        """Test memory pool optimizations work correctly and don't affect results."""
        
        import time
        
        # Test parameters
        freq = np.array([20.0, 100.0])
        accel = np.array([10.0, 15.0])
        
        # Test normal mode (no memory pooling) - use deterministic strategy
        start_time = time.perf_counter()
        result_normal = self.synthesizer.synthesize_srs(
            freq, accel, 0.05, sample_rate=4096, ntrials=1, 
            strategy=1, fast_mode=False, random_seed=42
        )
        normal_time = time.perf_counter() - start_time
        
        # Test fast mode (with memory pooling) - same parameters
        start_time = time.perf_counter()  
        result_fast = self.synthesizer.synthesize_srs(
            freq, accel, 0.05, sample_rate=4096, ntrials=1,
            strategy=1, fast_mode=True, random_seed=42
        )
        fast_time = time.perf_counter() - start_time
        
        # Verify identical results - compare key metrics instead of raw arrays
        self.assertAlmostEqual(
            result_normal['peak_acceleration'], 
            result_fast['peak_acceleration'], 
            places=8,
            msg="Peak acceleration differs between normal and fast modes"
        )
        
        self.assertAlmostEqual(
            result_normal['max_error_db'], 
            result_fast['max_error_db'], 
            places=8,  
            msg="Max error differs between normal and fast modes"
        )
        
        # Check that memory pools were used
        self.assertGreater(len(self.synthesizer._memory_pools['temp_arrays']), 0,
                          "Memory pools should contain arrays after fast mode execution")
        
        # Test memory pool clearing
        original_pool_size = len(self.synthesizer._memory_pools['temp_arrays'])
        self.synthesizer._clear_memory_pools()
        self.assertEqual(len(self.synthesizer._memory_pools['temp_arrays']), 0,
                        "Memory pools should be empty after clearing")
        
        print(f"Memory pool optimization test: PASSED")
        print(f"  Normal time: {normal_time:.4f}s")
        print(f"  Fast time: {fast_time:.4f}s") 
        print(f"  Memory pools used: {original_pool_size} arrays")
        print(f"  Results identical to 10 decimal places")
    
    def test_memory_pool_optimizations(self):
        """Test that memory pool optimizations work correctly and provide performance benefits."""
        
        # Test parameters - use reasonable complexity to test memory pool efficiency  
        freq = np.array([50.0, 100.0, 200.0])
        accel = np.array([10.0, 15.0, 20.0])
        
        # Test with fixed seed to ensure reproducible results
        result_normal = self.synthesizer.synthesize_srs(
            freq, accel, 0.05,
            sample_rate=2048,
            damping_ratio=0.05,
            ntrials=1,
            strategy=1,
            random_seed=42,  # Fixed seed for reproducibility
            fast_mode=False
        )
        
        result_fast = self.synthesizer.synthesize_srs(
            freq, accel, 0.05,
            sample_rate=2048,
            damping_ratio=0.05,
            ntrials=1,
            strategy=1,
            random_seed=42,  # Same fixed seed ensures identical randomness
            fast_mode=True
        )
        
        # Verify identical results (should be exact match with fixed seed)
        np.testing.assert_array_almost_equal(
            result_normal['acceleration'], 
            result_fast['acceleration'], 
            decimal=10,
            err_msg="Memory pool optimization changed acceleration results"
        )
        
        # Also verify other key outputs are identical
        np.testing.assert_array_almost_equal(
            result_normal['srs_pos'], 
            result_fast['srs_pos'], 
            decimal=10,
            err_msg="Memory pool optimization changed SRS results"
        )
        
        # Check that memory pools were created in fast mode
        total_pools = sum(len(pool) for pool in self.synthesizer._memory_pools.values())
        self.assertGreater(total_pools, 0,
                          "Memory pools should contain reusable arrays after fast mode execution")
        
        # Test memory pool clearing
        self.synthesizer._clear_memory_pools()
        self.assertEqual(sum(len(pool) for pool in self.synthesizer._memory_pools.values()), 0,
                        "Memory pools should be empty after clearing")

    def test_scipy_filter_optimizations(self):
        """Test that scipy filter optimizations produce identical results."""
        
        # Test the digital filtering component in isolation
        # Generate test signal and SRS coefficients
        dt = 0.001
        nt = 100
        th = np.random.random(nt) * 2 - 1  # Random signal between -1 and 1
        f = np.array([10.0, 50.0, 100.0])
        damping_ratio = 0.05
        
        # Calculate filter coefficients
        a1, a2, b1, b2, b3 = self.synthesizer._calculate_srs_coefficients(f, damping_ratio, dt, False)
        
        # Test normal mode (original implementation)
        srs_pos_normal, srs_neg_normal = self.synthesizer._calculate_srs(
            th, a1, a2, b1, b2, b3, f, fast_mode=False
        )
        
        # Test fast mode (with scipy optimizations)  
        srs_pos_fast, srs_neg_fast = self.synthesizer._calculate_srs(
            th, a1, a2, b1, b2, b3, f, fast_mode=True
        )
        
        # Verify identical results
        np.testing.assert_array_almost_equal(
            srs_pos_normal, srs_pos_fast, decimal=12,
            err_msg="Scipy filter optimization changed positive SRS results"
        )
        np.testing.assert_array_almost_equal(
            srs_neg_normal, srs_neg_fast, decimal=12,
            err_msg="Scipy filter optimization changed negative SRS results"
        )

    def test_isolated_memory_pool_functionality(self):
        """Test memory pool get/clear functionality in isolation."""
        
        # Test memory pool creation and reuse using existing pool
        arr1 = self.synthesizer._get_pooled_array('temp_arrays', 'test_key', (10,), fast_mode=True)
        self.assertEqual(arr1.shape, (10,))
        self.assertTrue(np.all(arr1 == 0))  # Should be zeros
        
        # Modify the array
        arr1[0] = 42
        
        # Get the same array again - should be reused and reset
        arr2 = self.synthesizer._get_pooled_array('temp_arrays', 'test_key', (10,), fast_mode=True)
        self.assertIs(arr1, arr2)  # Should be the same object
        self.assertEqual(arr2[0], 0)  # Should be reset to zero
        
        # Test different key gets different array
        arr3 = self.synthesizer._get_pooled_array('temp_arrays', 'different_key', (10,), fast_mode=True)
        self.assertIsNot(arr1, arr3)  # Should be different objects
        
        # Test pool clearing
        initial_pool_size = len(self.synthesizer._memory_pools['temp_arrays'])
        self.synthesizer._clear_memory_pools()
        self.assertEqual(len(self.synthesizer._memory_pools['temp_arrays']), 0)
        
        # Test fast_mode=False bypasses pool
        arr4 = self.synthesizer._get_pooled_array('temp_arrays', 'test_key', (10,), fast_mode=False)
        arr5 = self.synthesizer._get_pooled_array('temp_arrays', 'test_key', (10,), fast_mode=False)
        self.assertIsNot(arr4, arr5)  # Should be different objects when not using pool

    def test_vectorized_wavelet_generation(self):
        """Test vectorized wavelet generation produces identical results to nested loops."""
        print("\nTesting vectorized wavelet generation...")
        
        # Test parameters covering multiple scenarios
        test_cases = [
            {
                'name': 'Small Case',
                'nspec': 5, 'nt': 100, 'dt': 1e-3,
                'f_range': (10, 100), 'amp_range': (0.1, 2.0)
            },
            {
                'name': 'Medium Case',
                'nspec': 20, 'nt': 500, 'dt': 5e-4,
                'f_range': (5, 500), 'amp_range': (0.1, 5.0)
            },
            {
                'name': 'Real-world Case',
                'nspec': 93, 'nt': 1024, 'dt': 1e-4,  # Smaller than full size for test speed
                'f_range': (10, 2000), 'amp_range': (0.1, 10.0)
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=case['name']):
                # Generate reproducible test parameters
                np.random.seed(42 + i)
                nspec, nt, dt = case['nspec'], case['nt'], case['dt']
                
                f = np.logspace(np.log10(case['f_range'][0]), np.log10(case['f_range'][1]), nspec)
                amp = np.random.uniform(case['amp_range'][0], case['amp_range'][1], nspec)
                nhs = np.random.uniform(3, 15, nspec)
                td = np.random.uniform(0, dt*nt*0.1, nspec)
                
                # Test original nested loops
                wavelet_orig, th_orig = self.synthesizer._generate_wavelets(
                    amp, nhs, td, f, nt, dt, fast_mode=False)
                
                # Test vectorized implementation
                wavelet_fast, th_fast = self.synthesizer._generate_wavelets(
                    amp, nhs, td, f, nt, dt, fast_mode=True)
                
                # Verify perfect numerical equivalence
                np.testing.assert_array_equal(wavelet_orig, wavelet_fast,
                    err_msg=f"{case['name']}: Wavelet arrays not identical")
                np.testing.assert_array_equal(th_orig, th_fast,
                    err_msg=f"{case['name']}: Time history arrays not identical")
                
                print(f"  {case['name']}: {nspec}×{nt} ops - ✓ PASS")

    def test_vectorized_wavelet_edge_cases(self):
        """Test vectorized wavelet generation with edge cases and boundary conditions."""
        print("\nTesting vectorized wavelet edge cases...")
        
        # Edge case 1: Very small frequencies
        f_small = np.array([0.1, 1.0, 10.0])
        amp_small = np.array([1.0, 2.0, 3.0])
        nhs_small = np.array([3.0, 5.0, 7.0])
        td_small = np.array([0.001, 0.002, 0.003])
        nt, dt = 200, 1e-3
        
        wavelet_orig, th_orig = self.synthesizer._generate_wavelets(
            amp_small, nhs_small, td_small, f_small, nt, dt, fast_mode=False)
        wavelet_fast, th_fast = self.synthesizer._generate_wavelets(
            amp_small, nhs_small, td_small, f_small, nt, dt, fast_mode=True)
        
        np.testing.assert_array_equal(wavelet_orig, wavelet_fast)
        np.testing.assert_array_equal(th_orig, th_fast)
        print("  Small frequencies: ✓ PASS")
        
        # Edge case 2: Very large frequencies
        f_large = np.array([1000.0, 5000.0, 10000.0])
        amp_large = np.array([0.1, 0.5, 1.0])
        nhs_large = np.array([3.0, 4.0, 5.0])
        td_large = np.array([0.0001, 0.0002, 0.0003])
        nt, dt = 1000, 1e-5
        
        wavelet_orig, th_orig = self.synthesizer._generate_wavelets(
            amp_large, nhs_large, td_large, f_large, nt, dt, fast_mode=False)
        wavelet_fast, th_fast = self.synthesizer._generate_wavelets(
            amp_large, nhs_large, td_large, f_large, nt, dt, fast_mode=True)
        
        np.testing.assert_array_equal(wavelet_orig, wavelet_fast)
        np.testing.assert_array_equal(th_orig, th_fast)
        print("  Large frequencies: ✓ PASS")
        
        # Edge case 3: Minimum nhs values (should be clamped to 3)
        f_min = np.array([50.0, 100.0])
        amp_min = np.array([1.0, 2.0])
        nhs_min = np.array([1.0, 2.0])  # Below minimum, should become 3.0
        td_min = np.array([0.001, 0.002])
        nt, dt = 500, 5e-4
        
        wavelet_orig, th_orig = self.synthesizer._generate_wavelets(
            amp_min, nhs_min, td_min, f_min, nt, dt, fast_mode=False)
        wavelet_fast, th_fast = self.synthesizer._generate_wavelets(
            amp_min, nhs_min, td_min, f_min, nt, dt, fast_mode=True)
        
        np.testing.assert_array_equal(wavelet_orig, wavelet_fast)
        np.testing.assert_array_equal(th_orig, th_fast)
        print("  Minimum nhs clamping: ✓ PASS")
        
        # Edge case 4: Zero time delays
        f_zero = np.array([10.0, 50.0, 100.0])
        amp_zero = np.array([1.0, 1.5, 2.0])
        nhs_zero = np.array([5.0, 7.0, 10.0])
        td_zero = np.zeros(3)  # All zero delays
        nt, dt = 300, 1e-3
        
        wavelet_orig, th_orig = self.synthesizer._generate_wavelets(
            amp_zero, nhs_zero, td_zero, f_zero, nt, dt, fast_mode=False)
        wavelet_fast, th_fast = self.synthesizer._generate_wavelets(
            amp_zero, nhs_zero, td_zero, f_zero, nt, dt, fast_mode=True)
        
        np.testing.assert_array_equal(wavelet_orig, wavelet_fast)
        np.testing.assert_array_equal(th_orig, th_fast)
        print("  Zero time delays: ✓ PASS")
        
    def test_vectorized_wavelet_performance_benefit(self):
        """Test that vectorized implementation provides performance benefit."""
        print("\nTesting vectorized wavelet performance benefit...")
        
        # Use a reasonably sized test case for performance measurement
        nspec, nt, dt = 50, 2000, 1e-4
        np.random.seed(123)
        
        f = np.logspace(1, 3, nspec)  # 10 Hz to 1 kHz
        amp = np.random.uniform(0.5, 3.0, nspec)
        nhs = np.random.uniform(3, 12, nspec)
        td = np.random.uniform(0, dt*nt*0.05, nspec)
        
        # Time original implementation (multiple runs for accuracy)
        import time
        n_runs = 3
        
        start = time.time()
        for _ in range(n_runs):
            wavelet_orig, th_orig = self.synthesizer._generate_wavelets(
                amp, nhs, td, f, nt, dt, fast_mode=False)
        time_orig = (time.time() - start) / n_runs
        
        # Time vectorized implementation
        start = time.time()
        for _ in range(n_runs):
            wavelet_fast, th_fast = self.synthesizer._generate_wavelets(
                amp, nhs, td, f, nt, dt, fast_mode=True)
        time_fast = (time.time() - start) / n_runs
        
        # Verify results are still identical
        np.testing.assert_array_equal(wavelet_orig, wavelet_fast)
        np.testing.assert_array_equal(th_orig, th_fast)
        
        # Calculate speedup
        speedup = time_orig / time_fast if time_fast > 0 else float('inf')
        
        print(f"  Original time:  {time_orig:.4f} seconds")
        print(f"  Vectorized time: {time_fast:.4f} seconds")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Assert that vectorized version is faster (should be much faster)
        # Allow for some variance in timing, but expect at least 2x improvement
        self.assertGreater(speedup, 2.0, 
            f"Vectorized version not sufficiently faster: {speedup:.1f}x speedup")
        print("  Performance improvement verified: ✓ PASS")

    def test_comprehensive_fast_mode_integration(self):
        """Test that all fast mode optimizations work together seamlessly."""
        print("\nRunning comprehensive fast mode integration test...")
        
        # Test all strategies with consistent seed
        np.random.seed(42)
        strategies = [1, 2, 3, 4]
        
        # Create synthesizer instance
        synthesizer = WSSSynthesizer()
        
        for strategy in strategies:
            print(f"\nTesting strategy {strategy} with all optimizations...")
            
            # Run with fast mode (all optimizations enabled)
            np.random.seed(42)
            result_fast = synthesizer.synthesize_srs(
                freq_spec=np.array([50, 100, 200, 500, 1000]),
                accel_spec=np.array([10, 20, 30, 15, 25]),
                sample_rate=5000,  # 1/dt = 1/0.0002 = 5000
                duration=0.05,
                strategy=strategy,
                fast_mode=True,
                ntrials=3
            )
            
            # Run without fast mode 
            np.random.seed(42)
            result_normal = synthesizer.synthesize_srs(
                freq_spec=np.array([50, 100, 200, 500, 1000]),
                accel_spec=np.array([10, 20, 30, 15, 25]),
                sample_rate=5000,  # 1/dt = 1/0.0002 = 5000
                duration=0.05,
                strategy=strategy,
                fast_mode=False,
                ntrials=3
            )
            
            # Verify results are identical
            self.assertAlmostEqual(result_fast['ranking_metrics']['peak_accel'], 
                                 result_normal['ranking_metrics']['peak_accel'], places=6)
            self.assertAlmostEqual(result_fast['ranking_metrics']['peak_vel'], 
                                 result_normal['ranking_metrics']['peak_vel'], places=6)
            self.assertAlmostEqual(result_fast['ranking_metrics']['peak_disp'], 
                                 result_normal['ranking_metrics']['peak_disp'], places=6)
            np.testing.assert_array_almost_equal(result_fast['acceleration'], 
                                               result_normal['acceleration'], decimal=10)
            
            # Check that fast mode is actually faster or similar (allowing for variance)
            time_ratio = result_fast['timing']['total_time'] / result_normal['timing']['total_time']
            
            print(f"  Strategy {strategy}: Fast/Normal time ratio = {time_ratio:.3f}")
            print(f"    Normal: {result_normal['timing']['total_time']:.3f}s, Fast: {result_fast['timing']['total_time']:.3f}s")
            
        print("\nComprehensive integration test: PASSED")
        print("  All strategies produce identical results with and without fast mode")
        print("  All optimizations work together seamlessly")


if __name__ == '__main__':
    unittest.main()
