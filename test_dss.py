#!/usr/bin/env python3
"""
Fast unit tests for Damped Sine Synthesis (DSS) functionality.
Essential tests for core functionality, accuracy, and edge cases.
"""

import unittest
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from srs_damped_sine_synthesis import SRSSynthesizer
from srs_conversion import convert_srs


class TestSRSSynthesis(unittest.TestCase):
    """Test core SRS synthesis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = SRSSynthesizer()
        self.test_freq = np.array([100., 200.])        # 2 points (minimum required)
        self.test_accel = np.array([30., 40.])         # 2 points
        self.duration = 0.03                           # 30ms for speed
        self.sample_rate = 4096                        # Higher sample rate for stability
        self.damping = 0.05
    
    def test_synthesis_and_srs_accuracy(self):
        """Test basic synthesis and verify SRS calculation accuracy."""
        result = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, self.sample_rate,
            damping_ratio=self.damping, max_iterations=3  # Fast
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        required_keys = ['time', 'acceleration', 'srs_pos', 'synthesis_error', 'srs_freq']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Verify signal properties
        time_signal = result['acceleration']
        time = result['time']
        self.assertEqual(len(time_signal), len(time))
        self.assertGreater(len(time_signal), 10)  # Reasonable length
        
        # Verify synthesis error is reasonable (should be < 5 dB for simple case)
        self.assertLess(result['synthesis_error'], 5.0)
        
        # Verify SRS frequencies cover input range
        srs_freq = result['srs_freq']
        self.assertLessEqual(srs_freq[0], self.test_freq[0])
        self.assertGreaterEqual(srs_freq[-1], self.test_freq[-1])
        
        # Verify SRS values are positive and reasonable
        srs_pos = result['srs_pos']
        self.assertTrue(np.all(srs_pos > 0))
        self.assertTrue(np.all(srs_pos < 1000))  # Sanity check
    
    def test_fast_mode_equivalence(self):
        """Test that fast_mode produces identical results to standard mode."""
        np.random.seed(42)
        
        # Standard mode
        result_std = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, self.sample_rate,
            damping_ratio=self.damping, fast_mode=False, max_iterations=2
        )
        
        np.random.seed(42)  # Same random seed for fair comparison
        
        # Fast mode
        result_fast = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, self.sample_rate,
            damping_ratio=self.damping, fast_mode=True, max_iterations=2
        )
        
        # Results should be identical
        np.testing.assert_allclose(result_std['acceleration'], result_fast['acceleration'], 
                                   rtol=1e-10, atol=1e-10,
                                   err_msg="Fast mode should produce identical results")
        
        self.assertAlmostEqual(result_std['synthesis_error'], result_fast['synthesis_error'], 
                               places=10, msg="Synthesis errors should be identical")
    
    def test_input_validation_and_edge_cases(self):
        """Test input validation and edge cases."""
        
        # Test minimum configuration (2 points - what the system actually requires)
        min_freq = np.array([100.0, 200.0])
        min_accel = np.array([25.0, 35.0])
        result = self.synthesizer.synthesize_srs(
            min_freq, min_accel, self.duration, self.sample_rate,
            max_iterations=2
        )
        self.assertIsInstance(result, dict)
        self.assertIn('acceleration', result)
        
        # Test mismatched array lengths (should handle gracefully)
        with self.assertRaises((ValueError, AssertionError)):
            self.synthesizer.synthesize_srs(
                np.array([100., 200.]), np.array([30.]),  # Mismatched lengths
                self.duration, self.sample_rate, max_iterations=1
            )
        
        # Test zero duration (should fail)
        with self.assertRaises((ValueError, AssertionError, ZeroDivisionError)):
            self.synthesizer.synthesize_srs(
                self.test_freq, self.test_accel, 0.0, self.sample_rate, max_iterations=1
            )
    
    def test_signal_physics_properties(self):
        """Test that generated signals have reasonable physical properties."""
        result = self.synthesizer.synthesize_srs(
            self.test_freq, self.test_accel, self.duration, self.sample_rate,
            max_iterations=2  # Fast
        )
        
        accel = result['acceleration']
        time = result['time']
        dt = time[1] - time[0]
        
        # Test causality: signal should start and end near zero
        self.assertLess(abs(accel[0]), abs(accel).max() * 0.1)  # Start small
        self.assertLess(abs(accel[-1]), abs(accel).max() * 0.1)  # End small
        
        # Test energy conservation: RMS should be reasonable
        rms = np.sqrt(np.mean(accel**2))
        self.assertGreater(rms, 0.1)  # Should have some energy
        self.assertLess(rms, 1000)    # But not excessive
        
        # Test time vector is monotonic and properly spaced
        self.assertTrue(np.all(np.diff(time) > 0))  # Monotonic
        np.testing.assert_allclose(np.diff(time), dt, rtol=1e-10)  # Uniform spacing


class TestUnitConversions(unittest.TestCase):
    """Test SRS unit conversion functionality."""
    
    def test_conversion_accuracy_and_round_trips(self):
        """Test unit conversions work correctly and preserve values in round trips."""
        # Test data
        accel_g = np.array([10., 50., 100.])
        freq = np.array([10., 100., 1000.])
        
        # Test basic conversions
        accel_ms2 = convert_srs(accel_g, freq, "acceleration", "acceleration", "g", "m/s²")
        expected_ms2 = accel_g * 9.80665
        np.testing.assert_allclose(accel_ms2, expected_ms2, rtol=1e-10)
        
        # Test round-trip conversion preserves original values
        accel_back = convert_srs(accel_ms2, freq, "acceleration", "acceleration", "m/s²", "g")
        np.testing.assert_allclose(accel_back, accel_g, rtol=1e-10)
        
        # Test velocity conversion (use supported units)
        vel_direct = convert_srs(accel_g, freq, "acceleration", "velocity", "g", "in/sec")
        self.assertEqual(len(vel_direct), len(accel_g))
        self.assertTrue(np.all(vel_direct > 0))
        
        # Test invalid unit handling
        with self.assertRaises(ValueError):
            convert_srs(accel_g, freq, "acceleration", "acceleration", "invalid_unit", "m/s²")
    
    def test_wavelet_reconstruction(self):
        """Test wavelet reconstruction functionality if available."""
        synthesizer = SRSSynthesizer()
        
        # Try wavelet reconstruction - should either work or gracefully skip
        try:
            result = synthesizer.synthesize_srs(
                np.array([150., 300.]), np.array([25., 35.]), 0.03, 4096,
                fast_wavelet_mode=True, max_iterations=2, wavelet_trials=3
            )
            # If it works, should have basic structure
            self.assertIn('acceleration', result)
            self.assertIsInstance(result['synthesis_error'], (int, float))
        except (NotImplementedError, AttributeError):
            # If wavelet mode not implemented, should skip gracefully
            self.skipTest("Wavelet reconstruction not available")


if __name__ == '__main__':
    print("Damped Sine Synthesis (DSS) Essential Unit Tests")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    # Quick performance check
    print("\nPerformance Check:")
    print("=" * 20)
    synthesizer = SRSSynthesizer()
    
    import time
    start_time = time.time()
    result = synthesizer.synthesize_srs(
        np.array([200., 400.]), np.array([40., 50.]), 0.03, 4096,
        fast_mode=True, max_iterations=3
    )
    elapsed = time.time() - start_time
    
    print(f"Fast synthesis: {elapsed:.3f} seconds")
    print(f"Final error: {result['synthesis_error']:.3f} dB")
    print(f"Signal length: {len(result['acceleration'])} samples")
