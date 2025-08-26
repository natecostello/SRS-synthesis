#!/usr/bin/env python3
"""
SRS Unit Conversion Module

This module provides functions to convert between different SRS units:
- Acceleration SRS (G or m/s²)
- Velocity SRS (in/sec or m/sec)  
- Displacement SRS (inches or mm)

The conversions are based on the relationship between acceleration, velocity,
and displacement for a single degree of freedom (SDOF) system at resonance.

For an SDOF system with natural frequency fn and damping ratio ζ:
- Velocity SRS = Acceleration SRS / (2π * fn)
- Displacement SRS = Acceleration SRS / (2π * fn)²
- Acceleration SRS = Velocity SRS * (2π * fn)
- Acceleration SRS = Displacement SRS * (2π * fn)²

Author: SRS Synthesis Module
Date: August 2025
"""

import numpy as np
from typing import Union, Tuple, Literal
from enum import Enum

class SRSUnits(Enum):
    """Enumeration of supported SRS units."""
    ACCEL_G = "g"
    ACCEL_MS2 = "m/s²"
    VEL_INSEC = "in/sec"
    VEL_MSEC = "m/sec"
    DISP_IN = "inches"
    DISP_MM = "mm"

class SRSConverter:
    """
    Class for converting between different SRS units.
    
    This class provides methods to convert SRS values between acceleration,
    velocity, and displacement units, accounting for the natural frequencies
    of the SDOF systems.
    """
    
    # Unit conversion factors
    G_TO_MS2 = 9.80665  # Standard gravity in m/s²
    IN_TO_M = 0.0254    # Inches to meters
    MM_TO_M = 0.001     # Millimeters to meters
    
    def __init__(self):
        """Initialize the SRS converter."""
        pass
    
    def accel_to_velocity(self, 
                         accel_srs: np.ndarray, 
                         frequency: np.ndarray,
                         input_units: Literal["g", "m/s²"] = "g",
                         output_units: Literal["in/sec", "m/sec"] = "in/sec") -> np.ndarray:
        """
        Convert acceleration SRS to velocity SRS.
        
        Parameters:
        -----------
        accel_srs : np.ndarray
            Acceleration SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        input_units : str, default="g"
            Units of input acceleration SRS ("g" or "m/s²")
        output_units : str, default="in/sec"
            Units of output velocity SRS ("in/sec" or "m/sec")
            
        Returns:
        --------
        np.ndarray
            Velocity SRS values in specified units
        """
        # Convert input acceleration to m/s²
        if input_units == "g":
            accel_ms2 = accel_srs * self.G_TO_MS2
        elif input_units == "m/s²":
            accel_ms2 = accel_srs.copy()
        else:
            raise ValueError(f"Unsupported input units: {input_units}")
        
        # Convert to velocity SRS in m/sec
        omega = 2 * np.pi * frequency
        vel_msec = accel_ms2 / omega
        
        # Convert to output units
        if output_units == "m/sec":
            return vel_msec
        elif output_units == "in/sec":
            return vel_msec / self.IN_TO_M
        else:
            raise ValueError(f"Unsupported output units: {output_units}")
    
    def accel_to_displacement(self, 
                            accel_srs: np.ndarray, 
                            frequency: np.ndarray,
                            input_units: Literal["g", "m/s²"] = "g",
                            output_units: Literal["inches", "mm"] = "inches") -> np.ndarray:
        """
        Convert acceleration SRS to displacement SRS.
        
        Parameters:
        -----------
        accel_srs : np.ndarray
            Acceleration SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        input_units : str, default="g"
            Units of input acceleration SRS ("g" or "m/s²")
        output_units : str, default="inches"
            Units of output displacement SRS ("inches" or "mm")
            
        Returns:
        --------
        np.ndarray
            Displacement SRS values in specified units
        """
        # Convert input acceleration to m/s²
        if input_units == "g":
            accel_ms2 = accel_srs * self.G_TO_MS2
        elif input_units == "m/s²":
            accel_ms2 = accel_srs.copy()
        else:
            raise ValueError(f"Unsupported input units: {input_units}")
        
        # Convert to displacement SRS in meters
        omega = 2 * np.pi * frequency
        disp_m = accel_ms2 / (omega**2)
        
        # Convert to output units
        if output_units == "inches":
            return disp_m / self.IN_TO_M
        elif output_units == "mm":
            return disp_m / self.MM_TO_M
        else:
            raise ValueError(f"Unsupported output units: {output_units}")
    
    def velocity_to_accel(self, 
                         vel_srs: np.ndarray, 
                         frequency: np.ndarray,
                         input_units: Literal["in/sec", "m/sec"] = "in/sec",
                         output_units: Literal["g", "m/s²"] = "g") -> np.ndarray:
        """
        Convert velocity SRS to acceleration SRS.
        
        Parameters:
        -----------
        vel_srs : np.ndarray
            Velocity SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        input_units : str, default="in/sec"
            Units of input velocity SRS ("in/sec" or "m/sec")
        output_units : str, default="g"
            Units of output acceleration SRS ("g" or "m/s²")
            
        Returns:
        --------
        np.ndarray
            Acceleration SRS values in specified units
        """
        # Convert input velocity to m/sec
        if input_units == "in/sec":
            vel_msec = vel_srs * self.IN_TO_M
        elif input_units == "m/sec":
            vel_msec = vel_srs.copy()
        else:
            raise ValueError(f"Unsupported input units: {input_units}")
        
        # Convert to acceleration SRS in m/s²
        omega = 2 * np.pi * frequency
        accel_ms2 = vel_msec * omega
        
        # Convert to output units
        if output_units == "g":
            return accel_ms2 / self.G_TO_MS2
        elif output_units == "m/s²":
            return accel_ms2
        else:
            raise ValueError(f"Unsupported output units: {output_units}")
    
    def velocity_to_displacement(self, 
                               vel_srs: np.ndarray, 
                               frequency: np.ndarray,
                               input_units: Literal["in/sec", "m/sec"] = "in/sec",
                               output_units: Literal["inches", "mm"] = "inches") -> np.ndarray:
        """
        Convert velocity SRS to displacement SRS.
        
        Parameters:
        -----------
        vel_srs : np.ndarray
            Velocity SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        input_units : str, default="in/sec"
            Units of input velocity SRS ("in/sec" or "m/sec")
        output_units : str, default="inches"
            Units of output displacement SRS ("inches" or "mm")
            
        Returns:
        --------
        np.ndarray
            Displacement SRS values in specified units
        """
        # Convert input velocity to m/sec
        if input_units == "in/sec":
            vel_msec = vel_srs * self.IN_TO_M
        elif input_units == "m/sec":
            vel_msec = vel_srs.copy()
        else:
            raise ValueError(f"Unsupported input units: {input_units}")
        
        # Convert to displacement SRS in meters
        omega = 2 * np.pi * frequency
        disp_m = vel_msec / omega
        
        # Convert to output units
        if output_units == "inches":
            return disp_m / self.IN_TO_M
        elif output_units == "mm":
            return disp_m / self.MM_TO_M
        else:
            raise ValueError(f"Unsupported output units: {output_units}")
    
    def displacement_to_accel(self, 
                            disp_srs: np.ndarray, 
                            frequency: np.ndarray,
                            input_units: Literal["inches", "mm"] = "inches",
                            output_units: Literal["g", "m/s²"] = "g") -> np.ndarray:
        """
        Convert displacement SRS to acceleration SRS.
        
        Parameters:
        -----------
        disp_srs : np.ndarray
            Displacement SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        input_units : str, default="inches"
            Units of input displacement SRS ("inches" or "mm")
        output_units : str, default="g"
            Units of output acceleration SRS ("g" or "m/s²")
            
        Returns:
        --------
        np.ndarray
            Acceleration SRS values in specified units
        """
        # Convert input displacement to meters
        if input_units == "inches":
            disp_m = disp_srs * self.IN_TO_M
        elif input_units == "mm":
            disp_m = disp_srs * self.MM_TO_M
        else:
            raise ValueError(f"Unsupported input units: {input_units}")
        
        # Convert to acceleration SRS in m/s²
        omega = 2 * np.pi * frequency
        accel_ms2 = disp_m * (omega**2)
        
        # Convert to output units
        if output_units == "g":
            return accel_ms2 / self.G_TO_MS2
        elif output_units == "m/s²":
            return accel_ms2
        else:
            raise ValueError(f"Unsupported output units: {output_units}")
    
    def displacement_to_velocity(self, 
                               disp_srs: np.ndarray, 
                               frequency: np.ndarray,
                               input_units: Literal["inches", "mm"] = "inches",
                               output_units: Literal["in/sec", "m/sec"] = "in/sec") -> np.ndarray:
        """
        Convert displacement SRS to velocity SRS.
        
        Parameters:
        -----------
        disp_srs : np.ndarray
            Displacement SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        input_units : str, default="inches"
            Units of input displacement SRS ("inches" or "mm")
        output_units : str, default="in/sec"
            Units of output velocity SRS ("in/sec" or "m/sec")
            
        Returns:
        --------
        np.ndarray
            Velocity SRS values in specified units
        """
        # Convert input displacement to meters
        if input_units == "inches":
            disp_m = disp_srs * self.IN_TO_M
        elif input_units == "mm":
            disp_m = disp_srs * self.MM_TO_M
        else:
            raise ValueError(f"Unsupported input units: {input_units}")
        
        # Convert to velocity SRS in m/sec
        omega = 2 * np.pi * frequency
        vel_msec = disp_m * omega
        
        # Convert to output units
        if output_units == "m/sec":
            return vel_msec
        elif output_units == "in/sec":
            return vel_msec / self.IN_TO_M
        else:
            raise ValueError(f"Unsupported output units: {output_units}")
    
    def convert_srs(self, 
                   srs_values: np.ndarray, 
                   frequency: np.ndarray,
                   from_type: Literal["acceleration", "velocity", "displacement"],
                   to_type: Literal["acceleration", "velocity", "displacement"],
                   from_units: str,
                   to_units: str) -> np.ndarray:
        """
        General purpose SRS unit conversion function.
        
        Parameters:
        -----------
        srs_values : np.ndarray
            Input SRS values
        frequency : np.ndarray
            Natural frequencies (Hz) corresponding to SRS values
        from_type : str
            Type of input SRS ("acceleration", "velocity", "displacement")
        to_type : str
            Type of output SRS ("acceleration", "velocity", "displacement")
        from_units : str
            Units of input SRS
        to_units : str
            Units of output SRS
            
        Returns:
        --------
        np.ndarray
            Converted SRS values
        """
        # Direct conversion mapping
        conversion_map = {
            ("acceleration", "velocity"): self.accel_to_velocity,
            ("acceleration", "displacement"): self.accel_to_displacement,
            ("velocity", "acceleration"): self.velocity_to_accel,
            ("velocity", "displacement"): self.velocity_to_displacement,
            ("displacement", "acceleration"): self.displacement_to_accel,
            ("displacement", "velocity"): self.displacement_to_velocity,
        }
        
        # Handle same-type conversions (unit conversions only)
        if from_type == to_type:
            if from_type == "acceleration":
                if from_units == to_units:
                    return srs_values.copy()
                elif from_units == "g" and to_units == "m/s²":
                    return srs_values * self.G_TO_MS2
                elif from_units == "m/s²" and to_units == "g":
                    return srs_values / self.G_TO_MS2
                else:
                    raise ValueError(f"Unsupported unit conversion: {from_units} to {to_units}")
            
            elif from_type == "velocity":
                if from_units == to_units:
                    return srs_values.copy()
                elif from_units == "in/sec" and to_units == "m/sec":
                    return srs_values * self.IN_TO_M
                elif from_units == "m/sec" and to_units == "in/sec":
                    return srs_values / self.IN_TO_M
                else:
                    raise ValueError(f"Unsupported unit conversion: {from_units} to {to_units}")
            
            elif from_type == "displacement":
                if from_units == to_units:
                    return srs_values.copy()
                elif from_units == "inches" and to_units == "mm":
                    return srs_values * self.IN_TO_M / self.MM_TO_M
                elif from_units == "mm" and to_units == "inches":
                    return srs_values * self.MM_TO_M / self.IN_TO_M
                else:
                    raise ValueError(f"Unsupported unit conversion: {from_units} to {to_units}")
        
        # Cross-type conversions
        key = (from_type, to_type)
        if key in conversion_map:
            return conversion_map[key](srs_values, frequency, from_units, to_units)
        else:
            raise ValueError(f"Unsupported conversion: {from_type} to {to_type}")


# Convenience functions for direct use
def convert_srs(srs_values: np.ndarray, 
                frequency: np.ndarray,
                from_type: str,
                to_type: str,
                from_units: str,
                to_units: str) -> np.ndarray:
    """
    Convenience function to convert SRS between different types and units.
    
    Parameters:
    -----------
    srs_values : np.ndarray
        Input SRS values
    frequency : np.ndarray
        Natural frequencies (Hz) corresponding to SRS values
    from_type : str
        Type of input SRS ("acceleration", "velocity", "displacement")
    to_type : str
        Type of output SRS ("acceleration", "velocity", "displacement")
    from_units : str
        Units of input SRS
    to_units : str
        Units of output SRS
        
    Returns:
    --------
    np.ndarray
        Converted SRS values
        
    Example:
    --------
    >>> import numpy as np
    >>> from srs_conversion import convert_srs
    >>> 
    >>> # Convert acceleration SRS (G) to velocity SRS (in/sec)
    >>> freq = np.array([10, 100, 1000])  # Hz
    >>> accel_g = np.array([10, 50, 50])  # G
    >>> vel_insec = convert_srs(accel_g, freq, "acceleration", "velocity", "g", "in/sec")
    >>> print(f"Velocity SRS: {vel_insec}")
    """
    converter = SRSConverter()
    return converter.convert_srs(srs_values, frequency, from_type, to_type, from_units, to_units)


def accel_to_velocity_srs(accel_srs: np.ndarray, 
                         frequency: np.ndarray,
                         input_units: str = "g",
                         output_units: str = "in/sec") -> np.ndarray:
    """Shorthand function to convert acceleration SRS to velocity SRS."""
    converter = SRSConverter()
    return converter.accel_to_velocity(accel_srs, frequency, input_units, output_units)


def accel_to_displacement_srs(accel_srs: np.ndarray, 
                            frequency: np.ndarray,
                            input_units: str = "g",
                            output_units: str = "inches") -> np.ndarray:
    """Shorthand function to convert acceleration SRS to displacement SRS."""
    converter = SRSConverter()
    return converter.accel_to_displacement(accel_srs, frequency, input_units, output_units)


def velocity_to_accel_srs(vel_srs: np.ndarray, 
                         frequency: np.ndarray,
                         input_units: str = "in/sec",
                         output_units: str = "g") -> np.ndarray:
    """Shorthand function to convert velocity SRS to acceleration SRS."""
    converter = SRSConverter()
    return converter.velocity_to_accel(vel_srs, frequency, input_units, output_units)


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    
    print("SRS Conversion Module - Example Usage")
    print("=" * 50)
    
    # Example SRS specification
    frequency = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])  # Hz
    accel_g = np.array([5, 10, 20, 50, 80, 100, 80, 60, 40])  # G
    
    converter = SRSConverter()
    
    # Convert to different units
    accel_ms2 = converter.convert_srs(accel_g, frequency, "acceleration", "acceleration", "g", "m/s²")
    vel_insec = converter.accel_to_velocity(accel_g, frequency, "g", "in/sec")
    vel_msec = converter.accel_to_velocity(accel_g, frequency, "g", "m/sec")
    disp_in = converter.accel_to_displacement(accel_g, frequency, "g", "inches")
    disp_mm = converter.accel_to_displacement(accel_g, frequency, "g", "mm")
    
    # Print results
    print(f"Frequency (Hz):        {frequency}")
    print(f"Acceleration (G):      {accel_g}")
    print(f"Acceleration (m/s²):   {np.round(accel_ms2, 1)}")
    print(f"Velocity (in/sec):     {np.round(vel_insec, 3)}")
    print(f"Velocity (m/sec):      {np.round(vel_msec, 3)}")
    print(f"Displacement (in):     {np.round(disp_in, 6)}")
    print(f"Displacement (mm):     {np.round(disp_mm, 3)}")
    
    # Test round-trip conversion
    print(f"\nRound-trip conversion test:")
    accel_back = converter.velocity_to_accel(vel_insec, frequency, "in/sec", "g")
    max_error = np.max(np.abs(accel_back - accel_g))
    print(f"Original accel (G):    {accel_g}")
    print(f"Round-trip accel (G):  {np.round(accel_back, 6)}")
    print(f"Max error:             {max_error:.2e} G")
    
    if max_error < 1e-10:
        print("✅ Round-trip conversion successful!")
    else:
        print("❌ Round-trip conversion failed!")
    
    print(f"\nSRS Conversion Module ready for use!")
