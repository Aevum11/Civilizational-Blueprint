"""
Exception Theory Batch 7 Classes
Spectroscopy (Eq 71-80)

Complete spectroscopy implementation from ET primitives.
All derived from P∘D∘T manifold structure.

Classes:
- RydbergSeriesCalculator: General spectral series
- TransitionCalculator: Energy/wavelength/frequency
- WavelengthCalculator: Photon wavelengths
- FrequencyCalculator: Transition frequencies
- LymanSeries: UV series (n→1)
- BalmerSeries: Visible series (n→2)
- PaschenSeries: IR series (n→3)
- SelectionRules: Allowed transitions
- OscillatorStrength: Transition probabilities
- SpectralLineIntensity: Emission line calculator

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-18
Version: 3.1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from ..core.constants import (
    PLANCK_CONSTANT_H,
    PLANCK_CONSTANT_HBAR,
    SPEED_OF_LIGHT,
    RYDBERG_CONSTANT,
    RYDBERG_ENERGY,
)
from ..core.mathematics import ETMathV2


# =============================================================================
# Eq 71: RydbergSeriesCalculator - General spectral series
# =============================================================================

class RydbergSeriesCalculator:
    """
    Batch 7, Eq 71: Rydberg formula calculator.
    
    ET Math: 1/λ = R_∞(1/n₁² - 1/n₂²)
    
    R_∞ from manifold geometry constants.
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize for hydrogenic system.
        
        Args:
            nuclear_charge: Z (scales as Z²)
        """
        self.Z = nuclear_charge
        self.R_inf = RYDBERG_CONSTANT * (self.Z ** 2)  # Scaled for ion
    
    def wavelength(self, n1: int, n2: int, unit: str = 'nm') -> float:
        """
        Calculate λ for transition n2 → n1.
        
        Args:
            n1: Lower level
            n2: Upper level
            unit: 'm', 'nm', 'A' (Angstroms)
            
        Returns:
            Wavelength
        """
        if n2 <= n1:
            raise ValueError("n2 must be > n1")
        
        # Wavenumber
        wavenumber = self.R_inf * (1.0/(n1*n1) - 1.0/(n2*n2))
        
        if wavenumber == 0:
            return np.inf
        
        wavelength_m = 1.0 / wavenumber
        
        # Convert units
        if unit == 'm':
            return wavelength_m
        elif unit == 'nm':
            return wavelength_m * 1e9
        elif unit == 'A':
            return wavelength_m * 1e10
        else:
            raise ValueError("unit must be 'm', 'nm', or 'A'")
    
    def wavenumber(self, n1: int, n2: int) -> float:
        """
        Calculate 1/λ in m⁻¹.
        
        Args:
            n1: Lower level
            n2: Upper level
            
        Returns:
            Wavenumber
        """
        if n2 <= n1:
            raise ValueError("n2 must be > n1")
        
        return self.R_inf * (1.0/(n1*n1) - 1.0/(n2*n2))


# =============================================================================
# Eq 72: TransitionCalculator - Energy differences
# =============================================================================

class TransitionCalculator:
    """
    Batch 7, Eq 72: Transition energy calculator.
    
    ET Math: ΔE = E_final - E_initial = hf
    
    Energy conservation for photon emission/absorption.
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.Z = nuclear_charge
        self.Ry = RYDBERG_ENERGY * (self.Z ** 2)
        self.eV_to_J = 1.602176634e-19
    
    def energy_difference(self, n_initial: int, n_final: int, unit: str = 'eV') -> float:
        """
        Calculate ΔE = E_f - E_i.
        
        Args:
            n_initial: Initial state
            n_final: Final state
            unit: 'eV' or 'J'
            
        Returns:
            Energy difference (positive = absorption, negative = emission)
        """
        E_i = -self.Ry / (n_initial * n_initial)
        E_f = -self.Ry / (n_final * n_final)
        
        delta_E_eV = E_f - E_i
        
        if unit == 'eV':
            return delta_E_eV
        elif unit == 'J':
            return delta_E_eV * self.eV_to_J
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def photon_energy(self, n_initial: int, n_final: int, unit: str = 'eV') -> float:
        """
        Calculate photon energy |ΔE|.
        
        Args:
            n_initial: Initial state
            n_final: Final state
            unit: Energy unit
            
        Returns:
            Photon energy (positive)
        """
        return abs(self.energy_difference(n_initial, n_final, unit))
    
    def is_emission(self, n_initial: int, n_final: int) -> bool:
        """
        Check if transition is emission (n_i > n_f).
        
        Returns:
            True if emission, False if absorption
        """
        return n_initial > n_final


# =============================================================================
# Eq 73: WavelengthCalculator - Photon wavelengths
# =============================================================================

class WavelengthCalculator:
    """
    Batch 7, Eq 73: Wavelength calculator.
    
    ET Math: λ = hc/|ΔE|
    
    Photon wavelength from energy conservation.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.h = PLANCK_CONSTANT_H
        self.c = SPEED_OF_LIGHT
        self.hc = self.h * self.c
    
    def wavelength_from_energy(self, energy: float, energy_unit: str = 'eV',
                               wavelength_unit: str = 'nm') -> float:
        """
        Calculate λ = hc/E.
        
        Args:
            energy: Photon energy
            energy_unit: 'eV' or 'J'
            wavelength_unit: 'm', 'nm', 'A'
            
        Returns:
            Wavelength
        """
        # Convert energy to Joules
        if energy_unit == 'eV':
            E_J = energy * 1.602176634e-19
        elif energy_unit == 'J':
            E_J = energy
        else:
            raise ValueError("energy_unit must be 'eV' or 'J'")
        
        if E_J == 0:
            return np.inf
        
        wavelength_m = self.hc / E_J
        
        # Convert wavelength
        if wavelength_unit == 'm':
            return wavelength_m
        elif wavelength_unit == 'nm':
            return wavelength_m * 1e9
        elif wavelength_unit == 'A':
            return wavelength_m * 1e10
        else:
            raise ValueError("wavelength_unit must be 'm', 'nm', or 'A'")
    
    def energy_from_wavelength(self, wavelength: float, wavelength_unit: str = 'nm',
                               energy_unit: str = 'eV') -> float:
        """
        Calculate E = hc/λ.
        
        Args:
            wavelength: Photon wavelength
            wavelength_unit: 'm', 'nm', 'A'
            energy_unit: 'eV' or 'J'
            
        Returns:
            Energy
        """
        # Convert wavelength to meters
        if wavelength_unit == 'nm':
            lambda_m = wavelength * 1e-9
        elif wavelength_unit == 'A':
            lambda_m = wavelength * 1e-10
        elif wavelength_unit == 'm':
            lambda_m = wavelength
        else:
            raise ValueError("wavelength_unit must be 'm', 'nm', or 'A'")
        
        if lambda_m == 0:
            return np.inf
        
        E_J = self.hc / lambda_m
        
        if energy_unit == 'J':
            return E_J
        elif energy_unit == 'eV':
            return E_J / 1.602176634e-19
        else:
            raise ValueError("energy_unit must be 'eV' or 'J'")


# =============================================================================
# Eq 74: FrequencyCalculator - Transition frequencies
# =============================================================================

class FrequencyCalculator:
    """
    Batch 7, Eq 74: Frequency calculator.
    
    ET Math: f = |ΔE|/h = c/λ
    
    Photon frequency from energy.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.h = PLANCK_CONSTANT_H
        self.c = SPEED_OF_LIGHT
    
    def frequency_from_energy(self, energy: float, energy_unit: str = 'eV') -> float:
        """
        Calculate f = E/h.
        
        Args:
            energy: Photon energy
            energy_unit: 'eV' or 'J'
            
        Returns:
            Frequency in Hz
        """
        if energy_unit == 'eV':
            E_J = energy * 1.602176634e-19
        elif energy_unit == 'J':
            E_J = energy
        else:
            raise ValueError("energy_unit must be 'eV' or 'J'")
        
        return E_J / self.h
    
    def frequency_from_wavelength(self, wavelength: float, unit: str = 'nm') -> float:
        """
        Calculate f = c/λ.
        
        Args:
            wavelength: Photon wavelength
            unit: 'm', 'nm', 'A'
            
        Returns:
            Frequency in Hz
        """
        if unit == 'nm':
            lambda_m = wavelength * 1e-9
        elif unit == 'A':
            lambda_m = wavelength * 1e-10
        elif unit == 'm':
            lambda_m = wavelength
        else:
            raise ValueError("unit must be 'm', 'nm', or 'A'")
        
        if lambda_m == 0:
            return np.inf
        
        return self.c / lambda_m


# =============================================================================
# Eq 75: LymanSeries - UV series
# =============================================================================

class LymanSeries:
    """
    Batch 7, Eq 75: Lyman series (n→1, UV).
    
    ET Math: 1/λ = R_∞(1 - 1/n²)
    
    Lyman α (2→1): 121.6 nm
    Lyman limit (∞→1): 91.2 nm
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize Lyman series calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.series_calc = RydbergSeriesCalculator(nuclear_charge)
        self.n_lower = 1
    
    def wavelength(self, n_upper: int, unit: str = 'nm') -> float:
        """
        Calculate wavelength for n → 1 transition.
        
        Args:
            n_upper: Upper level (≥ 2)
            unit: Wavelength unit
            
        Returns:
            Wavelength
        """
        if n_upper < 2:
            raise ValueError("Lyman series requires n >= 2")
        
        return self.series_calc.wavelength(self.n_lower, n_upper, unit)
    
    def series_limit(self, unit: str = 'nm') -> float:
        """
        Calculate series limit (∞→1).
        
        Returns:
            Limit wavelength
        """
        # Use very large n as approximation
        return self.series_calc.wavelength(self.n_lower, 1000, unit)
    
    def all_lines(self, n_max: int = 10, unit: str = 'nm') -> Dict[str, float]:
        """
        Calculate all Lyman lines up to n_max.
        
        Returns:
            Dictionary {line_name: wavelength}
        """
        greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ']
        
        lines = {}
        for n in range(2, min(n_max + 1, 2 + len(greek_letters))):
            name = f"Lyman {greek_letters[n-2]}"
            lines[name] = self.wavelength(n, unit)
        
        return lines


# =============================================================================
# Eq 76: BalmerSeries - Visible series
# =============================================================================

class BalmerSeries:
    """
    Batch 7, Eq 76: Balmer series (n→2, visible).
    
    ET Math: 1/λ = R_∞(1/4 - 1/n²)
    
    Hα (3→2): 656.3 nm (red)
    Hβ (4→2): 486.1 nm (blue-green)
    Hγ (5→2): 434.0 nm (blue)
    Hδ (6→2): 410.2 nm (violet)
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize Balmer series calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.series_calc = RydbergSeriesCalculator(nuclear_charge)
        self.n_lower = 2
    
    def wavelength(self, n_upper: int, unit: str = 'nm') -> float:
        """
        Calculate wavelength for n → 2 transition.
        
        Args:
            n_upper: Upper level (≥ 3)
            unit: Wavelength unit
            
        Returns:
            Wavelength
        """
        if n_upper < 3:
            raise ValueError("Balmer series requires n >= 3")
        
        return self.series_calc.wavelength(self.n_lower, n_upper, unit)
    
    def series_limit(self, unit: str = 'nm') -> float:
        """
        Calculate series limit (∞→2).
        
        Returns:
            Limit wavelength
        """
        return self.series_calc.wavelength(self.n_lower, 1000, unit)
    
    def all_lines(self, n_max: int = 10, unit: str = 'nm') -> Dict[str, float]:
        """
        Calculate Balmer lines (Hα, Hβ, Hγ, Hδ, ...).
        
        Returns:
            Dictionary {line_name: wavelength}
        """
        h_names = ['Hα', 'Hβ', 'Hγ', 'Hδ', 'Hε', 'Hζ', 'Hη', 'Hθ']
        
        lines = {}
        for n in range(3, min(n_max + 1, 3 + len(h_names))):
            name = h_names[n-3]
            lines[name] = self.wavelength(n, unit)
        
        return lines


# =============================================================================
# Eq 77: PaschenSeries - IR series
# =============================================================================

class PaschenSeries:
    """
    Batch 7, Eq 77: Paschen series (n→3, IR).
    
    ET Math: 1/λ = R_∞(1/9 - 1/n²)
    
    Near-infrared transitions.
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize Paschen series calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.series_calc = RydbergSeriesCalculator(nuclear_charge)
        self.n_lower = 3
    
    def wavelength(self, n_upper: int, unit: str = 'nm') -> float:
        """
        Calculate wavelength for n → 3 transition.
        
        Args:
            n_upper: Upper level (≥ 4)
            unit: Wavelength unit
            
        Returns:
            Wavelength
        """
        if n_upper < 4:
            raise ValueError("Paschen series requires n >= 4")
        
        return self.series_calc.wavelength(self.n_lower, n_upper, unit)
    
    def series_limit(self, unit: str = 'nm') -> float:
        """
        Calculate series limit (∞→3).
        
        Returns:
            Limit wavelength
        """
        return self.series_calc.wavelength(self.n_lower, 1000, unit)
    
    def all_lines(self, n_max: int = 10, unit: str = 'nm') -> Dict[str, float]:
        """
        Calculate all Paschen lines.
        
        Returns:
            Dictionary {line_name: wavelength}
        """
        lines = {}
        for n in range(4, n_max + 1):
            name = f"Paschen {n}→3"
            lines[name] = self.wavelength(n, unit)
        
        return lines


# =============================================================================
# Eq 78: SelectionRules - Allowed transitions
# =============================================================================

class SelectionRules:
    """
    Batch 7, Eq 78: Electric dipole selection rules.
    
    ET Math: Δl = ±1 (required)
             Δn = any
    
    Manifold symmetry constraints.
    Photon carries L = 1 (spin-1 boson).
    """
    
    @staticmethod
    def electric_dipole_allowed(l_initial: int, l_final: int) -> bool:
        """
        Check if electric dipole transition is allowed.
        
        ET Math: Δl = ±1
        
        Args:
            l_initial: Initial orbital quantum number
            l_final: Final orbital quantum number
            
        Returns:
            True if allowed
        """
        delta_l = l_final - l_initial
        
        return abs(delta_l) == 1
    
    @staticmethod
    def magnetic_dipole_allowed(l_initial: int, l_final: int) -> bool:
        """
        Check if magnetic dipole transition is allowed.
        
        ET Math: Δl = 0 (but both can't be l=0)
        
        Args:
            l_initial, l_final: Quantum numbers
            
        Returns:
            True if allowed
        """
        if l_initial == 0 and l_final == 0:
            return False
        
        return l_initial == l_final
    
    @staticmethod
    def transition_type(l_initial: int, l_final: int) -> str:
        """
        Determine transition type.
        
        Returns:
            'E1' (electric dipole), 'M1' (magnetic dipole), 'forbidden', or 'higher-order'
        """
        if SelectionRules.electric_dipole_allowed(l_initial, l_final):
            return 'E1'
        elif SelectionRules.magnetic_dipole_allowed(l_initial, l_final):
            return 'M1'
        elif l_initial == l_final:
            return 'forbidden'
        else:
            return 'higher-order'


# =============================================================================
# Eq 79: OscillatorStrength - Transition probabilities
# =============================================================================

class OscillatorStrength:
    """
    Batch 7, Eq 79: Oscillator strength calculator.
    
    ET Math: f_if ∝ |⟨ψ_f|r|ψ_i⟩|² × (E_f - E_i)
    
    Transition probability weight.
    """
    
    @staticmethod
    def simple_approximation(n_initial: int, n_final: int) -> float:
        """
        Simplified oscillator strength.
        
        Rough approximation: f ∝ 1/(n_i² - n_f²)²
        
        Args:
            n_initial: Initial level
            n_final: Final level
            
        Returns:
            Relative oscillator strength
        """
        if n_final <= n_initial:
            return 0.0
        
        delta_n_sq = n_final * n_final - n_initial * n_initial
        
        if delta_n_sq == 0:
            return 0.0
        
        return 1.0 / (delta_n_sq * delta_n_sq)
    
    @staticmethod
    def sum_rule_check(oscillator_strengths: List[float]) -> float:
        """
        Check Thomas-Reiche-Kuhn sum rule: Σf = 1.
        
        Args:
            oscillator_strengths: List of f values
            
        Returns:
            Sum (should be ≈ 1)
        """
        return sum(oscillator_strengths)


# =============================================================================
# Eq 80: SpectralLineIntensity - Emission line calculator
# =============================================================================

class SpectralLineIntensity:
    """
    Batch 7, Eq 80: Spectral line intensity calculator.
    
    ET Math: I ∝ N_i × A_if × hf
    
    where N_i = population, A_if = Einstein A coefficient.
    """
    
    def __init__(self, temperature: Optional[float] = None):
        """
        Initialize intensity calculator.
        
        Args:
            temperature: Gas temperature in K (optional, for Boltzmann)
        """
        self.temperature = temperature
        self.k_B = 1.380649e-23  # Boltzmann constant
    
    def relative_intensity(self, n_initial: int, n_final: int, 
                          population: float = 1.0) -> float:
        """
        Calculate relative line intensity.
        
        ET Math: I ∝ N × f × ν
        
        Args:
            n_initial: Initial level
            n_final: Final level
            population: Population in initial state
            
        Returns:
            Relative intensity
        """
        # Frequency
        freq_calc = FrequencyCalculator()
        trans_calc = TransitionCalculator()
        
        energy = trans_calc.photon_energy(n_initial, n_final, 'J')
        freq = freq_calc.frequency_from_energy(energy, 'J')
        
        # Oscillator strength (proxy for Einstein A)
        f_if = OscillatorStrength.simple_approximation(n_final, n_initial)
        
        # I ∝ N × f × ν
        return population * f_if * freq
    
    def boltzmann_population(self, n: int, ground_population: float = 1.0) -> float:
        """
        Calculate population using Boltzmann distribution.
        
        ET Math: N_n ∝ exp(-E_n/k_BT)
        
        Args:
            n: Energy level
            ground_population: N₁ (reference)
            
        Returns:
            Population N_n
        """
        if self.temperature is None:
            raise ValueError("Temperature not set")
        
        trans_calc = TransitionCalculator()
        E_n = trans_calc.photon_energy(1, n, 'J')
        
        return ground_population * np.exp(-E_n / (self.k_B * self.temperature))


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'RydbergSeriesCalculator',
    'TransitionCalculator',
    'WavelengthCalculator',
    'FrequencyCalculator',
    'LymanSeries',
    'BalmerSeries',
    'PaschenSeries',
    'SelectionRules',
    'OscillatorStrength',
    'SpectralLineIntensity',
]
