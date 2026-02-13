"""
Exception Theory Batch 8 Classes
Fine Structure & Corrections (Eq 81-90)

Complete fine structure implementation from ET primitives.
All derived from P∘D∘T manifold structure.

Classes:
- SpinOrbitCoupling: L·S interaction calculator
- RelativisticCorrection: Kinetic energy corrections
- FineStructureShift: Total α² corrections
- LambShiftCalculator: QED vacuum effects
- HyperfineSplitting: Nuclear spin coupling
- Hydrogen21cmLine: Famous 21 cm transition
- AngularMomentumCoupler: j = l + s validator
- ZeemanEffect: Magnetic field splitting
- StarkEffect: Electric field shifts
- IsotopeShift: Mass-dependent corrections

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-18
Version: 3.1.0
"""

import numpy as np
from typing import Tuple, Optional

from ..core.constants import (
    PLANCK_CONSTANT_HBAR,
    PLANCK_CONSTANT_H,
    ELEMENTARY_CHARGE,
    ELECTRON_MASS,
    PROTON_MASS,
    FINE_STRUCTURE_CONSTANT,
    BOHR_RADIUS,
    RYDBERG_ENERGY,
    LAMB_SHIFT_2S,
    HYDROGEN_21CM_FREQUENCY,
    HYDROGEN_21CM_WAVELENGTH,
    SPEED_OF_LIGHT,
)
from ..core.mathematics import ETMathV2


# =============================================================================
# Eq 81: SpinOrbitCoupling - L·S interaction
# =============================================================================

class SpinOrbitCoupling:
    """
    Batch 8, Eq 81: Spin-orbit coupling calculator.
    
    ET Math: ΔE_so ∝ α² × E_n × [j(j+1) - l(l+1) - s(s+1)]
    
    Coupling between orbital (L) and spin (S) descriptors.
    O(α²) ≈ 10⁻⁵ correction.
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.Z = nuclear_charge
        self.alpha = FINE_STRUCTURE_CONSTANT
        self.alpha_sq = self.alpha ** 2
        self.Ry = RYDBERG_ENERGY * (self.Z ** 2)
        self.eV_to_J = 1.602176634e-19
    
    def energy_shift(self, n: int, l: int, j: float, unit: str = 'eV') -> float:
        """
        Calculate ΔE_so.
        
        ET Math: Descriptor coupling through manifold geometry
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            j: Total angular momentum (l ± 1/2)
            unit: 'eV' or 'J'
            
        Returns:
            Spin-orbit energy shift
        """
        if l == 0:
            return 0.0  # No spin-orbit for s orbitals
        
        if l < 0 or l >= n:
            raise ValueError("Invalid quantum numbers")
        
        # Base energy E_n = -Z²Ry/n²
        E_n_eV = -self.Ry / (n * n)
        
        # Spin s = 1/2
        s = 0.5
        
        # j(j+1) - l(l+1) - s(s+1)
        factor = j * (j + 1) - l * (l + 1) - s * (s + 1)
        
        # ΔE_so ∝ α² × E_n × factor / [n × l × (l + 0.5) × (l + 1)]
        if l > 0:
            delta_eV = E_n_eV * self.alpha_sq * factor / (n * l * (l + 0.5) * (l + 1))
        else:
            delta_eV = 0.0
        
        if unit == 'eV':
            return delta_eV
        elif unit == 'J':
            return delta_eV * self.eV_to_J
        else:
            raise ValueError("unit must be 'eV' or 'J'")


# =============================================================================
# Eq 82: RelativisticCorrection - Kinetic energy
# =============================================================================

class RelativisticCorrection:
    """
    Batch 8, Eq 82: Relativistic kinetic energy correction.
    
    ET Math: T = √(p²c² + m²c⁴) - mc²
              ≈ p²/(2m) - p⁴/(8m³c²)
    
    Second term is relativistic correction, O(α²).
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.Z = nuclear_charge
        self.alpha = FINE_STRUCTURE_CONSTANT
        self.alpha_sq = self.alpha ** 2
        self.Ry = RYDBERG_ENERGY * (self.Z ** 2)
        self.eV_to_J = 1.602176634e-19
    
    def energy_shift(self, n: int, l: int, unit: str = 'eV') -> float:
        """
        Calculate ΔE_rel.
        
        ET Math: Descriptor gradient approaching v_T,max (c)
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            unit: 'eV' or 'J'
            
        Returns:
            Relativistic correction
        """
        # Base energy
        E_n_eV = -self.Ry / (n * n)
        
        # ΔE_rel = -E_n × α² × (n/(l+0.5) - 3/4) / n²
        correction_eV = -E_n_eV * self.alpha_sq * (n / (l + 0.5) - 0.75) / (n * n)
        
        if unit == 'eV':
            return correction_eV
        elif unit == 'J':
            return correction_eV * self.eV_to_J
        else:
            raise ValueError("unit must be 'eV' or 'J'")


# =============================================================================
# Eq 83: FineStructureShift - Total corrections
# =============================================================================

class FineStructureShift:
    """
    Batch 8, Eq 83: Total fine structure shift.
    
    ET Math: ΔE_fs = ΔE_so + ΔE_rel
    
    Combines spin-orbit + relativistic corrections.
    Both O(α²).
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize calculator.
        
        Args:
            nuclear_charge: Z
        """
        self.so_calc = SpinOrbitCoupling(nuclear_charge)
        self.rel_calc = RelativisticCorrection(nuclear_charge)
    
    def total_shift(self, n: int, l: int, j: float, unit: str = 'eV') -> float:
        """
        Calculate total fine structure shift.
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            j: Total angular momentum
            unit: 'eV' or 'J'
            
        Returns:
            Total ΔE_fs
        """
        delta_so = self.so_calc.energy_shift(n, l, j, unit)
        delta_rel = self.rel_calc.energy_shift(n, l, unit)
        
        return delta_so + delta_rel
    
    def corrected_energy(self, n: int, l: int, j: float, unit: str = 'eV') -> float:
        """
        Calculate E_n + ΔE_fs.
        
        Args:
            n, l, j: Quantum numbers
            unit: Energy unit
            
        Returns:
            Corrected energy level
        """
        from ..classes.batch6 import HydrogenEnergyLevels
        
        energy_calc = HydrogenEnergyLevels(self.so_calc.Z)
        E_n = energy_calc.energy(n, unit)
        delta_fs = self.total_shift(n, l, j, unit)
        
        return E_n + delta_fs


# =============================================================================
# Eq 84: LambShiftCalculator - QED corrections
# =============================================================================

class LambShiftCalculator:
    """
    Batch 8, Eq 84: Lamb shift (QED correction).
    
    ET Math: ΔE_Lamb ≈ (α⁵/π) × m_e c² / n³ × δ_l0
    
    Quantum vacuum fluctuations.
    Manifold variance (BASE_VARIANCE = 1/12) allows (P∘D) fluctuations.
    O(α⁵) effect, tiny but measurable.
    
    2s₁/₂ - 2p₁/₂: 1057 MHz
    """
    
    def __init__(self):
        """Initialize with measured Lamb shift."""
        self.lamb_2s_freq = LAMB_SHIFT_2S  # Hz
        self.h = PLANCK_CONSTANT_H
    
    def energy_shift(self, n: int, l: int, unit: str = 'eV') -> float:
        """
        Calculate Lamb shift for state |nl⟩.
        
        Only s orbitals (l=0) have significant shift.
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            unit: 'eV' or 'J'
            
        Returns:
            Lamb shift energy
        """
        if l != 0:
            return 0.0  # Only s orbitals
        
        # For 2s, use measured value
        if n == 2:
            E_J = self.lamb_2s_freq * self.h
        else:
            # Scale approximately as 1/n³
            lamb_2s_J = self.lamb_2s_freq * self.h
            E_J = lamb_2s_J * (8.0 / (n * n * n))
        
        if unit == 'J':
            return E_J
        elif unit == 'eV':
            return E_J / 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def frequency_shift(self, n: int, l: int) -> float:
        """
        Calculate Lamb shift as frequency.
        
        Args:
            n, l: Quantum numbers
            
        Returns:
            Frequency shift in Hz
        """
        E_J = self.energy_shift(n, l, 'J')
        
        return E_J / self.h


# =============================================================================
# Eq 85: HyperfineSplitting - Nuclear spin coupling
# =============================================================================

class HyperfineSplitting:
    """
    Batch 8, Eq 85: Hyperfine structure splitting.
    
    ET Math: ΔE_hfs = A × [F(F+1) - I(I+1) - J(J+1)]
    
    Nuclear spin (I) coupling with electron (J).
    For ground state (1s): F = 0 or 1.
    """
    
    def __init__(self, nuclear_spin: float = 0.5):
        """
        Initialize calculator.
        
        Args:
            nuclear_spin: I (1/2 for proton)
        """
        self.I = nuclear_spin
        self.h = PLANCK_CONSTANT_H
        
        # 21 cm line energy (F=1 ↔ F=0)
        self.E_21cm = HYDROGEN_21CM_FREQUENCY * self.h
        
        # Hyperfine constant A
        self.A = self.E_21cm / 2.0  # For ΔF = 1
    
    def energy_shift(self, F: float, J: float = 0.5, unit: str = 'eV') -> float:
        """
        Calculate hyperfine shift.
        
        Args:
            F: Total angular momentum (electron + nuclear)
            J: Electron angular momentum (default 1/2 for ground state)
            unit: 'eV' or 'J'
            
        Returns:
            Hyperfine energy
        """
        # ΔE = A × [F(F+1) - I(I+1) - J(J+1)]
        factor = F * (F + 1) - self.I * (self.I + 1) - J * (J + 1)
        
        E_J = self.A * factor
        
        if unit == 'J':
            return E_J
        elif unit == 'eV':
            return E_J / 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def splitting_frequency(self, F_upper: float, F_lower: float, J: float = 0.5) -> float:
        """
        Calculate frequency of F_upper ↔ F_lower transition.
        
        Args:
            F_upper, F_lower: Total angular momentum values
            J: Electron angular momentum
            
        Returns:
            Frequency in Hz
        """
        E_upper = self.energy_shift(F_upper, J, 'J')
        E_lower = self.energy_shift(F_lower, J, 'J')
        
        delta_E = abs(E_upper - E_lower)
        
        return delta_E / self.h


# =============================================================================
# Eq 86: Hydrogen21cmLine - Famous transition
# =============================================================================

class Hydrogen21cmLine:
    """
    Batch 8, Eq 86: 21 cm line calculator.
    
    ET Math: ν = 1420.405751 MHz
             λ = 21.106 cm
    
    Ground state hyperfine splitting (F=1 ↔ F=0).
    Used to map neutral hydrogen in galaxies!
    """
    
    def __init__(self):
        """Initialize with 21 cm line constants."""
        self.frequency = HYDROGEN_21CM_FREQUENCY  # Hz
        self.wavelength = HYDROGEN_21CM_WAVELENGTH  # m
        self.h = PLANCK_CONSTANT_H
        self.c = SPEED_OF_LIGHT
    
    def get_frequency(self, unit: str = 'MHz') -> float:
        """
        Get 21 cm line frequency.
        
        Args:
            unit: 'Hz', 'MHz', 'GHz'
            
        Returns:
            Frequency
        """
        if unit == 'Hz':
            return self.frequency
        elif unit == 'MHz':
            return self.frequency / 1e6
        elif unit == 'GHz':
            return self.frequency / 1e9
        else:
            raise ValueError("unit must be 'Hz', 'MHz', or 'GHz'")
    
    def get_wavelength(self, unit: str = 'cm') -> float:
        """
        Get 21 cm line wavelength.
        
        Args:
            unit: 'm', 'cm', 'mm'
            
        Returns:
            Wavelength
        """
        if unit == 'm':
            return self.wavelength
        elif unit == 'cm':
            return self.wavelength * 100
        elif unit == 'mm':
            return self.wavelength * 1000
        else:
            raise ValueError("unit must be 'm', 'cm', or 'mm'")
    
    def photon_energy(self, unit: str = 'eV') -> float:
        """
        Calculate photon energy E = hf.
        
        Args:
            unit: 'eV' or 'J'
            
        Returns:
            Energy
        """
        E_J = self.h * self.frequency
        
        if unit == 'J':
            return E_J
        elif unit == 'eV':
            return E_J / 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")


# =============================================================================
# Eq 87: AngularMomentumCoupler - j = l + s validator
# =============================================================================

class AngularMomentumCoupler:
    """
    Batch 8, Eq 87: Angular momentum coupling validator.
    
    ET Math: |l - s| ≤ j ≤ l + s
    
    For electron s = 1/2, so j = l ± 1/2.
    """
    
    @staticmethod
    def valid_j_values(l: int, s: float = 0.5) -> Tuple[float, ...]:
        """
        Get valid j values for given l and s.
        
        Args:
            l: Orbital quantum number
            s: Spin (default 0.5)
            
        Returns:
            Tuple of allowed j values
        """
        if l == 0:
            # Only j = s for l=0
            return (s,)
        else:
            # j = l - 1/2 and l + 1/2
            j_minus = l - s
            j_plus = l + s
            return (j_minus, j_plus)
    
    @staticmethod
    def is_valid_coupling(l: int, s: float, j: float) -> bool:
        """
        Check if j is valid for given l and s.
        
        ET Math: |l - s| ≤ j ≤ l + s
        
        Args:
            l: Orbital quantum number
            s: Spin
            j: Total angular momentum
            
        Returns:
            True if valid
        """
        j_min = abs(l - s)
        j_max = l + s
        
        return j_min <= j <= j_max
    
    @staticmethod
    def coupling_coefficient(l: int, s: float, j: float) -> float:
        """
        Calculate Clebsch-Gordan-like coupling factor.
        
        Simplified for l and s=1/2.
        
        Args:
            l, s, j: Quantum numbers
            
        Returns:
            Coupling strength
        """
        if not AngularMomentumCoupler.is_valid_coupling(l, s, j):
            return 0.0
        
        # Simplified: |coefficient|² for l ± 1/2
        if l == 0:
            return 1.0
        else:
            if j == l + s:
                # j = l + 1/2 (parallel)
                return np.sqrt((l + 1) / (2 * l + 1))
            elif j == l - s:
                # j = l - 1/2 (antiparallel)
                return np.sqrt(l / (2 * l + 1))
            else:
                return 0.0


# =============================================================================
# Eq 88: ZeemanEffect - Magnetic field splitting
# =============================================================================

class ZeemanEffect:
    """
    Batch 8, Eq 88: Zeeman effect calculator.
    
    ET Math: ΔE = μ_B × g_j × m_j × B
    
    Magnetic field splits degenerate m_j levels.
    μ_B = Bohr magneton.
    """
    
    def __init__(self):
        """Initialize with fundamental constants."""
        # Bohr magneton: μ_B = eℏ/(2m_e)
        self.mu_B = (ELEMENTARY_CHARGE * PLANCK_CONSTANT_HBAR) / (2.0 * ELECTRON_MASS)
    
    def energy_shift(self, m_j: float, B_field: float, g_j: float = 2.0, 
                     unit: str = 'eV') -> float:
        """
        Calculate Zeeman shift.
        
        Args:
            m_j: Magnetic quantum number
            B_field: Magnetic field in Tesla
            g_j: Landé g-factor (default 2.0 for electron)
            unit: 'eV' or 'J'
            
        Returns:
            Energy shift
        """
        # ΔE = μ_B × g_j × m_j × B
        E_J = self.mu_B * g_j * m_j * B_field
        
        if unit == 'J':
            return E_J
        elif unit == 'eV':
            return E_J / 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def frequency_shift(self, m_j: float, B_field: float, g_j: float = 2.0) -> float:
        """
        Calculate Zeeman frequency shift.
        
        Args:
            m_j: Magnetic quantum number
            B_field: Magnetic field (T)
            g_j: Landé factor
            
        Returns:
            Frequency shift in Hz
        """
        E_J = self.energy_shift(m_j, B_field, g_j, 'J')
        
        return E_J / PLANCK_CONSTANT_H
    
    def lande_g_factor(self, j: float, l: int, s: float = 0.5) -> float:
        """
        Calculate Landé g-factor.
        
        ET Math: g_j = 1 + [j(j+1) + s(s+1) - l(l+1)] / [2j(j+1)]
        
        Args:
            j: Total angular momentum
            l: Orbital quantum number
            s: Spin
            
        Returns:
            g_j
        """
        if j == 0:
            return 0.0
        
        numerator = j * (j + 1) + s * (s + 1) - l * (l + 1)
        denominator = 2.0 * j * (j + 1)
        
        return 1.0 + numerator / denominator


# =============================================================================
# Eq 89: StarkEffect - Electric field shifts
# =============================================================================

class StarkEffect:
    """
    Batch 8, Eq 89: Stark effect calculator.
    
    ET Math: ΔE ∝ n × E_field
    
    Electric field mixes states, shifts energies.
    Linear in E for hydrogen (degenerate states).
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.a0 = BOHR_RADIUS
        self.e = ELEMENTARY_CHARGE
    
    def linear_shift(self, n: int, E_field: float, unit: str = 'eV') -> float:
        """
        Calculate linear Stark shift (approximate).
        
        ET Math: ΔE ∝ 3n × a₀ × e × E / 2
        
        Args:
            n: Principal quantum number
            E_field: Electric field in V/m
            unit: 'eV' or 'J'
            
        Returns:
            Energy shift (approximate)
        """
        # Linear Stark shift (simplified)
        # Exact requires degenerate perturbation theory
        shift_J = 1.5 * n * self.a0 * self.e * E_field
        
        if unit == 'J':
            return shift_J
        elif unit == 'eV':
            return shift_J / 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def quadratic_shift(self, n: int, l: int, m: int, E_field: float, unit: str = 'eV') -> float:
        """
        Calculate quadratic Stark shift (very approximate).
        
        For non-degenerate states: ΔE ∝ E²
        
        Args:
            n, l, m: Quantum numbers
            E_field: Electric field
            unit: Energy unit
            
        Returns:
            Energy shift (very approximate)
        """
        # This is a rough approximation
        # Real calculation requires full perturbation theory
        
        # Polarizability scales as n⁷
        alpha_pol = (n ** 7) * (self.a0 ** 3)
        
        shift_J = -0.5 * alpha_pol * (E_field ** 2)
        
        if unit == 'J':
            return shift_J
        elif unit == 'eV':
            return shift_J / 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")


# =============================================================================
# Eq 90: IsotopeShift - Mass-dependent corrections
# =============================================================================

class IsotopeShift:
    """
    Batch 8, Eq 90: Isotope shift calculator.
    
    ET Math: ΔE/E = Δμ/μ
    
    Different reduced mass shifts energy levels slightly.
    Deuterium vs. protium.
    """
    
    @staticmethod
    def mass_shift_ratio(mass_nucleus_1: float, mass_nucleus_2: float) -> float:
        """
        Calculate fractional energy shift due to different nuclear mass.
        
        ET Math: ΔE/E = Δμ/μ
        
        Args:
            mass_nucleus_1: First nucleus mass (e.g., protium)
            mass_nucleus_2: Second nucleus mass (e.g., deuterium)
            
        Returns:
            Fractional shift Δμ/μ
        """
        # Reduced masses
        mu_1 = (ELECTRON_MASS * mass_nucleus_1) / (ELECTRON_MASS + mass_nucleus_1)
        mu_2 = (ELECTRON_MASS * mass_nucleus_2) / (ELECTRON_MASS + mass_nucleus_2)
        
        # Fractional shift
        delta_mu = mu_2 - mu_1
        
        return delta_mu / mu_1
    
    @staticmethod
    def energy_shift(n: int, mass_nucleus_1: float, mass_nucleus_2: float, 
                    unit: str = 'eV') -> float:
        """
        Calculate absolute energy shift.
        
        Args:
            n: Principal quantum number
            mass_nucleus_1, mass_nucleus_2: Nuclear masses
            unit: Energy unit
            
        Returns:
            Energy shift
        """
        # Base energy for isotope 1
        E_n_1_eV = -RYDBERG_ENERGY / (n * n)
        
        # Fractional shift
        frac_shift = IsotopeShift.mass_shift_ratio(mass_nucleus_1, mass_nucleus_2)
        
        # Absolute shift
        delta_E_eV = E_n_1_eV * frac_shift
        
        if unit == 'eV':
            return delta_E_eV
        elif unit == 'J':
            return delta_E_eV * 1.602176634e-19
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    @staticmethod
    def hydrogen_deuterium_shift(n: int = 1, unit: str = 'eV') -> float:
        """
        Calculate isotope shift for H vs D (deuterium).
        
        Args:
            n: Principal quantum number
            unit: Energy unit
            
        Returns:
            Isotope shift
        """
        # Deuterium mass ≈ 2 × proton mass
        m_deuterium = 2.014 * PROTON_MASS  # More accurate
        
        return IsotopeShift.energy_shift(n, PROTON_MASS, m_deuterium, unit)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'SpinOrbitCoupling',
    'RelativisticCorrection',
    'FineStructureShift',
    'LambShiftCalculator',
    'HyperfineSplitting',
    'Hydrogen21cmLine',
    'AngularMomentumCoupler',
    'ZeemanEffect',
    'StarkEffect',
    'IsotopeShift',
]
