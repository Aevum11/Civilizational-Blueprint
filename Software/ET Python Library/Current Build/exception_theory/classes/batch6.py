"""
Exception Theory Batch 6 Classes
Hydrogen Atom Core (Eq 61-70)

Complete hydrogen atom implementation from ET primitives.
All derived from P∘D∘T manifold structure.

Classes:
- ReducedMassCalculator: Two-body system effective mass
- HydrogenEnergyLevels: Energy eigenvalue calculator  
- BohrRadiusSystem: Atomic scale calculator
- HydrogenHamiltonian: Complete Hamiltonian operator
- RadialWavefunction: Radial part of wavefunction
- SphericalHarmonicCalculator: Angular wavefunctions
- HydrogenWavefunction: Complete ψ_nlm
- OrbitalAngularMomentumCalculator: L quantum numbers
- TotalAngularMomentumCoupling: L+S coupling
- QuantumNumberValidator: State validation

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-18
Version: 3.1.0
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import special  # For spherical harmonics and associated Laguerre polynomials

from ..core.constants import (
    PLANCK_CONSTANT_HBAR,
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
    ELECTRON_MASS,
    PROTON_MASS,
    BOHR_RADIUS,
    RYDBERG_ENERGY,
)
from ..core.mathematics import ETMathV2


# =============================================================================
# Eq 61: ReducedMassCalculator - Two-body effective mass
# =============================================================================

class ReducedMassCalculator:
    """
    Batch 6, Eq 61: Reduced mass for two-body systems.
    
    ET Math: μ = (m₁m₂)/(m₁+m₂)
    
    Effective mass for relative motion.
    For hydrogen: μ ≈ m_e (since m_p >> m_e).
    """
    
    @staticmethod
    def reduced_mass(m1: float, m2: float) -> float:
        """
        Calculate reduced mass.
        
        Args:
            m1, m2: Masses in kg
            
        Returns:
            Reduced mass
        """
        return (m1 * m2) / (m1 + m2)
    
    @staticmethod
    def hydrogen_reduced_mass() -> float:
        """
        Calculate reduced mass for hydrogen (e⁻ + p⁺).
        
        Returns:
            μ_H in kg
        """
        return ReducedMassCalculator.reduced_mass(ELECTRON_MASS, PROTON_MASS)
    
    @staticmethod
    def mass_ratio_correction(m1: float, m2: float) -> float:
        """
        Calculate correction factor μ/m₁.
        
        Returns:
            Ratio (< 1)
        """
        mu = ReducedMassCalculator.reduced_mass(m1, m2)
        return mu / m1


# =============================================================================
# Eq 62: HydrogenEnergyLevels - Energy eigenvalues
# =============================================================================

class HydrogenEnergyLevels:
    """
    Batch 6, Eq 62: Hydrogen energy level calculator.
    
    ET Math: E_n = -(μe⁴)/(32π²ε₀²ℏ²n²) = -13.6 eV/n²
    
    Quantized descriptor configurations.
    Geometric eigenvalues from manifold.
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize for hydrogenic ion.
        
        Args:
            nuclear_charge: Z (1 for H, 2 for He+, etc.)
        """
        self.Z = nuclear_charge
        self.Ry = RYDBERG_ENERGY  # eV
        self.eV_to_J = 1.602176634e-19
        
        # Reduced mass
        self.mu = ReducedMassCalculator.hydrogen_reduced_mass()
    
    def energy(self, n: int, unit: str = 'eV') -> float:
        """
        Calculate E_n = -Z²Ry/n².
        
        Args:
            n: Principal quantum number
            unit: 'eV' or 'J'
            
        Returns:
            Energy (negative = bound)
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        E_eV = -(self.Z * self.Z * self.Ry) / (n * n)
        
        if unit == 'eV':
            return E_eV
        elif unit == 'J':
            return E_eV * self.eV_to_J
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def binding_energy(self, n: int = 1, unit: str = 'eV') -> float:
        """
        Calculate binding energy (positive).
        
        Args:
            n: Level (default ground state)
            unit: Energy unit
            
        Returns:
            |E_n|
        """
        return abs(self.energy(n, unit))
    
    def level_spacing(self, n1: int, n2: int, unit: str = 'eV') -> float:
        """
        Calculate ΔE = E_n2 - E_n1.
        
        Args:
            n1: Lower level
            n2: Upper level
            unit: Energy unit
            
        Returns:
            Energy difference
        """
        return self.energy(n2, unit) - self.energy(n1, unit)


# =============================================================================
# Eq 63: BohrRadiusSystem - Atomic scale
# =============================================================================

class BohrRadiusSystem:
    """
    Batch 6, Eq 63: Bohr radius calculator.
    
    ET Math: a₀ = 4πε₀ℏ²/(μe²) ≈ 0.529 Å
    
    Balance: quantum pressure = Coulomb attraction.
    All from manifold geometry!
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize for hydrogenic system.
        
        Args:
            nuclear_charge: Z
        """
        self.Z = nuclear_charge
        self.a0 = BOHR_RADIUS
    
    def bohr_radius(self) -> float:
        """
        Get effective Bohr radius a₀/Z.
        
        Returns:
            Radius in meters
        """
        return self.a0 / self.Z
    
    def most_probable_radius(self, n: int, l: int) -> float:
        """
        Calculate most probable radius for orbital n,l.
        
        For hydrogen ground state (1s): r_max = a₀
        General: depends on n, l
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            
        Returns:
            Most probable r
        """
        # For hydrogen-like atoms
        # r_max ≈ n²a₀/Z for s orbitals (l=0)
        # More complex for l > 0
        
        if l == 0:
            # s orbital
            return (n * n * self.a0) / self.Z
        else:
            # Approximate
            return ((n * n - l * (l + 1) / 2) * self.a0) / self.Z
    
    def average_radius(self, n: int, l: int) -> float:
        """
        Calculate ⟨r⟩_nl.
        
        ET Math: ⟨r⟩ = (n²a₀/Z)[1 + 0.5(1 - l(l+1)/n²)]
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            
        Returns:
            Average radius
        """
        factor = 1.0 + 0.5 * (1.0 - l * (l + 1) / (n * n))
        
        return (n * n * self.a0 / self.Z) * factor


# =============================================================================
# Eq 64: HydrogenHamiltonian - Complete Hamiltonian
# =============================================================================

class HydrogenHamiltonian:
    """
    Batch 6, Eq 64: Hydrogen Hamiltonian operator.
    
    ET Math: Ĥ = -ℏ²/(2μ)∇² + l(l+1)ℏ²/(2μr²) - Ze²/(4πε₀r)
    
    Kinetic + centrifugal + Coulomb.
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize Hamiltonian.
        
        Args:
            nuclear_charge: Z
        """
        self.Z = nuclear_charge
        self.mu = ReducedMassCalculator.hydrogen_reduced_mass()
        self.k_e = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY)
    
    def effective_potential(self, r: float, l: int) -> float:
        """
        Calculate V_eff(r) = V_centrifugal + V_Coulomb.
        
        ET Math: V_eff = l(l+1)ℏ²/(2μr²) - Ze²/(4πε₀r)
        
        Args:
            r: Radial distance
            l: Orbital angular momentum
            
        Returns:
            Effective potential energy
        """
        if r == 0:
            return np.inf if l > 0 else -np.inf
        
        # Centrifugal barrier
        V_cent = (l * (l + 1) * PLANCK_CONSTANT_HBAR**2) / (2.0 * self.mu * r * r)
        
        # Coulomb attraction
        V_coul = -self.k_e * self.Z * ELEMENTARY_CHARGE**2 / r
        
        return V_cent + V_coul
    
    def kinetic_energy_expectation(self, n: int) -> float:
        """
        Calculate ⟨T⟩ for state n.
        
        From virial theorem: ⟨T⟩ = -E_n
        
        Args:
            n: Principal quantum number
            
        Returns:
            Kinetic energy (positive)
        """
        energy_calc = HydrogenEnergyLevels(self.Z)
        E_n = energy_calc.energy(n, 'J')
        
        return -E_n
    
    def potential_energy_expectation(self, n: int) -> float:
        """
        Calculate ⟨V⟩ for state n.
        
        From virial theorem: ⟨V⟩ = 2E_n
        
        Args:
            n: Principal quantum number
            
        Returns:
            Potential energy (negative)
        """
        energy_calc = HydrogenEnergyLevels(self.Z)
        E_n = energy_calc.energy(n, 'J')
        
        return 2.0 * E_n


# =============================================================================
# Eq 65: RadialWavefunction - Radial part
# =============================================================================

class RadialWavefunction:
    """
    Batch 6, Eq 65: Radial wavefunction calculator.
    
    ET Math: R_nl(r) involves associated Laguerre polynomials
    
    For 1s (n=1, l=0): R₁₀(r) = 2(Z/a₀)^(3/2) exp(-Zr/a₀)
    """
    
    def __init__(self, nuclear_charge: int = 1):
        """
        Initialize for hydrogenic system.
        
        Args:
            nuclear_charge: Z
        """
        self.Z = nuclear_charge
        self.a0 = BOHR_RADIUS
    
    def R_nl(self, n: int, l: int, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate R_nl(r) using exact formula.
        
        ET Math: R_nl = normalization × (ρ^l) × exp(-ρ/2) × L_{n-l-1}^{2l+1}(ρ)
        
        where ρ = 2Zr/(na₀), L = generalized Laguerre polynomial
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            r: Radial coordinate(s)
            
        Returns:
            Radial wavefunction value(s)
        """
        if l >= n or l < 0:
            raise ValueError("Invalid quantum numbers: 0 <= l < n")
        
        r = np.asarray(r)
        
        # Scaled radius
        rho = (2.0 * self.Z * r) / (n * self.a0)
        
        # Normalization constant
        from scipy.special import factorial
        norm = np.sqrt((2.0 * self.Z / (n * self.a0))**3 * 
                      factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
        
        # Generalized Laguerre polynomial L_{n-l-1}^{2l+1}(ρ)
        laguerre = special.genlaguerre(n - l - 1, 2 * l + 1)(rho)
        
        # Complete radial function
        R = norm * (rho**l) * np.exp(-rho / 2.0) * laguerre
        
        return R
    
    def R_10(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        1s orbital (ground state).
        
        ET Math: R₁₀ = 2(Z/a₀)^(3/2) exp(-Zr/a₀)
        
        Args:
            r: Radius
            
        Returns:
            R₁₀(r)
        """
        r = np.asarray(r)
        norm = 2.0 * (self.Z / self.a0)**1.5
        
        return norm * np.exp(-self.Z * r / self.a0)


# =============================================================================
# Eq 66: SphericalHarmonicCalculator - Angular part
# =============================================================================

class SphericalHarmonicCalculator:
    """
    Batch 6, Eq 66: Spherical harmonics Y_lm(θ,φ).
    
    ET Math: Y_lm(θ,φ) = √[(2l+1)(l-m)!/(4π(l+m)!)] P_l^m(cosθ) e^(imφ)
    
    Angular wavefunctions for orbital motion.
    """
    
    @staticmethod
    def Y_lm(l: int, m: int, theta: Union[float, np.ndarray], 
             phi: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Calculate Y_lm(θ,φ).
        
        Args:
            l: Orbital quantum number
            m: Magnetic quantum number
            theta: Polar angle (0 to π)
            phi: Azimuthal angle (0 to 2π)
            
        Returns:
            Complex spherical harmonic value
        """
        if abs(m) > l:
            raise ValueError("|m| must be <= l")
        
        # Use scipy for spherical harmonics
        return special.sph_harm(m, l, phi, theta)
    
    @staticmethod
    def Y_00(theta: Optional[float] = None, phi: Optional[float] = None) -> float:
        """
        Y₀₀ = 1/√(4π) (s orbital).
        
        Spherically symmetric.
        
        Returns:
            Constant value
        """
        return 1.0 / np.sqrt(4.0 * np.pi)
    
    @staticmethod
    def probability_density_angular(l: int, m: int, theta: Union[float, np.ndarray],
                                    phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate |Y_lm|² probability density.
        
        Args:
            l, m: Quantum numbers
            theta, phi: Angles
            
        Returns:
            Angular probability density
        """
        Y = SphericalHarmonicCalculator.Y_lm(l, m, theta, phi)
        
        return np.abs(Y)**2


# =============================================================================
# Eq 67: HydrogenWavefunction - Complete ψ_nlm
# =============================================================================

class HydrogenWavefunction:
    """
    Batch 6, Eq 67: Complete hydrogen wavefunction.
    
    ET Math: ψ_nlm(r,θ,φ) = R_nl(r) Y_lm(θ,φ)
    
    Full 3D wavefunction = radial × angular.
    """
    
    def __init__(self, n: int, l: int, m: int, nuclear_charge: int = 1):
        """
        Initialize wavefunction for state |nlm⟩.
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            m: Magnetic quantum number
            nuclear_charge: Z
        """
        if l >= n or l < 0:
            raise ValueError("Invalid: 0 <= l < n")
        if abs(m) > l:
            raise ValueError("Invalid: |m| <= l")
        
        self.n = n
        self.l = l
        self.m = m
        self.Z = nuclear_charge
        
        self.R_calc = RadialWavefunction(nuclear_charge)
        self.Y_calc = SphericalHarmonicCalculator()
    
    def psi(self, r: float, theta: float, phi: float) -> complex:
        """
        Calculate ψ_nlm(r,θ,φ).
        
        Args:
            r: Radial distance
            theta: Polar angle
            phi: Azimuthal angle
            
        Returns:
            Complex wavefunction value
        """
        R = self.R_calc.R_nl(self.n, self.l, r)
        Y = self.Y_calc.Y_lm(self.l, self.m, theta, phi)
        
        return R * Y
    
    def probability_density(self, r: float, theta: float, phi: float) -> float:
        """
        Calculate |ψ|² probability density.
        
        Args:
            r, theta, phi: Coordinates
            
        Returns:
            Probability density
        """
        psi_val = self.psi(r, theta, phi)
        
        return np.abs(psi_val)**2
    
    def radial_probability(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate P(r) = r² |R_nl(r)|².
        
        Radial probability density.
        
        Args:
            r: Radial distance(s)
            
        Returns:
            P(r)
        """
        r = np.asarray(r)
        R = self.R_calc.R_nl(self.n, self.l, r)
        
        return r * r * np.abs(R)**2


# =============================================================================
# Eq 68: OrbitalAngularMomentumCalculator - L quantum numbers
# =============================================================================

class OrbitalAngularMomentumCalculator:
    """
    Batch 6, Eq 68: Orbital angular momentum calculator.
    
    ET Math: |L| = √[l(l+1)]ℏ
             L_z = mℏ
    
    Quantized rotational descriptor.
    """
    
    @staticmethod
    def magnitude(l: int) -> float:
        """
        Calculate |L| = √[l(l+1)]ℏ.
        
        Args:
            l: Orbital quantum number
            
        Returns:
            Angular momentum magnitude
        """
        return np.sqrt(l * (l + 1)) * PLANCK_CONSTANT_HBAR
    
    @staticmethod
    def z_component(m: int) -> float:
        """
        Calculate L_z = mℏ.
        
        Args:
            m: Magnetic quantum number
            
        Returns:
            z-component of L
        """
        return m * PLANCK_CONSTANT_HBAR
    
    @staticmethod
    def orbital_letter(l: int) -> str:
        """
        Get spectroscopic notation for l.
        
        Args:
            l: Orbital quantum number
            
        Returns:
            's', 'p', 'd', 'f', 'g', 'h', ...
        """
        letters = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']
        
        if l < len(letters):
            return letters[l]
        else:
            return f"(l={l})"
    
    @staticmethod
    def degeneracy(l: int) -> int:
        """
        Calculate degeneracy 2l+1.
        
        Number of m states for given l.
        
        Args:
            l: Orbital quantum number
            
        Returns:
            Number of degenerate states
        """
        return 2 * l + 1


# =============================================================================
# Eq 69: TotalAngularMomentumCoupling - L+S coupling
# =============================================================================

class TotalAngularMomentumCoupling:
    """
    Batch 6, Eq 69: Total angular momentum j = l ± s.
    
    ET Math: |l - s| ≤ j ≤ l + s
    
    Spin-orbit coupling for electron (s = 1/2).
    """
    
    @staticmethod
    def possible_j_values(l: int, s: float = 0.5) -> Tuple[float, ...]:
        """
        Get possible j values for l and s.
        
        For electron: j = l ± 1/2
        
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
    def j_magnitude(j: float) -> float:
        """
        Calculate |J| = √[j(j+1)]ℏ.
        
        Args:
            j: Total angular momentum quantum number
            
        Returns:
            Magnitude
        """
        return np.sqrt(j * (j + 1)) * PLANCK_CONSTANT_HBAR
    
    @staticmethod
    def spectroscopic_term(l: int, s: float, j: float) -> str:
        """
        Generate term symbol ²ˢ⁺¹Lⱼ.
        
        Args:
            l: Orbital quantum number
            s: Spin
            j: Total angular momentum
            
        Returns:
            Term symbol (e.g., "2P_{1/2}")
        """
        multiplicity = int(2 * s + 1)
        l_letter = OrbitalAngularMomentumCalculator.orbital_letter(l).upper()
        
        return f"{multiplicity}{l_letter}_{{{j}}}"


# =============================================================================
# Eq 70: QuantumNumberValidator - State validation
# =============================================================================

class QuantumNumberValidator:
    """
    Batch 6, Eq 70: Quantum number validation.
    
    ET Math: n ≥ 1, 0 ≤ l < n, |m| ≤ l, s = ±1/2
    
    Manifold geometric constraints.
    """
    
    @staticmethod
    def validate(n: int, l: int, m: int, s: float = 0.5) -> Tuple[bool, str]:
        """
        Validate quantum numbers.
        
        Args:
            n: Principal
            l: Orbital
            m: Magnetic
            s: Spin
            
        Returns:
            (is_valid, error_message)
        """
        # Principal quantum number
        if n < 1:
            return (False, "n must be >= 1")
        
        # Orbital quantum number
        if l < 0:
            return (False, "l must be >= 0")
        if l >= n:
            return (False, f"l must be < n (got l={l}, n={n})")
        
        # Magnetic quantum number
        if abs(m) > l:
            return (False, f"|m| must be <= l (got m={m}, l={l})")
        
        # Spin
        if abs(s) != 0.5:
            return (False, "s must be ±1/2 for electron")
        
        return (True, "Valid")
    
    @staticmethod
    def is_valid(n: int, l: int, m: int, s: float = 0.5) -> bool:
        """
        Quick validation check.
        
        Returns:
            True if valid
        """
        valid, _ = QuantumNumberValidator.validate(n, l, m, s)
        return valid
    
    @staticmethod
    def state_label(n: int, l: int, m: int) -> str:
        """
        Generate state label like "1s", "2p", "3d₂".
        
        Args:
            n, l, m: Quantum numbers
            
        Returns:
            State label
        """
        l_letter = OrbitalAngularMomentumCalculator.orbital_letter(l)
        
        if m == 0:
            return f"{n}{l_letter}"
        else:
            return f"{n}{l_letter}(m={m})"


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'ReducedMassCalculator',
    'HydrogenEnergyLevels',
    'BohrRadiusSystem',
    'HydrogenHamiltonian',
    'RadialWavefunction',
    'SphericalHarmonicCalculator',
    'HydrogenWavefunction',
    'OrbitalAngularMomentumCalculator',
    'TotalAngularMomentumCoupling',
    'QuantumNumberValidator',
]
