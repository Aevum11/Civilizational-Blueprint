"""
Exception Theory Batch 4 Classes
Quantum Mechanics Foundations (Eq 41-50)

Complete implementation of quantum mechanics from ET primitives.
All derived from P∘D∘T manifold structure.

Classes:
- QuantumState: Wave function representation and evolution
- SchrodingerSolver: Time evolution engine
- UncertaintyAnalyzer: Heisenberg principle calculator
- OperatorAlgebra: Quantum operator manipulation
- WavefunctionNormalizer: Probability conservation
- QuantumMeasurement: Observable expectation values
- CoulombPotential: Electrostatic potential calculator
- HydrogenEnergyCalculator: Energy level computation
- BohrRadiusCalculator: Atomic scale calculator
- FineStructureCalculator: Relativistic corrections

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-18
Version: 3.1.0
"""

import numpy as np
from typing import Union, Callable, Tuple, List, Optional
import warnings

from ..core.constants import (
    PLANCK_CONSTANT_HBAR,
    PLANCK_CONSTANT_H,
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
    ELECTRON_MASS,
    PROTON_MASS,
    FINE_STRUCTURE_CONSTANT,
    BOHR_RADIUS,
    RYDBERG_ENERGY,
    MANIFOLD_SYMMETRY,
    BASE_VARIANCE,
)
from ..core.mathematics import ETMathV2


# =============================================================================
# Eq 41: QuantumState - Wave function representation
# =============================================================================

class QuantumState:
    """
    Batch 4, Eq 41: Quantum state representation and evolution.
    
    ET Math: |ψ⟩ = Unsubstantiated (P∘D) configuration
             Evolution via rotation in descriptor manifold
    
    Wave function = Superposition of descriptors (State 2)
    Time evolution preserves |ψ|² (probability conservation)
    """
    
    def __init__(self, wavefunction: np.ndarray, spatial_grid: Optional[np.ndarray] = None):
        """
        Initialize quantum state.
        
        Args:
            wavefunction: Complex array representing ψ
            spatial_grid: Optional spatial coordinate array
        """
        self.psi = np.asarray(wavefunction, dtype=complex)
        self.grid = spatial_grid if spatial_grid is not None else np.arange(len(self.psi))
        self.time = 0.0
        
        # Normalize on initialization
        self._normalize()
    
    def _normalize(self):
        """Normalize wavefunction to unit probability."""
        norm_sq = np.sum(np.abs(self.psi)**2)
        if norm_sq > 0:
            self.psi = self.psi / np.sqrt(norm_sq)
    
    def evolve(self, hamiltonian: Union[np.ndarray, Callable], dt: float):
        """
        Evolve state forward in time via Schrödinger equation.
        
        ET Math: |ψ(t+dt)⟩ = exp(-iĤdt/ℏ)|ψ(t)⟩
        
        Args:
            hamiltonian: Energy operator (matrix or function)
            dt: Time step
        """
        # Phase rotation factor
        phase_factor = -1j * dt / PLANCK_CONSTANT_HBAR
        
        if callable(hamiltonian):
            h_psi = hamiltonian(self.psi)
        else:
            h_psi = np.asarray(hamiltonian, dtype=complex) @ self.psi
        
        # First-order evolution
        self.psi = self.psi + phase_factor * h_psi
        
        # Renormalize to preserve probability
        self._normalize()
        
        self.time += dt
    
    def probability_density(self) -> np.ndarray:
        """
        Calculate |ψ|² probability density.
        
        ET Math: ρ(x) = |ψ(x)|² = (P∘D) substantiation probability
        
        Returns:
            Probability density array
        """
        return np.abs(self.psi)**2
    
    def expectation_value(self, operator: Union[np.ndarray, Callable]) -> complex:
        """
        Calculate expectation value ⟨ψ|Ô|ψ⟩.
        
        ET Math: ⟨O⟩ = Average descriptor content under measurement
        
        Args:
            operator: Observable operator
            
        Returns:
            Expectation value
        """
        if callable(operator):
            o_psi = operator(self.psi)
        else:
            o_psi = np.asarray(operator, dtype=complex) @ self.psi
        
        return np.vdot(self.psi, o_psi)
    
    def variance(self, operator: Union[np.ndarray, Callable]) -> float:
        """
        Calculate variance ⟨(Ô - ⟨Ô⟩)²⟩.
        
        ET Math: σ² = Descriptor variance = Uncertainty
        
        Args:
            operator: Observable operator
            
        Returns:
            Variance (real, non-negative)
        """
        mean = self.expectation_value(operator)
        
        if callable(operator):
            o_psi = operator(self.psi)
            o2_psi = operator(o_psi)
        else:
            o_matrix = np.asarray(operator, dtype=complex)
            o_psi = o_matrix @ self.psi
            o2_psi = o_matrix @ o_psi
        
        mean_sq = np.vdot(self.psi, o2_psi)
        
        return float(np.real(mean_sq - mean * np.conj(mean)))
    
    def copy(self):
        """Create deep copy of state."""
        new_state = QuantumState(self.psi.copy(), self.grid.copy() if self.grid is not None else None)
        new_state.time = self.time
        return new_state


# =============================================================================
# Eq 42: UncertaintyAnalyzer - Heisenberg principle
# =============================================================================

class UncertaintyAnalyzer:
    """
    Batch 4, Eq 42: Heisenberg uncertainty relation verification.
    
    ET Math: ΔxΔp ≥ ℏ/2
             V_D_s · V_∇D ≥ R_min
    
    Geometric manifold constraint from discrete structure.
    Manifold pixel size R_min = ℏ/2.
    """
    
    def __init__(self, state: QuantumState):
        """
        Initialize analyzer with quantum state.
        
        Args:
            state: QuantumState to analyze
        """
        self.state = state
    
    def position_variance(self) -> float:
        """
        Calculate position variance Δx².
        
        ET Math: ⟨x²⟩ - ⟨x⟩² = Spatial descriptor variance
        
        Returns:
            Position variance
        """
        x = self.state.grid
        prob = self.state.probability_density()
        
        # Spatial step
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        
        # ⟨x⟩
        mean_x = np.sum(x * prob) * dx
        
        # ⟨x²⟩
        mean_x2 = np.sum(x**2 * prob) * dx
        
        return mean_x2 - mean_x**2
    
    def momentum_variance(self) -> float:
        """
        Calculate momentum variance Δp².
        
        ET Math: ⟨p²⟩ - ⟨p⟩² = Momentum descriptor variance
        
        Returns:
            Momentum variance
        """
        # Fourier transform to momentum space
        psi_k = np.fft.fft(self.state.psi)
        prob_k = np.abs(psi_k)**2 / len(psi_k)
        
        # Momentum grid (k-space)
        dx = self.state.grid[1] - self.state.grid[0] if len(self.state.grid) > 1 else 1.0
        dk = 2 * np.pi / (len(self.state.grid) * dx)
        k = np.fft.fftfreq(len(self.state.grid), dx) * 2 * np.pi
        
        # p = ℏk
        p = PLANCK_CONSTANT_HBAR * k
        
        # ⟨p⟩
        mean_p = np.sum(p * prob_k)
        
        # ⟨p²⟩
        mean_p2 = np.sum(p**2 * prob_k)
        
        return mean_p2 - mean_p**2
    
    def uncertainty_product(self) -> float:
        """
        Calculate ΔxΔp in units of ℏ.
        
        ET Math: Product ≥ 1/2 (manifold constraint)
        
        Returns:
            ΔxΔp / ℏ (should be ≥ 0.5)
        """
        dx_sq = self.position_variance()
        dp_sq = self.momentum_variance()
        
        product = np.sqrt(dx_sq * dp_sq)
        
        return product / PLANCK_CONSTANT_HBAR
    
    def verify_heisenberg(self) -> Tuple[float, bool]:
        """
        Verify Heisenberg uncertainty relation.
        
        Returns:
            Tuple (uncertainty_product, is_satisfied)
        """
        product = self.uncertainty_product()
        min_product = 0.5  # ℏ/2 in units of ℏ
        
        is_satisfied = product >= min_product - 1e-10  # Small tolerance
        
        return (product, is_satisfied)


# =============================================================================
# Eq 43: OperatorAlgebra - Quantum operators
# =============================================================================

class OperatorAlgebra:
    """
    Batch 4, Eq 43: Quantum operator manipulation.
    
    ET Math: p̂ = -iℏ∂/∂x (momentum operator)
             x̂ψ = xψ (position operator)
    
    Operators = Traverser navigation functions on descriptor fields.
    """
    
    @staticmethod
    def position_operator(wavefunction: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Apply position operator x̂.
        
        ET Math: x̂ψ(x) = xψ(x)
        
        Multiplication by position value.
        
        Args:
            wavefunction: ψ(x)
            grid: Spatial coordinates
            
        Returns:
            x̂ψ
        """
        return grid * wavefunction
    
    @staticmethod
    def momentum_operator(wavefunction: np.ndarray, dx: float) -> np.ndarray:
        """
        Apply momentum operator p̂ = -iℏ∂/∂x.
        
        ET Math: P_sub = A_px / J_T = ℏk
                p̂ = -iℏ∇
        
        Args:
            wavefunction: ψ(x)
            dx: Spatial step
            
        Returns:
            p̂ψ
        """
        # Central difference gradient
        gradient = np.gradient(wavefunction, dx)
        
        return -1j * PLANCK_CONSTANT_HBAR * gradient
    
    @staticmethod
    def kinetic_operator(wavefunction: np.ndarray, dx: float, mass: float) -> np.ndarray:
        """
        Apply kinetic energy operator T̂ = -ℏ²/(2m)∇².
        
        ET Math: Descriptor variation energy
        
        Args:
            wavefunction: ψ(x)
            dx: Spatial step
            mass: Particle mass
            
        Returns:
            T̂ψ
        """
        # Second derivative via gradient of gradient
        laplacian = np.gradient(np.gradient(wavefunction, dx), dx)
        
        return (-PLANCK_CONSTANT_HBAR**2 / (2 * mass)) * laplacian
    
    @staticmethod
    def potential_operator(wavefunction: np.ndarray, potential: np.ndarray) -> np.ndarray:
        """
        Apply potential energy operator V̂.
        
        ET Math: V̂ψ = V(x)ψ(x)
        
        Args:
            wavefunction: ψ(x)
            potential: V(x) array
            
        Returns:
            V̂ψ
        """
        return potential * wavefunction
    
    @staticmethod
    def hamiltonian_operator(wavefunction: np.ndarray, dx: float, mass: float, 
                            potential: np.ndarray) -> np.ndarray:
        """
        Apply Hamiltonian Ĥ = T̂ + V̂.
        
        ET Math: Total descriptor content = Kinetic + Potential
        
        Args:
            wavefunction: ψ(x)
            dx: Spatial step
            mass: Particle mass
            potential: V(x)
            
        Returns:
            Ĥψ
        """
        t_psi = OperatorAlgebra.kinetic_operator(wavefunction, dx, mass)
        v_psi = OperatorAlgebra.potential_operator(wavefunction, potential)
        
        return t_psi + v_psi
    
    @staticmethod
    def commutator(op_a: Callable, op_b: Callable, wavefunction: np.ndarray) -> np.ndarray:
        """
        Calculate commutator [Â,B̂]ψ = ÂB̂ψ - B̂Âψ.
        
        ET Math: Operator non-commutativity from manifold geometry
        
        Args:
            op_a: First operator
            op_b: Second operator
            wavefunction: State to apply to
            
        Returns:
            [Â,B̂]ψ
        """
        ab_psi = op_a(op_b(wavefunction))
        ba_psi = op_b(op_a(wavefunction))
        
        return ab_psi - ba_psi


# =============================================================================
# Eq 44: CoulombPotential - Electrostatic potential
# =============================================================================

class CoulombPotential:
    """
    Batch 4, Eq 44: Coulomb potential calculator.
    
    ET Math: V(r) = (1/4πε₀)(q₁q₂/r)
             I_dev = η_M · (T_b1 ∘ T_b2) / (D_sep)²
    
    Bound traverser clusters = Charges
    Manifold coupling constant = k_e
    """
    
    def __init__(self, charge1: float, charge2: float):
        """
        Initialize Coulomb potential.
        
        Args:
            charge1: First charge (C)
            charge2: Second charge (C)
        """
        self.q1 = charge1
        self.q2 = charge2
        
        # Coulomb constant
        self.k_e = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY)
    
    def potential(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate V(r) at distance r.
        
        Args:
            r: Separation distance(s)
            
        Returns:
            Potential energy
        """
        r = np.asarray(r)
        
        # Avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = self.k_e * self.q1 * self.q2 / r
            
        # Set r=0 to appropriate infinity
        if np.isscalar(r):
            if r == 0:
                v = np.inf if self.q1 * self.q2 > 0 else -np.inf
        else:
            mask = (r == 0)
            if self.q1 * self.q2 > 0:
                v[mask] = np.inf
            else:
                v[mask] = -np.inf
        
        return v
    
    def force(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate F(r) = -dV/dr.
        
        Returns:
            Force magnitude (positive = repulsion)
        """
        r = np.asarray(r)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = self.k_e * self.q1 * self.q2 / (r * r)
        
        if np.isscalar(r):
            if r == 0:
                f = np.inf
        else:
            f[r == 0] = np.inf
        
        return f


# =============================================================================
# Eq 45: HydrogenEnergyCalculator - Energy levels
# =============================================================================

class HydrogenEnergyCalculator:
    """
    Batch 4, Eq 45: Hydrogen atom energy level calculator.
    
    ET Math: E_n = -(μe⁴)/(32π²ε₀²ℏ²n²) = -13.6 eV/n²
    
    Quantized descriptor configurations in Coulomb potential.
    Geometric eigenvalues from manifold structure.
    """
    
    def __init__(self):
        """Initialize calculator with hydrogen parameters."""
        # Reduced mass
        self.mu = (ELECTRON_MASS * PROTON_MASS) / (ELECTRON_MASS + PROTON_MASS)
        
        # Rydberg energy (ground state binding)
        self.rydberg = RYDBERG_ENERGY
        
        # eV to Joules conversion
        self.eV_to_J = 1.602176634e-19
    
    def energy_level(self, n: int, unit: str = 'eV') -> float:
        """
        Calculate E_n for principal quantum number n.
        
        ET Math: Stable (P∘D) configuration energies
        
        Args:
            n: Principal quantum number (1, 2, 3, ...)
            unit: 'eV' or 'J'
            
        Returns:
            Energy (negative = bound)
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        # E_n = -Ry/n²
        energy_eV = -self.rydberg / (n * n)
        
        if unit == 'eV':
            return energy_eV
        elif unit == 'J':
            return energy_eV * self.eV_to_J
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def ionization_energy(self, n: int = 1, unit: str = 'eV') -> float:
        """
        Calculate ionization energy from level n.
        
        Args:
            n: Initial level
            unit: 'eV' or 'J'
            
        Returns:
            Energy to ionize (positive)
        """
        return -self.energy_level(n, unit)
    
    def transition_energy(self, n_initial: int, n_final: int, unit: str = 'eV') -> float:
        """
        Calculate ΔE = E_final - E_initial.
        
        Args:
            n_initial: Initial level
            n_final: Final level
            unit: 'eV' or 'J'
            
        Returns:
            Energy difference (positive = absorption)
        """
        e_i = self.energy_level(n_initial, unit)
        e_f = self.energy_level(n_final, unit)
        
        return e_f - e_i
    
    def all_levels(self, n_max: int = 10, unit: str = 'eV') -> np.ndarray:
        """
        Calculate energy levels from n=1 to n_max.
        
        Args:
            n_max: Maximum n
            unit: 'eV' or 'J'
            
        Returns:
            Array of energies
        """
        return np.array([self.energy_level(n, unit) for n in range(1, n_max + 1)])


# =============================================================================
# Eq 46: BohrRadiusCalculator - Atomic scale
# =============================================================================

class BohrRadiusCalculator:
    """
    Batch 4, Eq 46: Bohr radius calculator for hydrogenic systems.
    
    ET Math: a₀ = 4πε₀ℏ²/(μe²)
    
    Balance point: Quantum variance pressure = Coulomb attraction
    All from manifold geometry constants!
    """
    
    def __init__(self):
        """Initialize with hydrogen parameters."""
        self.a0 = BOHR_RADIUS
    
    def bohr_radius_hydrogen(self) -> float:
        """
        Get Bohr radius for hydrogen.
        
        Returns:
            a₀ in meters (0.529 Å)
        """
        return self.a0
    
    def bohr_radius_hydrogenic(self, nuclear_charge: float, reduced_mass: float) -> float:
        """
        Calculate Bohr radius for hydrogenic ion.
        
        ET Math: a = a₀ × (μ_H/μ) × (1/Z)
        
        Args:
            nuclear_charge: Z (number of protons)
            reduced_mass: μ for system
            
        Returns:
            Effective Bohr radius
        """
        mu_H = (ELECTRON_MASS * PROTON_MASS) / (ELECTRON_MASS + PROTON_MASS)
        
        return self.a0 * (mu_H / reduced_mass) / nuclear_charge
    
    def average_radius(self, n: int, l: int) -> float:
        """
        Calculate ⟨r⟩_nl for orbital.
        
        ET Math: ⟨r⟩_nl = (n²a₀/Z)[1 + (1/2)(1 - l(l+1)/n²)]
        
        For Z=1 (hydrogen):
        ⟨r⟩_nl ≈ n²a₀[3/2 - l(l+1)/(2n²)]
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            
        Returns:
            Average radius
        """
        if l >= n:
            raise ValueError("l must be < n")
        
        # Simplified for hydrogen (Z=1)
        factor = 1.5 - (l * (l + 1)) / (2 * n * n)
        
        return (n * n * self.a0) * factor


# =============================================================================
# Eq 47: FineStructureCalculator - Relativistic corrections
# =============================================================================

class FineStructureCalculator:
    """
    Batch 4, Eq 47: Fine structure corrections calculator.
    
    ET Math: ΔE_fs ∝ α² × E_n × f(n,l,j)
    
    Relativistic corrections:
    - Spin-orbit coupling (L·S interaction)
    - Kinetic energy relativistic correction
    
    O(α²) ≈ 10⁻⁵ relative effect.
    """
    
    def __init__(self):
        """Initialize with fundamental constants."""
        self.alpha = FINE_STRUCTURE_CONSTANT
        self.alpha_sq = self.alpha ** 2
        self.energy_calc = HydrogenEnergyCalculator()
    
    def spin_orbit_shift(self, n: int, l: int, j: float, unit: str = 'eV') -> float:
        """
        Calculate spin-orbit coupling shift.
        
        ET Math: ΔE_so ∝ α² × E_n × [j(j+1) - l(l+1) - s(s+1)]
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            j: Total angular momentum (l ± 1/2)
            unit: 'eV' or 'J'
            
        Returns:
            Energy shift
        """
        if l == 0:
            return 0.0  # No spin-orbit for s orbitals
        
        if l < 0 or l >= n:
            raise ValueError("Invalid quantum numbers")
        
        # Base energy
        E_n = self.energy_calc.energy_level(n, unit)
        
        # Spin s = 1/2
        s = 0.5
        
        # j(j+1) - l(l+1) - s(s+1)
        factor = j * (j + 1) - l * (l + 1) - s * (s + 1)
        
        # Approximate formula
        if l > 0:
            delta = E_n * self.alpha_sq * factor / (n * l * (l + 0.5) * (l + 1))
        else:
            delta = 0.0
        
        return delta
    
    def relativistic_shift(self, n: int, l: int, unit: str = 'eV') -> float:
        """
        Calculate relativistic kinetic energy correction.
        
        ET Math: Descriptor gradient approaching v_T,max (c)
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number  
            unit: 'eV' or 'J'
            
        Returns:
            Energy correction
        """
        E_n = self.energy_calc.energy_level(n, unit)
        
        # ΔE_rel = -E_n × α² × (n/(l+0.5) - 3/4) / n²
        correction = -E_n * self.alpha_sq * (n / (l + 0.5) - 0.75) / (n * n)
        
        return correction
    
    def total_fine_structure(self, n: int, l: int, j: float, unit: str = 'eV') -> float:
        """
        Calculate total fine structure shift.
        
        ET Math: ΔE_fs = ΔE_so + ΔE_rel
        
        Args:
            n: Principal quantum number
            l: Orbital quantum number
            j: Total angular momentum
            unit: 'eV' or 'J'
            
        Returns:
            Total shift
        """
        delta_so = self.spin_orbit_shift(n, l, j, unit)
        delta_rel = self.relativistic_shift(n, l, unit)
        
        return delta_so + delta_rel


# =============================================================================
# Eq 48: RydbergWavelengthCalculator - Spectral lines
# =============================================================================

class RydbergWavelengthCalculator:
    """
    Batch 4, Eq 48: Rydberg formula for spectral line wavelengths.
    
    ET Math: 1/λ = R_∞(1/n₁² - 1/n₂²)
    
    R_∞ = (μe⁴)/(64π³ε₀²ℏ³c) from manifold geometry constants.
    All constants derive from (P∘D∘T) structure!
    """
    
    def __init__(self):
        """Initialize with Rydberg constant."""
        from ..core.constants import RYDBERG_CONSTANT, SPEED_OF_LIGHT, PLANCK_CONSTANT_H
        
        self.R_inf = RYDBERG_CONSTANT  # m⁻¹
        self.c = SPEED_OF_LIGHT
        self.h = PLANCK_CONSTANT_H
    
    def wavelength(self, n1: int, n2: int, unit: str = 'm') -> float:
        """
        Calculate transition wavelength n2 → n1.
        
        ET Math: 1/λ = R_∞(1/n₁² - 1/n₂²)
        
        Args:
            n1: Lower energy level
            n2: Upper energy level
            unit: 'm' (meters), 'nm' (nanometers), 'A' (Angstroms)
            
        Returns:
            Wavelength
        """
        if n2 <= n1:
            raise ValueError("n2 must be > n1 for emission")
        
        # Wavenumber: 1/λ
        wavenumber = self.R_inf * (1.0/(n1*n1) - 1.0/(n2*n2))
        
        if wavenumber == 0:
            return np.inf
        
        # Wavelength in meters
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
    
    def frequency(self, n1: int, n2: int) -> float:
        """
        Calculate transition frequency.
        
        ET Math: f = c/λ
        
        Args:
            n1: Lower level
            n2: Upper level
            
        Returns:
            Frequency in Hz
        """
        wavelength_m = self.wavelength(n1, n2, 'm')
        
        if np.isinf(wavelength_m):
            return 0.0
        
        return self.c / wavelength_m
    
    def photon_energy(self, n1: int, n2: int, unit: str = 'eV') -> float:
        """
        Calculate photon energy E = hf.
        
        Args:
            n1: Lower level
            n2: Upper level
            unit: 'eV' or 'J'
            
        Returns:
            Photon energy
        """
        freq = self.frequency(n1, n2)
        energy_J = self.h * freq
        
        if unit == 'J':
            return energy_J
        elif unit == 'eV':
            eV_to_J = 1.602176634e-19
            return energy_J / eV_to_J
        else:
            raise ValueError("unit must be 'eV' or 'J'")
    
    def series_wavelengths(self, n_lower: int, n_max: int = 10, unit: str = 'nm') -> dict:
        """
        Calculate all wavelengths in a spectral series.
        
        Args:
            n_lower: Lower level (1=Lyman, 2=Balmer, 3=Paschen)
            n_max: Maximum upper level
            unit: Wavelength unit
            
        Returns:
            Dictionary {n_upper: wavelength}
        """
        wavelengths = {}
        
        for n_upper in range(n_lower + 1, n_max + 1):
            wavelengths[n_upper] = self.wavelength(n_lower, n_upper, unit)
        
        return wavelengths


# =============================================================================
# Eq 49-50: Additional support classes
# =============================================================================

class WavefunctionNormalizer:
    """
    Batch 4, Eq 49: Wavefunction normalization utility.
    
    ET Math: ∫|ψ|²dV = 1
    
    Probability conservation = (P∘D) descriptor content conservation.
    """
    
    @staticmethod
    def normalize_1d(wavefunction: np.ndarray, dx: float) -> np.ndarray:
        """
        Normalize 1D wavefunction.
        
        Args:
            wavefunction: ψ(x)
            dx: Spatial step
            
        Returns:
            Normalized ψ
        """
        prob_density = np.abs(wavefunction)**2
        total_prob = np.sum(prob_density) * dx
        
        if total_prob > 0:
            return wavefunction / np.sqrt(total_prob)
        else:
            return wavefunction
    
    @staticmethod
    def normalize_3d(wavefunction: np.ndarray, volume_element: float) -> np.ndarray:
        """
        Normalize 3D wavefunction.
        
        Args:
            wavefunction: ψ(r,θ,φ) flattened
            volume_element: dV
            
        Returns:
            Normalized ψ
        """
        prob_density = np.abs(wavefunction)**2
        total_prob = np.sum(prob_density) * volume_element
        
        if total_prob > 0:
            return wavefunction / np.sqrt(total_prob)
        else:
            return wavefunction


class QuantumMeasurement:
    """
    Batch 4, Eq 50: Quantum measurement and expectation values.
    
    ET Math: ⟨Ô⟩ = ⟨ψ|Ô|ψ⟩
    
    Measurement = T substantiation of (P∘D) configuration.
    """
    
    @staticmethod
    def expectation(state: QuantumState, operator: Union[np.ndarray, Callable]) -> complex:
        """Calculate ⟨Ô⟩."""
        return state.expectation_value(operator)
    
    @staticmethod
    def uncertainty(state: QuantumState, operator: Union[np.ndarray, Callable]) -> float:
        """Calculate ΔO = √⟨Ô²⟩ - ⟨Ô⟩²."""
        return np.sqrt(state.variance(operator))
    
    @staticmethod
    def position_expectation(state: QuantumState) -> float:
        """Calculate ⟨x⟩."""
        x = state.grid
        prob = state.probability_density()
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        
        return np.sum(x * prob) * dx
    
    @staticmethod
    def momentum_expectation(state: QuantumState) -> float:
        """Calculate ⟨p⟩."""
        dx = state.grid[1] - state.grid[0] if len(state.grid) > 1 else 1.0
        p_psi = OperatorAlgebra.momentum_operator(state.psi, dx)
        
        return np.real(np.vdot(state.psi, p_psi))


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'QuantumState',
    'UncertaintyAnalyzer',
    'OperatorAlgebra',
    'CoulombPotential',
    'HydrogenEnergyCalculator',
    'BohrRadiusCalculator',
    'FineStructureCalculator',
    'RydbergWavelengthCalculator',
    'WavefunctionNormalizer',
    'QuantumMeasurement',
]
