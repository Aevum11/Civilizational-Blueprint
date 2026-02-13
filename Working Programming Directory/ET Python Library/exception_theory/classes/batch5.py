"""
Exception Theory Batch 5 Classes
Electromagnetism (Eq 51-60)

Complete implementation of electromagnetic theory from ET primitives.
All derived from P∘D∘T manifold structure.

Classes:
- CoulombForceCalculator: Force between charges
- ElectricPotentialField: Electric potential calculator
- ElectricFieldCalculator: E-field from sources
- MagneticFieldCalculator: B-field from currents
- LorentzForceCalculator: Force on moving charges
- EMEnergyCalculator: Field energy density
- FineStructureConstant: EM coupling calculator
- VacuumImpedance: Free space impedance
- CoulombConstantCalculator: Electrostatic constant
- MagneticConstantCalculator: Magnetic permeability

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-18
Version: 3.1.0
"""

import numpy as np
from typing import Union, Tuple, List

from ..core.constants import (
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
    VACUUM_PERMEABILITY,
    SPEED_OF_LIGHT,
    FINE_STRUCTURE_CONSTANT,
    FINE_STRUCTURE_INVERSE,
)
from ..core.mathematics import ETMathV2


# =============================================================================
# Eq 51: CoulombForceCalculator - Force between charges
# =============================================================================

class CoulombForceCalculator:
    """
    Batch 5, Eq 51: Coulomb force between charges.
    
    ET Math: F = (1/4πε₀)(q₁q₂/r²)
             I_dev = η_M · (T_b1 ∘ T_b2) / (D_sep)²
    
    Bound traverser clusters = Charges
    Manifold coupling = k_e = 1/(4πε₀)
    """
    
    def __init__(self):
        """Initialize with Coulomb constant."""
        self.k_e = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY)
    
    def force_magnitude(self, q1: float, q2: float, r: float) -> float:
        """
        Calculate |F| between two charges.
        
        Args:
            q1, q2: Charges in Coulombs
            r: Separation in meters
            
        Returns:
            Force magnitude in Newtons (positive = repulsion)
        """
        if r == 0:
            return np.inf
        
        return self.k_e * abs(q1 * q2) / (r * r)
    
    def force_vector(self, q1: float, q2: float, r_vec: np.ndarray) -> np.ndarray:
        """
        Calculate vector force F = F_mag * r_hat.
        
        Args:
            q1, q2: Charges
            r_vec: Separation vector (q2 position - q1 position)
            
        Returns:
            Force vector (force on q2 due to q1)
        """
        r_vec = np.asarray(r_vec)
        r = np.linalg.norm(r_vec)
        
        if r == 0:
            return np.array([np.inf, np.inf, np.inf])
        
        r_hat = r_vec / r
        F_mag = self.k_e * q1 * q2 / (r * r)
        
        return F_mag * r_hat
    
    def potential_energy(self, q1: float, q2: float, r: float) -> float:
        """
        Calculate potential energy U = kq₁q₂/r.
        
        Args:
            q1, q2: Charges
            r: Separation
            
        Returns:
            Potential energy in Joules
        """
        if r == 0:
            return np.inf if q1 * q2 > 0 else -np.inf
        
        return self.k_e * q1 * q2 / r


# =============================================================================
# Eq 52: ElectricPotentialField - Potential calculator
# =============================================================================

class ElectricPotentialField:
    """
    Batch 5, Eq 52: Electric potential from point charges.
    
    ET Math: V(r) = (1/4πε₀)(q/r)
    
    Descriptor potential field from charge.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.k_e = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY)
    
    def potential_point_charge(self, q: float, r: float) -> float:
        """
        Calculate V(r) for single point charge.
        
        Args:
            q: Charge in Coulombs
            r: Distance in meters
            
        Returns:
            Potential in Volts
        """
        if r == 0:
            return np.inf if q > 0 else -np.inf
        
        return self.k_e * q / r
    
    def potential_multiple_charges(self, charges: List[float], positions: np.ndarray, 
                                   eval_point: np.ndarray) -> float:
        """
        Calculate total potential from multiple charges.
        
        ET Math: V_total = Σ V_i (superposition)
        
        Args:
            charges: List of charge values
            positions: Array of charge positions (N×3)
            eval_point: Point to evaluate potential (3D)
            
        Returns:
            Total potential
        """
        positions = np.asarray(positions)
        eval_point = np.asarray(eval_point)
        
        total_V = 0.0
        
        for q, pos in zip(charges, positions):
            r_vec = eval_point - pos
            r = np.linalg.norm(r_vec)
            
            if r > 0:
                total_V += self.k_e * q / r
            else:
                return np.inf if q > 0 else -np.inf
        
        return total_V
    
    def equipotential_radius(self, q: float, V_target: float) -> float:
        """
        Find radius where V(r) = V_target.
        
        Args:
            q: Charge
            V_target: Target potential
            
        Returns:
            Radius in meters
        """
        if V_target == 0:
            return np.inf
        
        return self.k_e * q / V_target


# =============================================================================
# Eq 53: ElectricFieldCalculator - E-field calculator
# =============================================================================

class ElectricFieldCalculator:
    """
    Batch 5, Eq 53: Electric field from point charges.
    
    ET Math: E(r) = (1/4πε₀)(q/r²) r̂
    
    Descriptor gradient field.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.k_e = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY)
    
    def field_magnitude(self, q: float, r: float) -> float:
        """
        Calculate |E| from point charge.
        
        Args:
            q: Charge
            r: Distance
            
        Returns:
            Field magnitude in V/m
        """
        if r == 0:
            return np.inf
        
        return self.k_e * abs(q) / (r * r)
    
    def field_vector(self, q: float, source_pos: np.ndarray, eval_point: np.ndarray) -> np.ndarray:
        """
        Calculate E-field vector at evaluation point.
        
        Args:
            q: Source charge
            source_pos: Position of charge
            eval_point: Point to evaluate field
            
        Returns:
            E-field vector
        """
        r_vec = np.asarray(eval_point) - np.asarray(source_pos)
        r = np.linalg.norm(r_vec)
        
        if r == 0:
            return np.array([np.inf, np.inf, np.inf])
        
        r_hat = r_vec / r
        E_mag = self.k_e * q / (r * r)
        
        return E_mag * r_hat
    
    def field_multiple_charges(self, charges: List[float], positions: np.ndarray,
                               eval_point: np.ndarray) -> np.ndarray:
        """
        Calculate total E-field from multiple charges.
        
        ET Math: E_total = Σ E_i (vector superposition)
        
        Args:
            charges: Charge values
            positions: Charge positions (N×3)
            eval_point: Evaluation point
            
        Returns:
            Total E-field vector
        """
        positions = np.asarray(positions)
        eval_point = np.asarray(eval_point)
        
        E_total = np.zeros(3)
        
        for q, pos in zip(charges, positions):
            E_total += self.field_vector(q, pos, eval_point)
        
        return E_total


# =============================================================================
# Eq 54: MagneticFieldCalculator - B-field from currents
# =============================================================================

class MagneticFieldCalculator:
    """
    Batch 5, Eq 54: Magnetic field from current-carrying wire.
    
    ET Math: B = (μ₀/2π)(I/r)
    
    Rotational descriptor field from moving charges.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.mu_0 = VACUUM_PERMEABILITY
    
    def field_straight_wire(self, current: float, r: float) -> float:
        """
        Calculate B-field magnitude around infinite straight wire.
        
        Args:
            current: Current in Amperes
            r: Perpendicular distance from wire
            
        Returns:
            Magnetic field in Tesla
        """
        if r == 0:
            return np.inf
        
        return (self.mu_0 / (2.0 * np.pi)) * current / r
    
    def field_circular_loop(self, current: float, radius: float, z: float) -> float:
        """
        Calculate B-field on axis of circular current loop.
        
        ET Math: B_z = (μ₀IR²)/(2(R²+z²)^(3/2))
        
        Args:
            current: Current in loop
            radius: Loop radius
            z: Distance along axis from center
            
        Returns:
            Axial magnetic field
        """
        R_sq = radius * radius
        z_sq = z * z
        denominator = (R_sq + z_sq) ** 1.5
        
        if denominator == 0:
            return np.inf
        
        return (self.mu_0 * current * R_sq) / (2.0 * denominator)
    
    def field_solenoid(self, current: float, turns_per_length: float) -> float:
        """
        Calculate B-field inside ideal solenoid.
        
        ET Math: B = μ₀nI
        
        Args:
            current: Current
            turns_per_length: n (turns/meter)
            
        Returns:
            Interior magnetic field
        """
        return self.mu_0 * turns_per_length * current


# =============================================================================
# Eq 55: LorentzForceCalculator - Force on moving charges
# =============================================================================

class LorentzForceCalculator:
    """
    Batch 5, Eq 55: Lorentz force on moving charge.
    
    ET Math: F = q(E + v × B)
    
    Combined electric + magnetic descriptor forces.
    """
    
    @staticmethod
    def force_vector(charge: float, E_field: np.ndarray, velocity: np.ndarray, 
                     B_field: np.ndarray) -> np.ndarray:
        """
        Calculate Lorentz force F = q(E + v×B).
        
        Args:
            charge: Charge in Coulombs
            E_field: Electric field vector (V/m)
            velocity: Velocity vector (m/s)
            B_field: Magnetic field vector (T)
            
        Returns:
            Force vector in Newtons
        """
        E = np.asarray(E_field)
        v = np.asarray(velocity)
        B = np.asarray(B_field)
        
        # F = q(E + v×B)
        return charge * (E + np.cross(v, B))
    
    @staticmethod
    def magnetic_force_only(charge: float, velocity: np.ndarray, B_field: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic force F = qv×B.
        
        Args:
            charge: Charge
            velocity: Velocity vector
            B_field: Magnetic field vector
            
        Returns:
            Magnetic force vector
        """
        v = np.asarray(velocity)
        B = np.asarray(B_field)
        
        return charge * np.cross(v, B)
    
    @staticmethod
    def cyclotron_radius(charge: float, mass: float, velocity: float, B_field: float) -> float:
        """
        Calculate radius of circular motion in uniform B-field.
        
        ET Math: r = mv/(qB)
        
        Args:
            charge: Particle charge
            mass: Particle mass
            velocity: Speed perpendicular to B
            B_field: Magnetic field magnitude
            
        Returns:
            Cyclotron radius
        """
        if B_field == 0:
            return np.inf
        
        return (mass * velocity) / (abs(charge) * B_field)
    
    @staticmethod
    def cyclotron_frequency(charge: float, mass: float, B_field: float) -> float:
        """
        Calculate cyclotron angular frequency ω = qB/m.
        
        Args:
            charge: Particle charge
            mass: Particle mass
            B_field: Magnetic field
            
        Returns:
            Angular frequency in rad/s
        """
        return abs(charge) * B_field / mass


# =============================================================================
# Eq 56: EMEnergyCalculator - Field energy density
# =============================================================================

class EMEnergyCalculator:
    """
    Batch 5, Eq 56: Electromagnetic field energy density.
    
    ET Math: u = (ε₀E²/2) + (B²/2μ₀)
    
    Descriptor field energy content.
    """
    
    def __init__(self):
        """Initialize with vacuum constants."""
        self.eps_0 = VACUUM_PERMITTIVITY
        self.mu_0 = VACUUM_PERMEABILITY
    
    def electric_energy_density(self, E_field: Union[float, np.ndarray]) -> float:
        """
        Calculate u_E = ε₀E²/2.
        
        Args:
            E_field: Electric field magnitude or vector
            
        Returns:
            Energy density in J/m³
        """
        if isinstance(E_field, np.ndarray):
            E_sq = np.dot(E_field, E_field)
        else:
            E_sq = E_field * E_field
        
        return 0.5 * self.eps_0 * E_sq
    
    def magnetic_energy_density(self, B_field: Union[float, np.ndarray]) -> float:
        """
        Calculate u_B = B²/(2μ₀).
        
        Args:
            B_field: Magnetic field magnitude or vector
            
        Returns:
            Energy density in J/m³
        """
        if isinstance(B_field, np.ndarray):
            B_sq = np.dot(B_field, B_field)
        else:
            B_sq = B_field * B_field
        
        return 0.5 * B_sq / self.mu_0
    
    def total_energy_density(self, E_field: Union[float, np.ndarray],
                            B_field: Union[float, np.ndarray]) -> float:
        """
        Calculate total u = u_E + u_B.
        
        Args:
            E_field: Electric field
            B_field: Magnetic field
            
        Returns:
            Total EM energy density
        """
        u_E = self.electric_energy_density(E_field)
        u_B = self.magnetic_energy_density(B_field)
        
        return u_E + u_B
    
    def poynting_vector(self, E_field: np.ndarray, B_field: np.ndarray) -> np.ndarray:
        """
        Calculate Poynting vector S = (E×B)/μ₀.
        
        Energy flux density.
        
        Args:
            E_field: Electric field vector
            B_field: Magnetic field vector
            
        Returns:
            Poynting vector (W/m²)
        """
        E = np.asarray(E_field)
        B = np.asarray(B_field)
        
        return np.cross(E, B) / self.mu_0


# =============================================================================
# Eq 57: FineStructureConstant - EM coupling
# =============================================================================

class FineStructureConstant:
    """
    Batch 5, Eq 57: Fine structure constant calculator.
    
    ET Math: α = e²/(4πε₀ℏc) ≈ 1/137.036
    
    Dimensionless EM coupling from manifold structure (Eq 183).
    """
    
    def __init__(self):
        """Initialize with fundamental constants."""
        self.alpha = FINE_STRUCTURE_CONSTANT
        self.alpha_inv = FINE_STRUCTURE_INVERSE
    
    def get_alpha(self) -> float:
        """Get fine structure constant α."""
        return self.alpha
    
    def get_alpha_inverse(self) -> float:
        """Get α⁻¹ ≈ 137.036."""
        return self.alpha_inv
    
    def calculate_from_fundamentals(self) -> float:
        """
        Calculate α from fundamental constants.
        
        ET Math: α = e²/(4πε₀ℏc)
        
        Returns:
            Calculated α (should match FINE_STRUCTURE_CONSTANT)
        """
        from ..core.constants import PLANCK_CONSTANT_HBAR
        
        numerator = ELEMENTARY_CHARGE ** 2
        denominator = 4.0 * np.pi * VACUUM_PERMITTIVITY * PLANCK_CONSTANT_HBAR * SPEED_OF_LIGHT
        
        return numerator / denominator
    
    def verify_consistency(self) -> bool:
        """
        Verify α matches calculation from fundamentals.
        
        Returns:
            True if consistent within tolerance
        """
        calculated = self.calculate_from_fundamentals()
        difference = abs(calculated - self.alpha)
        
        return difference < 1e-10


# =============================================================================
# Eq 58: VacuumImpedance - Free space impedance
# =============================================================================

class VacuumImpedance:
    """
    Batch 5, Eq 58: Characteristic impedance of free space.
    
    ET Math: Z₀ = √(μ₀/ε₀) ≈ 377 Ω
    
    Manifold resistance to EM wave propagation.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.Z_0 = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)
    
    def get_impedance(self) -> float:
        """
        Get vacuum impedance.
        
        Returns:
            Z₀ in Ohms (≈ 376.73 Ω)
        """
        return self.Z_0
    
    def wave_speed_relation(self) -> float:
        """
        Verify c = 1/√(μ₀ε₀).
        
        Returns:
            Calculated speed of light
        """
        return 1.0 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
    
    def impedance_from_speed(self) -> float:
        """
        Calculate Z₀ = μ₀c.
        
        Alternative formula.
        
        Returns:
            Vacuum impedance
        """
        return VACUUM_PERMEABILITY * SPEED_OF_LIGHT


# =============================================================================
# Eq 59: CoulombConstantCalculator - Electrostatic constant
# =============================================================================

class CoulombConstantCalculator:
    """
    Batch 5, Eq 59: Coulomb's constant k_e calculator.
    
    ET Math: k_e = 1/(4πε₀) ≈ 8.99×10⁹ N·m²/C²
    
    Manifold radial coupling constant.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.k_e = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY)
    
    def get_constant(self) -> float:
        """
        Get Coulomb's constant.
        
        Returns:
            k_e in SI units
        """
        return self.k_e
    
    def permittivity_relation(self) -> float:
        """
        Calculate ε₀ from k_e.
        
        ET Math: ε₀ = 1/(4πk_e)
        
        Returns:
            Vacuum permittivity
        """
        return 1.0 / (4.0 * np.pi * self.k_e)
    
    def force_at_unit_distance(self, q1: float = 1.0, q2: float = 1.0) -> float:
        """
        Calculate force between unit charges at 1 meter.
        
        Args:
            q1, q2: Charges (default 1 C each)
            
        Returns:
            Force in Newtons
        """
        return self.k_e * q1 * q2


# =============================================================================
# Eq 60: MagneticConstantCalculator - Permeability
# =============================================================================

class MagneticConstantCalculator:
    """
    Batch 5, Eq 60: Magnetic constant (permeability of free space).
    
    ET Math: μ₀ = 4π×10⁻⁷ H/m (exact by definition)
    
    Manifold rotational coupling constant.
    """
    
    def __init__(self):
        """Initialize calculator."""
        self.mu_0 = VACUUM_PERMEABILITY
    
    def get_constant(self) -> float:
        """
        Get vacuum permeability.
        
        Returns:
            μ₀ in H/m
        """
        return self.mu_0
    
    def exact_value(self) -> float:
        """
        Calculate exact defined value μ₀ = 4π×10⁻⁷.
        
        Returns:
            Exact μ₀
        """
        return 4.0 * np.pi * 1e-7
    
    def speed_of_light_relation(self) -> float:
        """
        Calculate c from μ₀ and ε₀.
        
        ET Math: c = 1/√(μ₀ε₀)
        
        Returns:
            Speed of light
        """
        return 1.0 / np.sqrt(self.mu_0 * VACUUM_PERMITTIVITY)
    
    def verify_consistency(self) -> bool:
        """
        Verify μ₀ matches exact definition.
        
        Returns:
            True if consistent
        """
        exact = self.exact_value()
        difference = abs(self.mu_0 - exact)
        
        return difference < 1e-15


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'CoulombForceCalculator',
    'ElectricPotentialField',
    'ElectricFieldCalculator',
    'MagneticFieldCalculator',
    'LorentzForceCalculator',
    'EMEnergyCalculator',
    'FineStructureConstant',
    'VacuumImpedance',
    'CoulombConstantCalculator',
    'MagneticConstantCalculator',
]
