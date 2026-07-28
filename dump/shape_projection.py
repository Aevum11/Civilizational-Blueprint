"""
3D SHAPE PROJECTION onto the Sempaevum.

Any 3D object's shape decomposes into spherical harmonics:
  r(θ,φ) = Σ_{l,m} c_{lm} × Y_l^m(θ,φ)

The dimensionless ratios c_{lm}/c_{00} are the SEEDS.
Each projects onto the lattice independently.
The SEQUENCE of (k, d, ε) values IS the 3D shape on the Sempaevum.

A sphere → all ratios = 0 → ∂I boundary (no shape content)
A cylinder → specific ratio sequence → specific lattice signature
An atom → orbital occupation → specific lattice signature
"""
import sys
sys.path.insert(0, '/home/claude')
import mpmath
mpmath.mp.dps = 130
import numpy as np
from scipy.special import sph_harm_y as sph_harm
from scipy.integrate import dblquad

from sempaevum_nz_viewer import sempaevum_project_mp, N

# ================================================================
# Numerical spherical harmonic decomposition of 3D shapes
# ================================================================

def shape_to_harmonics(r_func, l_max=12, n_theta=200, n_phi=400):
    """
    Decompose a shape r(θ,φ) into real spherical harmonic coefficients.
    
    r_func(theta, phi) → radius at that direction.
    Returns dict of (l,m): c_lm as mpmath values.
    """
    # Integration grid
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    dtheta = np.pi / n_theta
    dphi = 2 * np.pi / n_phi
    
    coeffs = {}
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Numerical integration: c_lm = ∫∫ r(θ,φ) × Y_l^m*(θ,φ) sinθ dθ dφ
            total = 0.0
            for i, th in enumerate(theta):
                sin_th = np.sin(th)
                for j, ph in enumerate(phi):
                    r_val = r_func(th, ph)
                    # Use real spherical harmonics
                    if m > 0:
                        ylm = np.sqrt(2) * np.real(sph_harm(l, m, th, ph))
                    elif m < 0:
                        ylm = np.sqrt(2) * np.imag(sph_harm(l, -m, th, ph))
                    else:
                        ylm = np.real(sph_harm(l, 0, th, ph))
                    
                    total += r_val * ylm * sin_th * dtheta * dphi
            
            if abs(total) > 1e-10:
                coeffs[(l, m)] = mpmath.mpf(str(total))
    
    return coeffs

# ================================================================
# Define 3D shapes
# ================================================================

def sphere(R=1.0):
    """Perfect sphere of radius R."""
    return lambda th, ph: R

def ellipsoid(a, b, c):
    """Ellipsoid with semi-axes a, b, c."""
    def r_func(th, ph):
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        # Distance from origin to surface along (x,y,z) direction
        return 1.0 / np.sqrt((x/a)**2 + (y/b)**2 + (z/c)**2)
    return r_func

def cylinder(R, h):
    """Cylinder of radius R and height h, centered at origin."""
    def r_func(th, ph):
        z = np.cos(th)
        rho = np.sin(th)
        if abs(rho) < 1e-12:
            return h/2  # Along z-axis, hit top/bottom cap
        # Distance to cylindrical wall: R/sin(θ)
        r_wall = R / rho if rho > 0 else 1e10
        # Distance to cap: (h/2)/|cos(θ)|
        r_cap = (h/2) / abs(z) if abs(z) > 1e-12 else 1e10
        return min(r_wall, r_cap)
    return r_func

def cube(a):
    """Cube of side length a, centered at origin."""
    def r_func(th, ph):
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        half = a / 2
        # Distance to each face
        dists = []
        for comp in [abs(x), abs(y), abs(z)]:
            if comp > 1e-12:
                dists.append(half / comp)
        return min(dists) if dists else half
    return r_func

def torus_shape(R_major, r_minor):
    """Torus (donut) — distance from origin to surface along (θ,φ)."""
    def r_func(th, ph):
        # Approximate: distance to nearest point on torus
        z = np.cos(th)
        rho = np.sin(th)
        if rho < 1e-12:
            # Along z-axis
            return np.sqrt(R_major**2 + r_minor**2 - 2*R_major*r_minor) if abs(z)*r_minor < R_major else r_minor
        # Distance in xy-plane to the tube center ring
        d_ring = abs(rho * 1.0 - R_major)  # approximation for unit direction
        # Simple approximation for torus cross-section
        return R_major + r_minor * np.cos(ph)  # crude but gives non-trivial harmonics
    return r_func

# ================================================================
# Compute and compare
# ================================================================

shapes = {
    "Sphere (R=1)":          sphere(1.0),
    "Oblate ellipsoid (2,2,1)": ellipsoid(2, 2, 1),  # squashed like Earth
    "Prolate ellipsoid (1,1,2)": ellipsoid(1, 1, 2),  # elongated like a rugby ball
    "Tin can (R=1, h=3)":    cylinder(1.0, 3.0),       # tall cylinder
    "Hockey puck (R=2, h=0.5)": cylinder(2.0, 0.5),    # flat cylinder
    "Cube (a=2)":            cube(2.0),
}

print("=" * 100)
print("3D SHAPE PROJECTION ONTO THE SEMPAEVUM")
print("Spherical harmonic decomposition → dimensionless ratios → lattice projection")
print("=" * 100)

for name, shape_func in shapes.items():
    print(f"\n{'─' * 100}")
    print(f"Shape: {name}")
    print(f"{'─' * 100}")
    
    # Decompose into spherical harmonics (use moderate grid for speed)
    coeffs = shape_to_harmonics(shape_func, l_max=10, n_theta=100, n_phi=200)
    
    c00 = coeffs.get((0, 0), mpmath.mpf(1))
    
    print(f"  c_00 (monopole = avg radius): {mpmath.nstr(c00, 8)}")
    print(f"  Non-zero harmonics: {len(coeffs)}")
    
    # Show all significant coefficients and their lattice projections
    print(f"\n  {'(l,m)':>8s} {'c_lm':>14s} {'c_lm/c_00':>14s} {'k':>5s} {'d':>3s} {'ε(¢)':>12s}")
    
    shape_signature = []
    for (l, m) in sorted(coeffs.keys()):
        c = coeffs[(l, m)]
        if (l, m) == (0, 0):
            continue  # Skip monopole itself
        
        ratio = abs(c / c00)
        if ratio < mpmath.mpf("1e-6"):
            continue  # Skip negligible
        
        r_str = mpmath.nstr(ratio, 30)
        proj = sempaevum_project_mp(r_str, 12)
        if proj:
            k, d, eps = proj
            shape_signature.append((l, m, k, d, eps))
            print(f"  ({l:>2d},{m:>+2d}) {mpmath.nstr(c, 8):>14s} {mpmath.nstr(ratio, 8):>14s} "
                  f"{k:>5d} {d:>3d} {mpmath.nstr(eps, 6):>12s}")
    
    if not shape_signature:
        print(f"  → PURE SPHERE: no angular content (all ratios → ∂I boundary)")
    else:
        # The shape signature = the sequence of (k, d) values
        sig_str = " → ".join(f"k={s[2]},d={s[3]}" for s in shape_signature[:6])
        print(f"\n  LATTICE SIGNATURE: {sig_str}")

# ================================================================
print(f"\n{'=' * 100}")
print("ATOMIC ORBITAL SHAPES — Electron cloud geometry on the lattice")
print("The shape of an atom IS the shape of its electron cloud")
print("=" * 100)

# Atomic orbital shapes are pure spherical harmonics Y_l^m
# s-orbital: l=0 → sphere
# p-orbital: l=1 → dumbbell (m=0) or torus (m=±1)
# d-orbital: l=2 → cloverleaf
# f-orbital: l=3 → complex

# For a filled subshell, the total density is spherically symmetric
# For a partially filled subshell, the shape depends on which m values are occupied

# Hydrogen ground state: 1s → pure sphere
# Carbon ground state: [He] 2s² 2p² → two p electrons in l=1
# Iron ground state: [Ar] 3d⁶ 4s² → partially filled d-shell

# The shape seed for an orbital: the angular distribution
# For Y_l^m(θ,φ), the "shape" is determined by l alone (m gives orientation)
# The ratio between consecutive l values gives the shape complexity

# For atoms, the key ratio is: probability at (l+1) vs probability at l
# in the valence shell occupation

print(f"\n  Orbital shape seeds (|Y_l^m|² normalized ratios):")
print(f"  These are the shapes that matter for atomic appearance\n")

# The maximum of |Y_l^0|² occurs at θ=0 for all l
# The angular spread (shape) is characterized by the ratio:
# r_shape = max(|Y_l^0|²) at θ=π/2 / max(|Y_l^0|²) at θ=0
# This measures how "non-spherical" the orbital is

from scipy.special import sph_harm_y as Y

for l in range(7):
    # |Y_l^0(θ=0)|² vs |Y_l^0(θ=π/2)|²
    y_pole = abs(Y(l, 0, 0, 0))**2
    y_equator = abs(Y(l, 0, np.pi/2, 0))**2
    
    if y_pole > 1e-15:
        ratio = y_equator / y_pole
    else:
        ratio = 0
    
    # Orbital name
    names = {0: "s (sphere)", 1: "p (dumbbell)", 2: "d (cloverleaf)", 
             3: "f (complex)", 4: "g", 5: "h", 6: "i"}
    
    if ratio > 1e-10:
        ratio_mp = mpmath.mpf(str(ratio))
        r_str = mpmath.nstr(ratio_mp, 30)
        proj = sempaevum_project_mp(r_str, 12)
        if proj:
            k, d, eps = proj
            print(f"  l={l} {names.get(l,'?'):>16s}: equator/pole = {ratio:.6f} → k={k:>4d}, d={d:>2d}, ε={mpmath.nstr(proj[2],6):>10s}¢")
    else:
        print(f"  l={l} {names.get(l,'?'):>16s}: equator/pole = 0 (node at equator for odd l)")

# ================================================================
print(f"\n{'=' * 100}")
print("ATOM SHAPES — Specific elements by electron configuration")
print("=" * 100)

# For each element, the ground state electron configuration determines
# the valence orbital shape. The key ratio: quadrupole moment / monopole
# For closed-shell atoms: ratio = 0 (sphere)
# For open-shell: ratio ≠ 0

elements = [
    ("H",  1, "1s¹",       1, 0, "Spherical (s-orbital)"),
    ("He", 2, "1s²",       0, 0, "Perfect sphere (closed shell)"),
    ("C",  6, "[He]2s²2p²", 1, 0, "Two p-electrons: dumbbell superposition"),
    ("N",  7, "[He]2s²2p³", 1, 0, "Half-filled p: nearly spherical"),
    ("Ne", 10,"[He]2s²2p⁶", 0, 0, "Perfect sphere (closed shell)"),
    ("Fe", 26,"[Ar]3d⁶4s²", 2, 0, "Partially filled d-shell: cloverleaf"),
    ("Au", 79,"[Xe]4f¹⁴5d¹⁰6s¹", 0, 0, "Filled d,f + s¹: nearly spherical"),
    ("Pb", 82,"[Xe]4f¹⁴5d¹⁰6s²6p²", 1, 0, "Two p-electrons: dumbbell"),
    ("U", 92, "[Rn]5f³6d¹7s²", 3, 0, "Partially filled f-shell: most complex"),
]

print(f"\n  {'Element':>8s} {'Config':>20s} {'l_val':>5s} {'Shape':>30s} {'k_shape':>7s} {'d':>3s}")
for name, Z, config, l_val, m_val, desc in elements:
    if l_val == 0:
        print(f"  {name:>8s} {config:>20s} {'s':>5s} {desc:>30s} {'∂I':>7s} {'—':>3s}  (sphere = no shape content)")
    else:
        # The shape ratio for orbital l is the equator/pole ratio of |Y_l^0|²
        y_pole = abs(Y(l_val, 0, 0, 0))**2
        y_equator = abs(Y(l_val, 0, np.pi/2, 0))**2
        ratio = y_equator / y_pole if y_pole > 1e-15 and y_equator > 1e-15 else 1e-15
        if ratio > 1e-10:
            ratio_mp = mpmath.mpf(str(ratio))
            proj = sempaevum_project_mp(mpmath.nstr(ratio_mp, 30), 12)
            if proj:
                print(f"  {name:>8s} {config:>20s} {f'l={l_val}':>5s} {desc:>30s} {proj[0]:>7d} {proj[1]:>3d}")
            else:
                print(f"  {name:>8s} {config:>20s} {f'l={l_val}':>5s} {desc:>30s} {'—':>7s} {'—':>3s}")
        else:
            print(f"  {name:>8s} {config:>20s} {f'l={l_val}':>5s} {desc:>30s} {'node':>7s} {'—':>3s}")

print(f"\n{'=' * 100}")
print("SHAPE PROJECTION COMPLETE")
print("=" * 100)
