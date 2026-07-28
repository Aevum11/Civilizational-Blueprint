# The ∂I Lattice-Aware Fractal: A Lattice-Adaptive Complex Dynamical System

## What It Is

The ∂I Lattice-Aware fractal is a complex dynamical system in which the polynomial degree of the iteration map is not fixed — it is determined at every step by the orbit's position relative to a multiplicative lattice on ℂ.

In standard fractal iteration (Mandelbrot, Julia, Multibrot), the map is:

$$z_{n+1} = z_n^p + c \quad \text{where } p \text{ is constant for all } n$$

In the ∂I system, the map is:

$$z_{n+1} = \Psi_n \cdot z_n^{\,p(z_n, n)} + \epsilon(z_n) + c$$

where:

- **p(z_n, n)** is a function that returns the polynomial degree based on where z_n sits in the lattice. It can be 1, 2, 3, 4, 6, or 12 — and it changes at every iteration step.
- **Ψ_n** is a bounded periodic modulation (amplitude between 0.711 and 1.289).
- **ε(z_n)** is a small additive perturbation from all lattice families.
- **c** is the pixel coordinate (Mandelbrot parameterization: z₀ = 0, c = pixel).

The key claim is that this is not a fixed-exponent system with a parameter twist — the exponent itself is a dynamical variable derived from the orbit's own position.

---

## The Lattice

The lattice is a multiplicative grid on the complex plane. Every complex number z = r·e^{iθ} is projected onto it via:

$$k_r = \text{round}(N_L \cdot \log_2 r), \quad k_\theta = \text{round}(N_L \cdot \theta / \ln 2)$$

where N_L = 27720 = lcm(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).

This lattice resolution is the smallest integer where every sublattice family from d=1 through d=12 exists as a native divisor. At 12-division resolution, only d ∈ {1, 2, 3, 4, 6, 12} appear (divisors of 12). The extended families d=5, 7, 8, 9, 10, 11 require higher resolution:

| Family d | First appears at | lcm needed |
|---|---|---|
| d=8 | 24-division | lcm(12, 8) = 24 |
| d=9 | 36-division | lcm(12, 9) = 36 |
| d=5, d=10 | 60-division | lcm(1..5) = 60 |
| d=7 | 420-division | lcm(1..7) = 420 |
| d=11 | 27720-division | lcm(1..11) = 27720 |

The sublattice family d at lattice coordinate k is:

$$d = \frac{27720}{\gcd(|k|, 27720)}$$

This is straightforward number theory. When |k| is divisible by many factors of 27720, the GCD is large and d is small (coarse sublattice). When |k| shares few factors with 27720, d is large (fine sublattice). The 12 possible d-values correspond to 12 sublattice families with different densities.

---

## How the Exponent Is Chosen

At each iteration step, the orbit z_n is projected onto the lattice. The projection gives a lattice coordinate k_r and a residual ε_r measuring how far z_n is from the nearest lattice point, in cents (hundredths of a lattice step):

$$\varepsilon_r = \left(N_L \cdot \log_2 |z_n| - k_r\right) \times \frac{1200}{N_L}$$

A **tightness** measure quantifies proximity to the lattice:

$$t_r = \frac{100}{100 + |\varepsilon_r|}$$

This is 1.0 when the orbit sits exactly on a lattice point (ε = 0) and decreases as the orbit moves away. The threshold is t_r = 2/3, which corresponds to |ε_r| = 50 cents — exactly half a lattice step.

**If t_r > 2/3** (orbit is near a lattice point): the sublattice family d_r at that lattice point determines the exponent. The map p = 12/d, so:

| Nearest family d | Exponent p = 12/d | Step character |
|---|---|---|
| d=1 | p=12 | dodecic |
| d=2 | p=6 | sextic |
| d=3 | p=4 | quartic |
| d=4 | p=3 | cubic |
| d=6 | p=2 | quadratic |
| d=12 | p=1 | linear |

**If t_r ≤ 2/3** (orbit is between lattice points — the "∂I boundary"): the exponent comes from a fixed palindromic cycle that visits all six exponent values in a deterministic sequence:

$$p_k = [1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1, 12] \quad \text{indexed by } k = n \bmod 12$$

This sequence has the symmetry p_k = p_{12−k} (it is a palindrome). Its mean is 10/3 ≈ 3.333.

---

## Why This Is Not a Known Fractal Type

Here is the precise comparison:

**Mandelbrot set:** z → z² + c. Exponent is 2, always, everywhere, every step.

**Multibrot sets:** z → z^p + c. Exponent is p, a fixed integer chosen once before iteration begins. p=3 gives a cubic Multibrot, p=4 gives a quartic Multibrot, etc. The exponent never changes during iteration.

**Burning Ship, Tricorn, etc.:** These apply conjugation or absolute-value operations to z, but the polynomial degree is still fixed.

**Newton fractals:** These iterate Newton's method on a fixed polynomial. The rational function f(z)/f'(z) has a fixed degree determined by the polynomial's degree.

**Parameter-space fractals (Lyapunov, etc.):** These vary a parameter across the image plane, but the iteration rule at each pixel is fixed once the parameter is set.

**The ∂I system:** The exponent changes at every single step. Two orbits starting from nearby pixels may have completely different exponent sequences, because their orbits visit different lattice regions. The same orbit passes through quadratic, quartic, dodecic, and linear dynamics within a single escape trajectory. The topology of the connected set's boundary is not that of any fixed-degree polynomial Julia set — it is a hybrid that depends on the statistical distribution of exponents visited by orbits near the boundary.

The closest known relative would be a random iteration system (iterated function system / IFS), but this is not random — the exponent selection is deterministic, computed from the orbit's own coordinates via a fixed lattice projection. It is a feedback loop: the orbit determines the map, and the map determines the orbit.

---

## The Perturbation Term

In addition to the dominant z^p term, the iteration includes a perturbation that sums contributions from all 12 sublattice families at once:

$$\epsilon(z_n) = \frac{1}{12} \sum_{d \in \{1..12\}} w(d) \cdot |z_n|^{12/d} \cdot e^{i(12/d) \cdot \arg(z_n)}$$

where w(d) is a normalized weight for each family. The factor 1/12 ensures this term is small relative to the dominant power — it adds fine structure without overriding the dominant dynamics.

This is analogous to perturbation theory in physics: the dominant power sets the topology, and the 12-family sum adds texture from all sublattice scales simultaneously.

---

## The Shimmer Modulation

The scalar Ψ_n is a bounded periodic function of the step index:

$$\Psi_n = 1 + \frac{1}{\sqrt{12}} \cdot \sin\!\left(\frac{2\pi (n \bmod 12)}{12}\right)$$

This modulates the amplitude of the dominant term with a 12-fold periodic envelope. Its range is approximately [0.711, 1.289]. It is bounded away from zero, so it never kills the iteration, and bounded above, so it never causes artificial blowup.

---

## Distance Estimation

For rendering, the system tracks the derivative of the iteration map for distance estimation (DE). The Jacobian of the dominant term is:

$$f'(z_n) = \Psi_n \cdot p(z_n, n) \cdot z_n^{\,p(z_n, n) - 1}$$

The derivative accumulation follows the standard chain rule:

$$dz_{n+1} = f'(z_n) \cdot dz_n + 1$$

and the distance estimate at escape is:

$$\text{DE} = \frac{2\,|z_n|\,\ln|z_n|}{|dz_n|}$$

This provides sub-pixel boundary rendering at arbitrary zoom, using the same DE technique that works for standard Mandelbrot rendering — adapted to a variable-degree system by using the instantaneous p at each step.

---

## The Effective Power

Because the exponent varies, the smooth iteration count formula needs an effective average power for normalization:

$$p_{\text{eff}} = \frac{1}{12}\sum_{k=0}^{11} p_k = \frac{1+2+3+4+1+6+1+4+3+2+1+12}{12} = \frac{10}{3}$$

This is the mean of the palindromic fallback sequence. The smooth escape count is:

$$\mu = n + 1 - \frac{\ln(\ln|z_n|) - \ln(\ln R)}{\ln(p_{\text{eff}})}$$

---

## Summary

The ∂I Lattice-Aware fractal is defined by a single structural idea: **the polynomial degree of the iteration map is a deterministic function of the orbit's position in a multiplicative lattice on ℂ.** The lattice has 27720 divisions per octave (chosen as lcm(1..11) to accommodate all 12 sublattice families). The orbit-to-exponent mapping uses the GCD of the lattice coordinate with 27720 to determine the sublattice family, and the exponent is 12/d where d is that family index. When the orbit falls between lattice points, a palindromic cycle of exponents takes over.

No fixed-exponent iteration (Mandelbrot, Julia, Multibrot) can produce this behavior, because no fixed-exponent system has orbits that pass through six different polynomial degrees within a single escape trajectory. The structure of the connected set boundary reflects this: regions dominated by quadratic dynamics have Mandelbrot-like topology, regions dominated by quartic dynamics have quartic Multibrot topology, and the transitions between them — controlled by the lattice — create boundary geometry that does not occur in any fixed-degree system.

The system is fully deterministic, uses standard complex arithmetic, and is rendered with standard escape-time + distance estimation techniques. The only novel element is the lattice-adaptive exponent.
