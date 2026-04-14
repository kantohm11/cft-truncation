# Local Coordinates and ρ₀

## T-domain geometry

The SC derivative f'(z) = C√(z²-p²)/(z(z²-1)) maps the UHP to a T-shaped domain:
- **3 semi-infinite arms** at poles z = 0, ±1
- **2 reflex corners** (270°) at zeros z = ±p, where p = ℓ/√(4+ℓ²)
- **T arm width** = ℓ (from residue: Re(f) shifts by ℓ as arg(z) goes 0 → π near z = 0)

### The T-domain in the f-plane

The SC map f(z) = ∫ f'(z) dz maps UHP to the following domain. The R corner
is placed at f = i (i.e., σ = 0, τ = 1) by the ρ₀ convention.

```
     τ (Im f)
     |
     |         ║             ║
     |         ║   T arm     ║
     |         ║   width ℓ   ║
     |         ║             ║
  1  ─ ─ ─ ─ ─╝             ╚─ ─ ─ ─ ─ ─ ─   ← upper boundary (τ = 1)
     |     R corner       L corner
     |      (0, 1)        (ℓ, 1)
     |         :              :
     |  R arm  :              :  L arm
     |  ←────  :              :  ────→
     |         :              :
  0  ─ ─ ─ ─ ─:─ ─ ─ ─ ─ ─ ─:─ ─ ─ ─ ─ ─ ─   ← lower boundary (τ = 0)
     |         :              :
     └─────────┼──────────────┼──────────→ σ (Re f)
              σ=0            σ=ℓ
```

- R arm: extends from R corner (σ = 0) to σ → -∞. Strip of width 1 in τ.
- L arm: extends from L corner (σ = ℓ) to σ → -∞. Also width 1 in τ.
- T arm: extends upward from τ = 1 to τ → +∞. Width ℓ in σ.
- Both L and R arm tips go to σ → -∞ (same direction, shifted by ℓ).
- The outer boundary (from z → ±∞) connects the arms at moderate σ.

For **centered plots**: post-shift by -ℓ/2 gives corners at (±ℓ/2, 1).

### Boundary assignments (tracing z along the real axis)

- z ∈ (1, +∞): R arm lower boundary, τ = 0
- z ∈ (p, 1): R arm upper boundary, τ = 1
- z ∈ (0, p): T arm right wall, σ = 0
- z ∈ (-p, 0): T arm left wall, σ = ℓ
- z ∈ (-1, -p): L arm lower boundary, τ = 1
- z ∈ (-∞, -1): L arm upper boundary, τ = 2 (then curves back)

## Domain symmetries

**Z₂ translation**: f(-z) = f(z) + ℓ. From f'(-z) = -f'(z) (f' is odd).

**Anti-holomorphic reflection**: f(-z̄) = -(f(z))* + ℓ. This maps
(σ, τ) → (ℓ - σ, τ) — left-right reflection about σ = ℓ/2.

## Local coordinates and ρ₀

### The integration constant

The SC map near puncture x_i:

```
f(x_i + ζ) = Res_i · log ζ + ρ₀ + ρ₁ζ + ρ₂ζ² + ...
```

The ρ₁, ρ₂, ... are determined by f'. The ρ₀ is the integration constant.

### Local coordinates with shifted mouths

The naive local coordinate ξ = exp(π · f) has its mouth (|ξ| = 1) at σ = 0
for ALL arms. But the L corner is at σ = ℓ, not σ = 0. To place each arm's
mouth at its OWN corner:

```
  ξ_R = exp(π · f)           mouth at σ = 0 = R corner   ✓
  ξ_L = exp(π · (f − ℓ))     mouth at σ = ℓ = L corner   ✓
  ξ_T = exp(iπ/ℓ · f)        (T arm, different coeff_factor)
```

The L arm shift by -ℓ ensures |ξ_L| = 1 at σ = ℓ (the L corner), not at σ = 0.

### Effect on ρ₀ and α

In the code: the local coordinate is built from the arm's f-expansion using ρ₀.
The shift ξ_L = exp(π(f - ℓ)) is equivalent to using ρ₀^L with an effective
real part equal to ρ₀^R (the ℓ absorbed into the exponent):

```
ρ₀^R = ∫_p^1 g(z) dz − (1/π) log(1−p)     [real, from numerical integration]
ρ₀^L = ρ₀^R + i                             [purely imaginary shift, NOT +ℓ+i]
ρ₀^T = i − f_T^{raw}(p)                     [from corner matching f_T(p) = i]
```

**Key**: ρ₀^L − ρ₀^R = i (purely imaginary). This gives:
- **|α_L| = |α_R|** (symmetric local coordinates)
- **N^{LL}\_{m,k} = (-1)^{m+k} N^{RR}\_{m,k}** (standard Z₂ Neumann relation)

### Why ρ₀^L = ρ₀^R + i (not + ℓ + i)

The Z₂ of the SC map gives ρ₀^L = ρ₀^R + ℓ + i (from f(-z) = f(z) + ℓ and
log(-ζ) = log(ζ) - iπ). But the LOCAL COORDINATE for the L arm uses
ξ_L = exp(π(f - ℓ)), which absorbs the ℓ shift. In the code, this means the
EFFECTIVE ρ₀ for the L arm exponent is ρ₀^R + ℓ + i - ℓ = ρ₀^R + i.

Equivalently: ρ₀^L in the code is set to ρ₀^R + i (not ρ₀^R + ℓ + i), and
the code builds ξ_L = exp(π · f_L) where f_L = (1/π)log ζ + ρ₀^L + ..., which
automatically gives ξ_L = exp(π(f - ℓ)) because the L arm's f-expansion
near z = -1 is f = f_R(-ζ) + ℓ (from Z₂) and the ρ₀^L = ρ₀^R + i absorbs
the -ℓ relative to the Z₂ formula.

### ρ₀^R computation

Singularity-subtracted numerical integration on [p, 1]:

```
ρ₀^R = ∫_p^1 g(z) dz − (1/π) log(1−p)
```

where g(z) = f'(z) − (1/π)/(z−1) is smooth (pole subtracted analytically).
Result is real. ~10 digit accuracy with 10000 Simpson points.

This targets Re(f(p)) = 0 (mouth at R corner).

### ρ₀^T computation

The T arm series at z = p (converges well, |p| < 1):

```
f_T^{raw}(p) = (-iℓ/π) log(p) + Σ ρ_n^T p^n
ρ₀^T = i − f_T^{raw}(p)
```

This ensures f_T(p) = f_R(p) = i (arm expansions agree at the corner).

## BPZ sign convention

For the U(1) current J (weight h = 1):

```
bpz(J_{-k}) = (-1)^{k+1} J_k
```

Odd modes: no sign. Even modes: sign flip.
BPZ sign for partition λ: ∏ (-1)^{k_i+1} = (-1)^{#even_parts}.
Fixed in `src/BPZ.jl`.

## |B^open⟩ capping the T arm

Inserting |B^open⟩ seals the T arm notch, producing a straight strip of width ℓ
between the R and L mouths. The raw vertex contracted with |B⟩ gives a propagator
with d = πℓ (code's width-1 convention).
