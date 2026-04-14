# Session Memo: ρ₀ and |B^open⟩ Contraction (April 2026)

## What was accomplished

### BPZ sign fix (committed)
- U(1) current (weight h=1): bpz(J_{-k}) = (-1)^{k+1} J_k
- BPZ sign for partition λ: ∏(-1)^{k_i+1} = (-1)^{#even_parts}
- NOT (-1)^N (that's the Virasoro/weight-2 convention)
- Verified via the strip 2-point vertex

### ρ₀ computation (committed)
- ρ₀ is the integration constant of the SC map f(z) = ∫f'(z)dz
- Computed geometrically: mouth (|ξ|=1) at corner for L/R arms
- ρ₀^R: singularity-subtracted numerical integration on [p,1]
- ρ₀^L = ρ₀^R + i (absorbs the ℓ shift from ξ_L = exp(π(f-ℓ)))
- ρ₀^T: corner matching f_T(p) = f_R(p) = i
- Result: |α_L| = |α_R| (symmetric), N^{LL} = N^{RR}
- Domain uncentered (R corner σ=0, L corner σ=ℓ); center by -ℓ/2 for plots

### T-domain geometry (committed, notebook 07)
- SC map f'(z) = C√(z²-p²)/(z(z²-1)): 2 corners + 3 infinities
- T arm width = ℓ exactly (verified numerically)
- Corners at τ = 1, both arms at τ ∈ [0,1]
- Symmetries: f(-z) = f(z) + ℓ (glide), f(-z̄) = -(f(z))* + ℓ (reflection)
- sqrt branch fix needed for numerical integration: `if imag(s) < 0; s = -s; end`

### Design doc (committed)
- `docs/dev/local_coordinates_and_strip.md`: full explanation with ASCII art

## Open problem: |B^open⟩ contraction with the vertex

### The question
Contract the raw T-vertex with |B^open⟩ on the T arm → should give a propagator
with d = πℓ (width-1 convention). This is the key cross-check between the vertex
formalism and the strip geometry.

### What we found (NOT definitive — needs more thought)
- The vertex entry V([k,k], vac, vac) grows with k. The growth rate depends on |α_T|.
- B([k,k]) = 1/√2 for all k (does NOT decay with level).
- The product B·V grows, and the partial sum over T arm states does NOT converge
  at finite truncation for the tested parameter ranges.
- The growth per 2 levels is approximately C/|α_T|², where C is a geometry-dependent
  factor (the "junction interaction growth"):
  - C ≈ 22 at ℓ = 0.5
  - C ≈ 6 at ℓ = 1
  - C ≈ 2 at ℓ = 2
  - C ≈ 1.2 at ℓ = 3-4
- Convergence requires |α_T|² > C, i.e., |α_T| > √C.
- At the corner (mouth at corner): |α_T| ~ exp(-π/ℓ) → 0, far below threshold.
- At ρ₀^T = 0: |α_T| = 1, still below √C for ℓ ≤ 4.
- The off-diagonal fraction does NOT decrease with h_phys in the tested range.

### Caveats (things that might be wrong or incomplete)
- The non-convergence could be an artifact of how we extract d and off-diagonal
  from a partially-converged sum. The partial sums oscillate.
- The exact answer IS finite (from the product formula η(e^{-2T})^{-1/2}), so the
  physics works — the question is whether the VERTEX BASIS expansion converges
  term by term.
- The C factor was measured from only 3-4 data points (levels 4,6,8) — the
  asymptotic behavior at very high levels might be different.
- We did NOT test with |α_T| > √C (which should converge). This would confirm
  whether the issue is purely the convergence rate.
- The relationship between |α_T|, the T arm entrance length, and the physical
  propagation distance needs more careful analysis.
- log|α_T^corner| ≈ -π/ℓ as ℓ → 0, meaning the corner is ~1 strip-width deep
  inside the T arm in physical units. This geometric fact might have implications
  for the convergence.

## Code state
- All 745 tests pass
- CACHE_VERSION bumped to v2_rho0
- Notebooks 05 (B^open contraction) and 07 (T-domain geometry) exist and run clean
- The ρ₀^T in the committed code matches at the CORNER (|α_T| ~ exp(-π/ℓ)),
  which is in the non-convergent regime. For the contraction to work, ρ₀^T
  may need to be set differently (e.g., ρ₀^T = 0 giving |α_T| = 1).

## Key files
- `src/LocalCoordinates.jl`: ρ₀ computation (_compute_rho0, _compute_rho0_R, _compute_rho0_T)
- `src/BPZ.jl`: U(1) BPZ sign convention
- `src/Cache.jl`: CACHE_VERSION = "v2_rho0"
- `docs/dev/local_coordinates_and_strip.md`: design doc
- `experiments/notebooks/07_t_domain_geometry.jl`: T-domain visualization
- `experiments/notebooks/05_boundary_vertex_contraction.jl`: B^open contraction
