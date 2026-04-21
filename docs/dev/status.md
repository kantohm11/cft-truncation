# Implementation Status

## Current strategic direction (2026-04-21)

Working on **Strategy B** — CFT as an interface inside a fixed TQFT.
See [`../design/truncation_strategies.md`](../design/truncation_strategies.md)
for the full framing (what was tried, what's current, open questions).

The earlier "T-vertex as approximate module multiplication" framing
(including the modified-vertex $\widetilde V_\ell = e^{(H_L+H_R)\ell/2} V_\ell$
as the primary object) is abandoned.

## Core library: complete
All modules implemented and tested (747 tests passing):
TruncLaurent, SCMap, LocalCoordinates, NeumannCoefficients, FockSpace,
JMatrices, BPZ, PrimaryVertex, CompactBoson, Cache, Recursion.

CACHE_VERSION = v4_rho0_R_corner_plus1

## Session memos (reverse chronological)

- **[session_memo_rconv_fix.md](session_memo_rconv_fix.md)** (2026-04-15)
  Fixed ρ₀^T: corners at |ξ_T| = ±1, R_conv = 1. Vertex validated at small ℓ
  (d/(πℓ) → 1). Moderate ℓ still marginal. Design review: no formula errors.

- **[session_memo_rho0_and_bopen.md](session_memo_rho0_and_bopen.md)** (2026-04)
  BPZ sign fix (U(1) convention). ρ₀ corner-matching convention (superseded by
  R_conv fix above). |B^open⟩ contraction — initial investigation found divergence.

## Design documents

- [conformal_map_cross.md](../design/conformal_map_cross.md) — SC map, local coordinates, Neumann coefficients
- [plaquette_amplitude.md](../design/plaquette_amplitude.md) — Ward identity, primary vertex, tensor structure
- [local_coordinates_and_strip.md](local_coordinates_and_strip.md) — ρ₀ conventions, Z₂ symmetry, ASCII diagrams
- [modified_vertex.md](modified_vertex.md) — propagator factor, analogy with OSFT e^{K/2}
- [open_boundary_state.md](open_boundary_state.md) — |B^open⟩ definition, squeezed vacuum, exact overlap
- [decisions.md](decisions.md) — design decision log

## Open questions

1. **R_conv = 1 marginal at moderate ℓ** — undamped |B^open⟩ contraction doesn't
   converge for ℓ ≥ 0.5. Options: work at small ℓ, push R_conv > 1, or resum.
2. **Slight overshoot at very small ℓ** — d/(πℓ) > 1.0 at ℓ < 0.005. Minor.
3. **Resolved**: ρ₀^R shifted by +i so ξ_R(p) = +1, α_R = α_L. Propagator sign fixed.
   Z₂ now manifests as N^{LL}_{mk} = (-1)^{m+k} N^{RR}_{mk}.
