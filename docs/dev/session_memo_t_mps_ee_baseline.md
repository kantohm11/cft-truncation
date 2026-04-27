# T-MPS EE Qualitative Baseline (no stab, all charges)

**Date.** 2026-04-26.

**Script.** `experiments/scripts/phase1_SL_baseline.jl`.

## Setup

Uniform T-MPS tiled in PBC chain of length N. Bond and physical
spaces include **all charge sectors** (option 2 — full vertex tensor
flattened over (n_T, n_L, n_R) with `n_T + n_L + n_R = 0`), not just
charge-0. The Jacobian-convention fix (commit `1713c48`) is therefore
in play here.

Parameters: `R = 1.0`, `h_bond = 4`, `h_phys = 2`, ℓ ∈ {0.1, 0.5, 1.0},
N ∈ {8, 10}, `Lmax = 4`. Effective dims: `D = 34`, `d_T = 10`.

## Results

| ℓ | `S` range | `c_est` (N=8) | `c_est` (N=10) |
|---|---|---|---|
| 0.1 | [5.4e−3, 1.4e−2] | 2.36e−2 | 1.96e−2 |
| 0.5 | [2.7e−2, 3.2e−2] | 1.83e−2 | 1.57e−2 |
| 1.0 | [4.6e−2, 4.9e−2] | 9.41e−3 | 8.04e−3 |

Sanity flags per ℓ:
- `S > 0` everywhere: ✓.
- `S` monotone in `L` at fixed N: ✓.
- `S` monotone in `N` at fixed L: ✗ — `S(L; N=10)` is sometimes
  *less than* `S(L; N=8)`, especially at ℓ=0.1.

Figures: `docs/design/figures/T_mps_ee_baseline_ell{0.1, 0.5, 1.0}.png`
(per-ℓ S-vs-L and CFT-axis panels) and `T_mps_ee_baseline_summary.png`
(S(L=4; N) vs N for each ℓ).

## Interpretation

1. **EE is non-trivial but small** — peak ≈ 0.05 at ℓ=1, vs the c=1
   CFT prediction `(1/3) log(N) ≈ 0.7` at N=10.
2. **Saturation in L kicks in immediately** at L=2–3 — the curve is
   essentially a step + plateau. No `(c/3) log L` regime visible.
3. **No growth with N** — `S(L=4; N)` is flat or slightly decreasing
   from N=8 to N=10. For a CFT GS we'd see `(c/3) log(N)` growth;
   here the truncated MPS is approaching a product state of vacua as
   the chain grows, which is the opposite signature.
4. **Including charged bond primaries helps materially**: at ℓ=1,
   `S` is ~3× larger than the charge-0-only result for the same
   geometry. So charged primaries DO carry meaningful EE despite
   `|B⟩⟩_open` being purely in u(1) vacuum module — they enter via
   the truncation's coupling between sectors.
5. **EE grows with ℓ** in this experiment, opposite to the charge-0-only
   trend. At larger ℓ the bond's `|α_L|` is larger, which now
   *suppresses* charged primaries less aggressively (since
   `(1/α_L)^Δ` only mildly decays for `|α_L| ~ 2`), and the charged
   sector contributes more.

## Verdict

The T-MPS gives non-trivial-but-small EE with the correct *qualitative*
shape near the origin, but doesn't realize CFT scaling. The observed
non-monotonicity in N and early L-saturation are both consistent with
the constraint analysis: the (clean regulator + near-GS β + N ≥ 4)
window is empty, so any CFT scaling we'd hope for is squashed.

This baseline is now in place. Next steps to discuss:
- Whether to add a stab (regulator on bond legs at small ℓ) to test
  whether that recovers some scaling.
- When to move to the cross MPO to extend β_eff.
