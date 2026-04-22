# Session Memo: Phase 1 Diagnostic (April 2026)

**Context.** This memo summarises what we learned during the Phase 1
diagnostic of strategy B
([`truncation_strategies.md`](../design/truncation_strategies.md),
[`finite_entanglement_scaling.md`](../design/finite_entanglement_scaling.md)):
treat the raw T-vertex $V_\ell$ as a uniform-MPS tensor and measure
entanglement entropy as a diagnostic of whether it reproduces CFT
boundary-state physics.

**Short answer.** Raw $V_\ell$ tiled as a uMPS is *qualitatively*
CFT-shaped (finite-$N$ EE decreases with $N$, grows with $L$, saturates
at $L \approx N/2$) but *quantitatively* suppressed by a large,
ℓ-dependent prefactor. At ℓ = 0.01 the extracted $c$ is ~$10^{-4}$;
at ℓ = 0.1 it is ~$10^{-2}$; target is $c = 1$. Extrapolating,
$S \sim \ell^{\sim 2\text{–}3}$: raw $V_\ell$ behaves as
identity + $O(\ell^2)$ perturbation, so the CFT content is entirely in
the subleading piece.

## What we did

### 1. Half-infinite canonical EE (notebook 09)

Built the MPS transfer matrix $E = \sum_T V_{T,L,R} V_{T,L',R'}$ with
$V_T$ as physical leg, computed left/right fixed points of $E$,
constructed the canonical-form $C$ matrix, computed
$S_\text{canon} = -\sum C_i^2 \log(C_i^2)$ per Vanderstraeten eq 21.

Results (ℓ ∈ {0.01, 0.05, 0.1, 0.3}, $h_\text{bond} \in \{4, 6, 8\}$):
$S_\text{canon}$ grows slowly with $h_\text{bond}$ (0.002–0.006 range
at ℓ = 0.1) but the correlation length $\xi$ from the subleading
eigenvalue of $E$ is essentially **flat in $h_\text{bond}$** —
spectrum of $E$ doesn't change as bond dimension grows beyond 4.

**Physical reading:** the $N \to \infty$ tiled $V_\ell$ is the open
boundary state $|B\rangle$ on a half-plane — a product state. So
$S_\text{canon} \to 0$ is the correct thermodynamic limit, not a bug.

### 2. Stab on V_T (per user request)

Inserted $e^{-x N_T}$ on the V_T leg to regularise. Expected: stab
"dresses" the vertex with extra propagation to reveal its CFT content.

**Sweep at ℓ=0.1, x ∈ {0, 0.1, 0.3, 0.5, 1, 2}:** $\xi$ barely moves
(3.24–3.31); $S_\text{canon}$ *decreases* monotonically (0.0036 →
0.0001). Stab makes things more trivial. Same picture at ℓ=0.01.

**Physical reading:** the stab projects V_T toward vacuum, reinforcing
the semi-infinite arm propagator already baked into raw $V_\ell$.
Moves the vertex toward identity.

### 3. GHZ sanity check

At the user's request, tested the EE pipeline on the GHZ-like uMPS
$A^1 = \operatorname{diag}(1,0)$, $A^2 = \operatorname{diag}(0,1)$,
expected $S = \log 2$ on every cut.

**Found a bug** in `canonical_C`: when the dominant eigenvalue of $E$
is degenerate (non-injective MPS), taking `argmax` of `eigen(E).values`
picks a single eigenvector, yielding rank-1 $l, r$ and $S = 0$
instead of $\log 2$.

**Fix:** collect all eigenvectors within tolerance of $\lambda_{max}$,
sum their Hermitian-positive-projected contributions. Applied to notebook
09 (commit `abc688e`). V_ℓ's transfer matrix is strictly non-degenerate,
so the bug didn't affect its $S_\text{canon}$ result.

### 4. Finite-$N$ $S(L)$ (notebook 10)

For PBC MPS on $N$ sites, the reduced density matrix on $L$
contiguous sites:
$$\rho(L)_{s, s'} = \frac{1}{Z_N}\,\operatorname{tr}\!\bigl[E^{N-L} \cdot (A^{s_1}\cdots A^{s_L}) \otimes (A^{s_1'}\cdots A^{s_L'})^*\bigr]$$

Implemented naive dense version, swept ℓ ∈ {0.01, 0.1},
$h_\text{phys} = 3$ (d_T=7), $h_\text{bond} = 6$ (D=30), $N \in
\{6, 8, 10, 12, 16\}$, $L \in \{1,\ldots,4\}$.

**Findings:**
- $S(L;N)$ **non-zero** at finite $N$: ~$10^{-3}$ at ℓ=0.1, ~$10^{-5}$ at ℓ=0.01.
- **Decreases with $N$** at fixed $L$ (approaching the $N\to\infty$ product-state limit).
- **Increases with $L$** then **saturates around $L \approx N/2$** (CFT-like PBC shape).
- **Slope** of $S(L)$ vs $\log[(N/\pi)\sin(\pi L/N)]$: ~0.004 at ℓ=0.1, ~$3\times 10^{-5}$ at ℓ=0.01.
  Target $c/3 \approx 0.333$. So off by factor ~100× at ℓ=0.1 and ~$10^4$× at ℓ=0.01.

### 5. Rank-$D^2$ bond-space algorithm

The L-site RDM has rank $\le D^2$ (two bond cuts in PBC), so its
spectrum is shared with a $D^2 \times D^2$ matrix
$K = W \cdot (M^T M) / Z_N$ where $W$ is a specific reshuffle of
$E^{N-L}$ and $M \in \mathbb{R}^{d_T^L \times D^2}$ is the
$L$-site-product tensor flattened. Built in `/tmp/phase1_SL_bondspace.jl`.

**Validation:** agrees with dense to $10^{-15}$ at $L \in \{1,2,3\}, N=8$.

**Performance:** $O(d_T^L \cdot D^4)$ for the build +
$O(D^6)$ for the eigensolve. At $h_\text{bond}=6$ ($D=30$), L=5,
d_T=7: runs in 0.9 s. Dense would need 30 min and 2.3 GB.

At ℓ=0.01, N=10, L=1..5: saturation at L=4→5 confirmed. Dominant
eigenvalue 1.000; second/third ~$10^{-6}$; sharp drop to $10^{-10}$
— essentially a pure vacuum projector with tiny admixture.

## Interpretation

Raw $V_\ell$ is close to the identity tensor:
$V_\ell = \mathbb{I} + \ell^2 \cdot (\text{CFT perturbation}) + \ldots$

Tiling the identity gives a pure product state. The CFT physics lives
entirely in the $O(\ell^2)$ piece, which is what sets the scale of any
entanglement. Hence:

- $S \sim \ell^{\alpha}$ with $\alpha \approx 2\text{–}3$.
- Both "shape" of S(L;N) and ℓ-scaling are consistent with this.
- To expose a coefficient of order $c = 1$, we'd either need
  (a) large enough ℓ for the perturbation to be $O(1)$, though
  R_conv convergence is marginal for ℓ ≳ 1, or
  (b) strip the arm propagators entirely — i.e. the modified vertex
  $\widetilde V_\ell = e^{(H_L + H_R)\ell/2} V_\ell$ — which was
  abandoned earlier for truncation-amplification reasons but may need
  revisiting.

## State of the code

- `experiments/notebooks/09_phase1_ee_correlation.jl` — half-infinite
  EE pipeline with fixed canonical_C (handles degenerate dominant
  eigenspace).
- `experiments/notebooks/10_finite_N_SL.jl` — naive dense finite-$N$
  $S(L)$ pipeline with CFT slope fit and plots.
- `/tmp/phase1_SL_bondspace.jl` (not yet integrated) — rank-$D^2$
  bond-space algorithm for L-site RDM spectrum. Worth promoting into
  notebook 10 or a separate module before the next deep dive.

## Status against strategy B

This is the Phase 1 diagnostic; it was meant to tell us whether raw
$V_\ell$ tiled as an MPS even *behaves* like a CFT. The answer is:
qualitative yes, quantitative no — prefactor collapses to zero as
$\ell \to 0$.

Does this kill strategy B? **No** — it only tells us that the naive
"raw V_ℓ tiled" is not the CFT-saturating construction. The strategy
B picture (CFT as non-simple-BC interface in the fixed TQFT
$A = \mathbb{C}[\mathbb{Z}_2]$) requires the multifusion machinery
and the off-diagonal $\mathcal{H}_{B_0 B_\pi}$ sector (see
`truncation_strategies.md` §5–§6), none of which is in this Phase 1
test. The test just confirms that the charge-0, single-BC vertex
alone isn't enough — consistent with the overall strategy B claim.

## What's next (options, deferred)

1. **Modified vertex Phase 1.** Re-run the finite-N $S(L)$ test with
   $\widetilde V_\ell$ in place of $V_\ell$. Expect the ℓ^2 prefactor
   to disappear; truncation-amplification issues may kick in.
2. **Multifusion upgrade.** Build the 8-block / non-simple-BC vertex
   described in `truncation_strategies.md` §6 Q1, then re-run Phase 1
   EE tests on each block.
3. **Phase 2.** Define $\perp$ (180°-flipped T) concretely, compose
   $T \circ \perp$ to get the cross / MPO, extract a Hamiltonian, check
   against XXZ or other known lattice models.
4. **Promote the bond-space algorithm** into a permanent utility if
   we'll do more finite-$N$ MPS work.
