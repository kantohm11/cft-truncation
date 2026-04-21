# Finite-Entanglement Scaling via the T-Vertex

**Purpose.** This document specifies the **Phase 1 diagnostic** of the
strategy outlined in [`truncation_strategies.md`](./truncation_strategies.md):
use the existing T-vertex $V_\ell$ to extract an entanglement entropy
$S_D$ and correlation length $\xi_D$ as a function of the Fock-space
truncation $D$ (controlled by $h_{\max}$), and read off the central
charge from the Calabrese–Cardy / Tagliacozzo–Pollmann scaling
$S_D \sim (c/6) \log \xi_D$.

This is the fastest non-trivial test that the truncated vertex data
describes a $c = 1$ CFT. It does **not** require any of the
multifusion / non-simple-BC / $\perp$-composition machinery of Phase 2
— it uses only the existing single-BC code.

## 1. Picture

Treat $V_\ell$ as an MPS tensor with one "physical" leg ($V_T$) and two
"bond" legs ($V_L, V_R$). Concretely:

- Physical leg: the truncated Fock space on the T arm, $V_T$.
- Bond legs: the truncated Fock spaces on L and R arms, $V_L$ and $V_R$.

The uniform MPS built by tiling $V_\ell$ along a 1d lattice is the
"ground state" we'll characterise via its entanglement properties.
Bond dimension $D = \dim V_L = \dim V_R$, set by $h_{\max}$.

Because we are at criticality (the underlying CFT is massless), the
MPS correlation length is finite but grows with $D$. The
finite-entanglement scaling predicts

$$S_D \;\sim\; \tfrac{c}{6} \log \xi_D + \text{const},$$

where $S_D$ is the half-infinite-chain bipartite EE and $\xi_D$ is the
MPS correlation length. Plotting $S_D$ vs $\log \xi_D$ over several
values of $D$ extracts $c$ as the slope times 6.

## 2. Transfer matrix and correlation length

The MPS transfer matrix (Vanderstraeten–Haegeman–Verstraete
[arXiv:1810.07006](https://arxiv.org/abs/1810.07006), eq 38) is

$$E \;=\; \sum_s A^s \otimes \bar{A}^s,$$

a 4-leg tensor with two pairs of bond indices. In our setup, $A = V_\ell$
with physical leg = $V_T$, so

$$E \;=\; \text{contract } V_\ell \otimes V_\ell^\perp \text{ over } V_T,$$

which is the **I-shape** tensor of the project — see
[`truncation_strategies.md`](./truncation_strategies.md) §6. Here
$V_\ell^\perp$ plays the role of $\bar{A}$; identifying it is
[open question](./truncation_strategies.md) 2 of the "still open"
list. Candidates: complex conjugate of $V_\ell$ with a specific leg
swap; the 180°-rotated T-vertex; the BPZ image; or a combination.

Viewing $E$ as a matrix
$(V_L \otimes V_L) \to (V_R \otimes V_R)$, its spectrum gives:

- Leading eigenvalue $\lambda_0$: set to $1$ by normalising $V_\ell$
  (absorb the factor $\lambda_0^{-1/2}$ into each $V_\ell$).
- Subleading eigenvalue $\lambda_1$: defines the correlation length
  (Vanderstraeten eq 40):
$$\xi_D \;=\; -\frac{1}{\log |\lambda_1|}.$$

## 3. Entanglement entropy via the mixed canonical form

The Vanderstraeten recipe (§2.1, eqs 14–21):

1. Find the **left dominant eigenvector** $l$ of $E$ (Hermitian
   positive $D \times D$ matrix on the left-bond double space).
2. Find the **right dominant eigenvector** $r$ similarly.
3. Factor $l = L^\dagger L$ and $r = R R^\dagger$ (Cholesky or
   symmetric square root).
4. The **center matrix** on a bond is $C = L R$ (a pure bond-to-bond
   $D \times D$ matrix, no physical leg).
5. SVD: $C = U \operatorname{diag}(C_i) V^\dagger$. The singular
   values $C_i \ge 0$ are the **Schmidt values** of the MPS across any
   bond. (The residual unitaries $U, V^\dagger$ are absorbed into
   $A_L, A_R$ as gauge — we only need the $C_i$ for EE.)
6. Half-infinite bipartite EE (Vanderstraeten eq 21):
$$S_D \;=\; -\sum_i C_i^2 \log(C_i^2).$$

The reason this works: in left- and right-canonical gauges, the
half-chain states $|\Psi_L^i\rangle, |\Psi_R^i\rangle$ defined by
absorbing the left / right halves into the bond index are orthonormal
(the isometry conditions collapse the infinite chain to the identity
on the surviving bond index). Therefore
$|\Psi\rangle = \sum_i C_i |\Psi_L^i\rangle \otimes |\Psi_R^i\rangle$
is already in Schmidt form — $C_i$ are the Schmidt values by
construction.

## 4. Algorithm

Inputs: $V_\ell$ at a chosen $\ell$, for several truncation levels
$h_{\max} \in H$.

For each $h_{\max} \in H$:

1. Compute $V_\ell^{(h_{\max})}$ (existing `compute_vertex`).
2. Construct $V_\ell^\perp$ by the (to-be-fixed) symmetry map from
   $V_\ell$. Verify against a direct recomputation on a low-level block
   as a one-time sanity check.
3. Build $E = V_\ell \otimes V_\ell^\perp$ with $V_T$ contracted. Shape
   $D^2 \times D^2$ as a matrix.
4. Arnoldi-style eigendecomposition: obtain $\lambda_0, \lambda_1$ and
   the dominant left/right eigenvectors $l, r$. Rescale $V_\ell$ by
   $\lambda_0^{-1/2}$ (or equivalently divide the eigenvalues by
   $\lambda_0$).
5. $\xi_D = -1 / \log |\lambda_1|$.
6. Factor $l = L^\dagger L$, $r = R R^\dagger$ (Cholesky on the $D^2$-
   dim Hermitian positive matrices reshaped appropriately onto the
   bond space).
7. Build $C = L R$, SVD it, get $C_i$.
8. $S_D = -\sum C_i^2 \log C_i^2$.

Across the sweep: plot $S_D$ vs $\log \xi_D$; fit slope; compare to $c/6$
with target $c = 1$.

## 5. Parameters

- $\ell$: start with a single value in the good regime (propagator
  test convergent). Suggested $\ell = 0.1$; extend to $\{0.05, 0.1\}$
  later.
- $h_{\max}$: $\{4, 6, 8, 10\}$ on a laptop if compute allows. Need
  the sweep to span enough bond dimensions to see a slope.
- `series_order`: default (20).
- Rényi check: optionally also compute
  $S_n = \frac{1}{1-n} \log \sum C_i^{2n}$ for $n = 2, 3$; the ratio
  $S_2/S_1 \approx 3/4$, $S_3/S_1 \approx 2/3$ at $c = 1$ per
  Calabrese–Cardy.

## 6. Shortcut for sanity checking

Before running the full canonical-form computation, a cheaper first
pass: SVD the I-shape matrix directly (without projecting out the
fixed point or going through $l, r, C$). Use the singular values
$\sigma_i$ as pseudo-probabilities $p_i = \sigma_i^2 / \sum \sigma_j^2$
and compute $-\sum p_i \log p_i$. This is **not** the Schmidt EE, but
it should still grow logarithmically with $h_{\max}$ if the vertex
data is CFT-like. If the shortcut is ragged or saturating, debug
before chasing the full algorithm.

## 7. What success and failure look like

**Success:** $S_D$ grows linearly with $\log \xi_D$ across the sweep,
with slope close to $1/6$. Rényi ratios are within ~10 % of
Calabrese–Cardy. The correlation length $\xi_D$ grows smoothly with
$h_{\max}$ (power-law or faster).

**Failure modes:**
- $S_D$ saturates or oscillates: the truncation isn't seeing a CFT —
  either $\ell$ is wrong (too close to a degenerate limit) or $V_\ell$
  data has a bug.
- Correct slope, wrong intercept: fine, expected.
- Clean power law but slope $\neq 1/6$: either wrong identification of
  $V_\ell^\perp$ (so $E$ isn't the MPS transfer matrix), or a
  normalisation issue.
- Extremely small $\xi_D$ (order 1) even at large $h_{\max}$: the MPS
  is near-trivial; likely $\ell$ too small and we're close to the
  identity limit.

## 8. Relation to Phase 2

Phase 2 (composing $V_\ell \circ V_\ell^\perp$ over $V_L, V_R$ to form
the cross / MPO, then extracting a Hamiltonian — see
`plaquette_amplitude.md` §7.4) uses the **same** two ingredients
$V_\ell$ and $V_\ell^\perp$ but contracts along the *other* pair of
legs. Phase 1 validates the vertex as a well-behaved MPS tensor; Phase 2
uses it to build a lattice Hamiltonian. If Phase 1 doesn't show $c = 1$,
Phase 2's target of identifying a concrete Hamiltonian (XXZ or similar)
is premature.
