# The Open Boundary State

## Definition

Given a 2d BCFT with conformal boundary condition $a$, the **open boundary state** $|B_a^{\text{open}}\rangle$ is a formal state in the open-string Hilbert space $\mathcal{H}_{aa}$ (states on an interval $[0,\pi]$ with BC $a$ at both endpoints).

It is prepared by an infinitesimally thin strip: take the rectangle $[0,\pi] \times [0,\epsilon]$ with BC $a$ on all four sides, and read off the state at $\tau = \epsilon$. In the $\epsilon \to 0$ limit this gives a formal (infinite-norm) state in $\mathcal{H}_{aa}$.

This is the open-string analogue of the standard (closed-channel) Cardy boundary state $|B_a\rangle \in \mathcal{H}_{\text{closed}}$.

## Gluing condition (conjectured)

Since on the open string $\bar{T} = T$ (method of images), the BC at the bottom boundary may translate into the condition

$$(L_n - L_{-n})|B_a^{\text{open}}\rangle = 0$$

within a single Virasoro algebra acting on $\mathcal{H}_{aa}$. For the compact boson, the analogous U(1) condition would be:

$$(J_n - J_{-n})|B_a^{\text{open}}\rangle = 0 \qquad (n \geq 1)$$

for Neumann BC, or $(J_n + J_{-n})|B_a^{\text{open}}\rangle = 0$ for Dirichlet BC.

**This needs further investigation.** The sign may depend on conventions.

## Open-open duality

The key consistency condition comes from a rectangle $[0,\pi] \times [0,T]$ with BC $a$ on all four sides. This geometry can be quantized in two dual open channels:

### Channel 1 (quantize along $\tau$)

Hilbert space $\mathcal{H}_{aa}$ from left/right BCs:

$$Z = \langle B_a^{\text{open}} | e^{-T(L_0 - c/24)} | B_a^{\text{open}} \rangle$$

### Channel 2 (quantize along $\sigma$)

Hilbert space $\mathcal{H}'_{aa}$ from top/bottom BCs:

$$Z = \langle B_a^{\prime\text{open}} | e^{-\pi(L'_0 - c/24)} | B_a^{\prime\text{open}} \rangle$$

Both must agree, but with a **modular weight** from the conformal anomaly:

$$f_1(T) / f_2(T) = (T/\pi)^{c/4}$$

## Conformal anomaly prefactor

The $(T/\pi)^{c/4}$ factor is the **Weyl anomaly** arising from the change of quantization direction. Physically, it comes from the zero-mode normalization: the Gaussian path integral over the zero mode of each scalar field has a measure proportional to $\sqrt{L}$ where $L$ is the spatial extent. Going from spatial width $\pi$ (channel 1) to spatial width $T$ (channel 2) gives a factor $\sqrt{T/\pi}$ per scalar. For $c$ scalars raised to the $-1/2$ power (from $f = \eta^{-c/2}$), this gives $(T/\pi)^{c/4}$.

Equivalently, this follows from the Dedekind eta modular transformation $\eta(-1/\tau) = \sqrt{-i\tau}\,\eta(\tau)$ with $\tau = iT/\pi$.

## Relation to the T-vertex

The open boundary state enters the T-vertex construction when capping an arm: inserting $|B_a^{\text{open}}\rangle$ on the physical (T) arm of the vertex yields a two-leg object (the propagator + boundary interaction). The modified vertex $\widetilde{V}_\ell$ composed with $|B_a^{\text{open}}\rangle$ on one arm should give the boundary operator insertion $X_\psi$ discussed in [modified_vertex.md](modified_vertex.md).

## Closed form for the compact boson

### Momentum sector

For Neumann BC, the zero-mode integral over the strip projects onto **zero momentum only**: $|B^{\text{open}}\rangle$ lives purely in the $n = 0$ sector (Virasoro vacuum module).

### Squeezed vacuum formula

The gluing condition $(J_k - J_{-k})|B\rangle = 0$ for $k \geq 1$ determines a **squeezed vacuum**:

$$|B^{\text{open}}\rangle = \exp\!\left(\sum_{k=1}^{\infty} \frac{J_{-k}^2}{2k}\right)|0\rangle$$

This is the open-string analogue of the closed-string Ishibashi state $\exp(\sum \alpha_{-n}\tilde{\alpha}_{-n}/n)|0\rangle$, but pairing the mode with **itself** ($J_{-k}^2$ instead of $J_{-k}\tilde{J}_{-k}$).

### Coefficients in the normalised basis

The single-mode coefficient at even occupation $2m$ is:

$$b(2m) = \prod_{j=1}^{m} \sqrt{\frac{2j-1}{2j}} = \frac{\sqrt{(2m)!}}{2^m\,m!}, \qquad b(\text{odd}) = 0$$

The full coefficient for a partition $\lambda$ with multiplicities $m_k$:

$$b_\lambda = \prod_{k} b(m_k(\lambda))$$

Nonzero only when **all multiplicities are even** ("doubled partitions").

### Exact overlap (product formula)

Since $b(2m)^2 = \binom{2m}{m}/4^m$ and the generating function is $(1-x)^{-1/2}$, the boundary overlap factors over modes:

$$f(T) = \langle B^{\text{open}}|e^{-T(L_0 - c/24)}|B^{\text{open}}\rangle = \eta(e^{-2T})^{-c/2}$$

For $c = 1$: $f(T) = \eta(e^{-2T})^{-1/2}$. This converges exponentially fast as a product (no truncation needed).

## Novelty

A web search (April 2026) found no prior discussion of this object in the literature. The standard BCFT boundary state literature and OSFT literature (Kiermaier-Okawa-Zwiebach, Ellwood invariants, etc.) all concern the closed-string boundary state. The open boundary state as defined here appears to be a novel construction.
