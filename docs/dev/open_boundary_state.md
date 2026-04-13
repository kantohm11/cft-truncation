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

Both must agree — this is a modular-like (open-open duality) constraint purely within the open sector, with no closed strings involved.

## Relation to the T-vertex

The open boundary state enters the T-vertex construction when capping an arm: inserting $|B_a^{\text{open}}\rangle$ on the physical (T) arm of the vertex yields a two-leg object (the propagator + boundary interaction). The modified vertex $\widetilde{V}_\ell$ composed with $|B_a^{\text{open}}\rangle$ on one arm should give the boundary operator insertion $X_\psi$ discussed in [modified_vertex.md](modified_vertex.md).

## Computation for the compact boson

For the compact boson at radius $R$ with Neumann BC, the open boundary state in each momentum sector $n$ is determined by the gluing condition $(J_k - J_{-k})|B_n\rangle = 0$ for $k \geq 1$. In the normalised Fock basis $|\hat{\lambda}; n\rangle$, the coefficients $b_{n,\lambda} = \langle \hat{\lambda}; n | B_n^{\text{open}}\rangle$ must satisfy:

$$\sqrt{k \cdot m_k(\lambda)} \cdot b_{\lambda \setminus k} = \sqrt{k \cdot (m_k(\lambda) + 1)} \cdot b_{\lambda \cup k}$$

for each partition $\lambda$ and each $k \geq 1$, where $m_k(\lambda)$ is the multiplicity of part $k$ in $\lambda$.

This is a set of linear recursion relations that determine $b_\lambda$ in terms of $b_\varnothing$ (the primary coefficient). The implementation constructs the state level by level.

## Novelty

A web search (April 2026) found no prior discussion of this object in the literature. The standard BCFT boundary state literature and OSFT literature (Kiermaier-Okawa-Zwiebach, Ellwood invariants, etc.) all concern the closed-string boundary state. The open boundary state as defined here appears to be a novel construction.
