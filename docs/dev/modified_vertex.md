# The Modified Vertex and the Boundary Operator Insertion

## The key object: $e^{(H_L + H_R)\ell/2} \cdot V_\ell$

The raw T-vertex $V_\ell(\alpha_T, \alpha_L, \alpha_R)$ as computed by the Ward identity recursion includes the propagation of states along the semi-infinite horizontal arms. The more physically fundamental object is the vertex **modified** by stripping off the arm propagators:

$$\widetilde{V}_\ell \;=\; e^{(H_L + H_R)\,\ell/2}\;\cdot\; V_\ell,$$

where $H_{L,R} = \pi(L_0 - c/24)$ is the open-string Hamiltonian on the arm of unit width. This is the central object for the lattice construction.

## Motivation: replacing a state by a boundary operator insertion

### The T-shape as a rectangle with three inputs

The T-shaped domain has three semi-infinite arms ($A_L$, $A_R$, $A_T$). Since the arms are propagators from infinity, the T-shape can be regarded as a **rectangle with three input states** — the "propagation from infinity" is just the limit of a long but finite strip.

### A non-standard state/operator correspondence

We want to replace the state $\psi_L$ (prepared at the infinity of arm $A_L$) by an **operator insertion** on the boundary. This is **not** the standard state/operator correspondence, which relates:

- A state on the semicircular arc of the upper semidisk $D^+$, and
- An operator at the origin of $D^+$.

Instead, the replacement we seek is:

> A state $\psi$ on one edge of a rectangle $\leftrightarrow$ an operator $X_\psi$ inserted at the midpoint of the opposite edge, with the BCFT boundary condition $|B\rangle$ imposed on the other three edges.

### Regularisation via finite-width rectangles

The operator $X_\psi$ is singular and requires regularisation. Consider an $\ell \times a$ rectangle:

- One width-$\ell$ edge carries the state $\psi$.
- The other three edges (two of length $a$, one of length $\ell$) carry the BCFT boundary condition $|B\rangle$.

A conformal map from the unit semidisk $D^+$ into this thin rectangle — sending the origin to the **midpoint of the $\ell$-edge opposite to $\psi$** — defines a regularised operator $X^{(a)}_\psi$.

The limit

$$X_\psi \;=\; \lim_{a \to 0}\; X^{(a)}_\psi$$

defines the boundary operator corresponding to the state $\psi$ in the strip geometry. This is analogous to the $\arctan$ conformal map to the **sliver frame** in open string field theory (OSFT).

### Analogy with $e^{K/2}$ in OSFT

In Witten's cubic OSFT, the star product involves three open-string states glued at their midpoints. The **identity string field** is the sliver state, and important simplifications arise from the $KBc$ subalgebra, where $K$ is the strip Hamiltonian. The operation of sandwiching a vertex by $e^{K/2}$ effectively "shrinks" the interaction region from a finite strip to an infinitesimal one.

The operation $e^{(H_L + H_R)\,\ell/2} \cdot V_\ell$ is the direct analogue: it shrinks the T-vertex (a strip of width $\ell$ with the operator $X_\psi$ inserted on the vertical arm) to an **infinitesimal strip** with $X_\psi$ still inserted. The modified vertex $\widetilde{V}_\ell$ is thus the "interaction kernel" with the arm propagation factored out.

## Computational realisation

In the truncated Fock basis, $L_0$ is diagonal with eigenvalue $h_i + N_i$ (conformal weight + descendant level) on each state. So the modification is a **diagonal rescaling** of the vertex entries:

$$\widetilde{V}_\ell(\alpha_T, \alpha_L, \alpha_R) \;=\; e^{\pi\ell/2\,(h_L + N_L - c/24)}\; e^{\pi\ell/2\,(h_R + N_R - c/24)}\;\cdot\; V_\ell(\alpha_T, \alpha_L, \alpha_R),$$

where $h_i + N_i$ is the conformal dimension of the state $\alpha_i$ on arm $i$. This is trivial to compute given the raw vertex $V_\ell$.

## Why this matters for convergence

The raw vertex has entries that decay like $e^{-\pi h_T/\ell}$ for heavy states on the vertical arm (see plaquette_amplitude.md, Section 5.4). Multiplying by the propagator factor on the horizontal arms compensates for the similar decay there, revealing the **intrinsic coupling strength** after stripping off the "trivial" propagation contribution. The modified vertex $\widetilde{V}_\ell$ should exhibit cleaner convergence properties as a function of the truncation level $h_{\max}$, and is the natural object to study when assessing whether the truncation produces a valid Frobenius algebra.
