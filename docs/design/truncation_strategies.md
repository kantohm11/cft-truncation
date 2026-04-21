# Truncation Strategies for CFT → TQFT

**Status (as of 2026-04-21).** Strategy **B** is the current working direction.
Strategy **A** is believed plausible but deferred (level truncation doesn't
seem to give a practical realization). The earlier framing — treating the
raw or modified T-vertex as an approximate *multiplication* map — is
abandoned.

Readers who have not seen the project before: start with
[`project_overview.md`](./project_overview.md) for the motivation
(approximating a 2d CFT by a sequence of honest 2d TQFTs / special
Frobenius algebras, with locality preserved). This document is about
*which* algebraic role the computed T-vertex is supposed to play.

## 1. The question

We have a concrete numerical object: the truncated T-vertex
$V_\ell : V_L \otimes V_T \to V_R$ (and its modified form
$\widetilde V_\ell = e^{(H_L + H_R)\ell/2}\, V_\ell$; see
[`modified_vertex.md`](../dev/modified_vertex.md)). The project's goal
is to read off a Frobenius-algebra / TQFT structure from this data.

The framings below differ in *what algebraic morphism* the vertex is
supposed to represent, and *where the Frobenius algebra lives*.

## 2. Prior framing (abandoned): T as approximate module multiplication

**The idea.** Let $A$ be a special Frobenius algebra with multiplication
$\mu : A \otimes A \to A$. Embed each arm space $V_i$ into $A$ via a
cutoff map $X_i : A \to A$ with $\operatorname{Im}(X_i) \cong V_i$ and
factoring through a "small" $A$-$A$-bimodule $B_i$. Then

$$V_\ell \;\approx\; X_R \circ \mu \circ (X_L \otimes X_T)$$

restricted to the arm subspaces. "Close to a TQFT" = "$B_i$ has few
simple components". $A$ would be constructed numerically from the vertex
data.

**Why abandoned.** A multiplication $\mu$ has a directional structure
(two inputs and one output, all in the same algebra), but the three arms
of the T-shape play *different* geometric roles in the CFT path
integral. Forcing all three onto the same algebra via $X_i$ papers over
this asymmetry.

The concrete failure mode is the modified vertex $\widetilde V_\ell$:
the propagator-stripping factor $e^{\pi(h+N)\ell/2}$ *grows* with level,
cancelling exactly the decay that keeps level truncation honest. The
raw vertex's weak-convergence results for the up/down commutator (see
[`session_memo_updown_commutator.md`](../dev/session_memo_updown_commutator.md))
showed the vertex data is fine — it was the "multiplication"
reinterpretation that was being forced.

## 3. Strategy A (plausible, deferred): T as bimodule map, A emergent

**The idea.** $V_L, V_R$ are modules over a Frobenius algebra $A$; $V_T$
is an $A$-$A$-bimodule. The T-vertex is a bimodule map

$$T \;:\; V_L \otimes_{\mathbb{C}} V_R' \;\longrightarrow\; V_T,$$

where $V_R'$ is the BPZ-conjugate / dual of $V_R$ so that
$V_L \otimes V_R'$ carries a natural $A$-$A$-bimodule structure. $A$
itself is emergent — extracted from the vertex data, e.g. as the
commutant of $T T^\dagger$ (Schur-like).

**TQFT dictionary for reference:**
- $A$ — a 2d open TQFT.
- A module over $A$ — a boundary condition.
- An $A$-$A$-bimodule — an interface / defect in that TQFT.

**Why deferred.** For the commutant / bimodule structure extracted this
way to be meaningful, the level truncation must be compatible with the
bimodule structure — and level truncation by conformal weight is
generically *not* compatible (the truncation boundary creates spurious
interfaces and garbles the commutant). Realising A probably needs a
different, bimodule-compatible truncation scheme — an idea we don't
have yet. A single-step algorithmic path from truncated vertex data to
emergent $A$ is not visible.

## 4. Strategy B (current): TQFT given, CFT as interface

**The idea.** Fix a TQFT: a special Frobenius algebra $A$, *externally
specified* (not derived from the vertex). Consider a **non-simple**
boundary condition of $A$ — a module $M$ that is a direct sum of simples
$M = \bigoplus_i M_i$. The direct-sum structure carries information
beyond the bare TQFT, and we regard this non-simple BC as a
**CFT ↔ TQFT interface**. In this picture, the 2d CFT strip is an
interface / defect inside the ambient TQFT.

**What varies with truncation.** The TQFT algebra $A$ is fixed. Level
truncation acts on the *interface data* (the CFT-side module / bimodule
on top of $A$), and the question is how this interface data converges
as $h_{\max} \to \infty$. "Hyperfinite-style" convergence (no strict
inclusions, just approximation of the limit) is the working picture.

**Trade-off vs A.** Less ambitious — $A$ is supplied, not emergent —
but much more tractable. Truncation errors only need to be controlled
within the fixed module category of $A$, instead of preserving an
emergent algebra structure.

**SFT parallel (non-load-bearing but clarifying).** This mirrors the
open SFT question of whether single-brane SFT contains multi-brane SFT
via non-trivial classical solutions. Non-simple $M$ here plays the role
of the multi-brane configuration realised *within* the single
underlying theory $A$. The two strategies A and B aren't mutually
exclusive; if both hold, the relation is intricate — as with the SFT
debate.

## 5. Concrete realisation for the compact boson

Strategy B with $A$ = the classical two-vacuum TQFT:

$$A \;=\; \mathbb{C}^2 \;=\; \mathbb{C}\,e_0 \oplus \mathbb{C}\,e_\pi,\qquad e_i \cdot e_j = \delta_{ij}\, e_i.$$

This $A$ has two simple modules $M_0, M_\pi$, and the non-simple BC is
the direct sum $M = M_0 \oplus M_\pi$. The TQFT is *a part of the
choice* — fixed throughout the truncation sequence (we defer the
variant where $A$ itself grows with the sequence).

**Identifying the simples for the existing code.**
- $M_0$ = Neumann + zero Wilson line (what the code currently computes).
- $M_\pi$ = Neumann + Wilson line $\pi$.

These are T-dual to Dirichlet with $\phi|_\partial = 0$ and
$\phi|_\partial = \pi$ respectively — more intuitive on the Dirichlet
side, but the code is Neumann.

**$A$ as a group algebra.** Explicitly $A = \mathbb{C}[\mathbb{Z}_2]$:
basis $\{1, g\}$ with $g^2 = 1$, so $A$ is commutative. Its primitive
idempotents are
$$p_\pm \;=\; \tfrac{1 \pm g}{2}, \qquad p_+ \leftrightarrow e_0, \quad p_- \leftrightarrow e_\pi.$$
Pushed from the TQFT bulk onto the CFT/TQFT interface, $p_\pm$ project
the non-simple CFT boundary $M$ onto the corresponding simple factor
from the CFT's point of view. The nontrivial group element $g$ is the
Wilson-line generator; inserted on the interface it acts as the
$\mathbb{Z}_2$-charge eigenvalue ($+1$ on $M_0$, $-1$ on $M_\pi$).

**Geometric picture (settled).** Option (a): BCFT on a 2d region with
the non-simple BC $M$ on its 1-dim boundary. $A$ acts only at the
boundary via the module structure — it's not ambient.

**Physical motivation — RG flow.** $A = \mathbb{C}^2$ is the natural
IR of the compact boson under a $\cos 2\tilde\phi$ deformation:
$\cos 2\tilde\phi$ breaks $U(1)_{\text{winding}} \to \mathbb{Z}_2$
(the residual $\tilde\phi \to \tilde\phi + \pi$); in the strong-coupling
IR, $\tilde\phi$ pins to one of two values, and the $\mathbb{Z}_2$ is
spontaneously broken. The IR phase is exactly the two-vacuum TQFT $A$.

Strategy B with this $A$ is therefore well-suited for
(i) studying this flow and (ii) producing a lattice / tensor-network
regularisation where $\mathbb{Z}_2$-SSB is natural — a qubit spin
chain (each $\mathbb{C}^2$ factor of $A$ = one qubit).

**What this concretely requires on top of the existing code.** The
CFT-side truncation is the same level cutoff we already do. What's new:
- A second copy of the relevant vertex / boundary-state data for
  $M_\pi$ (possibly derivable from the $M_0$ data by $\mathbb{Z}_2$).
- The module-category structure linking $M_0$ and $M_\pi$: morphisms,
  Wilson-line operators acting as matrix elements between them, etc.

## 6. Open questions (strategy B)

Settled (2026-04-21):
- **What is $A$?** $\mathbb{C}[\mathbb{Z}_2] \cong \mathbb{C}^2$, the
  $\mathbb{Z}_2$-SSB two-vacuum TQFT (see §5).
- **What is $M$?** $M_0 \oplus M_\pi$ = Neumann (0 Wilson) ⊕ Neumann
  (π Wilson).
- **Geometric picture.** BCFT on a 2d region with $M$ on the 1-dim
  boundary; $A$ acts at the boundary, not ambient.
- **Does $A$ grow with truncation?** Not for now — TQFT fixed through
  the sequence. A variant where $A$ grows is a later consideration.

Still open:

1. **Translation of $V_\ell$ into strategy B.** The existing T-vertex
   is computed for the three-arm T-shape with a single simple BC
   ($M_0$, Neumann + zero Wilson) on the outer boundary. In strategy B,
   what does $V_\ell$ become? A $2\times 2\times 2$-block tensor
   labelled by which simple $M_i$ sits on each arm's outer boundary
   segment? Something else entirely?
2. **$\mathbb{Z}_2$-equivariant vertex for $M_\pi$.** Does the $M_\pi$
   vertex follow from $M_0$ by the $\mathbb{Z}_2$ symmetry (Wilson-line
   shift), or does it need a separate computation?
3. **Wilson-line / morphism data.** The hom-spaces
   $\operatorname{Hom}(M_0, M_\pi)$ and $\operatorname{Hom}(M_\pi, M_0)$
   carry the off-diagonal module-category data. How are these
   represented concretely — as operator insertions on the boundary, as
   additional Fock-space blocks, or as separate amplitudes?
4. **Norm for approximation.** What quantity measures convergence of
   the truncated CFT-side data to the (formal, infinite-dim) BCFT with
   non-simple BC? HS / operator / spectral / matrix-element?
5. **First diagnostic.** What computation — using the existing code
   plus a $\mathbb{Z}_2$-partner vertex — is the fastest check that
   strategy B is on the right track? Candidates: matching the BCFT
   partition function on an interval with $M$ at both ends; a Wilson-
   line matrix element between the two vacua; the spectrum in the
   $M$-sector.

## 7. Relation to the existing concrete objects

Nothing in the existing core code (`compute_vertex`, `modified_vertex`,
`|B^{\text{open}}\rangle`, the Fock bases, the Ward recursion) changes
under this reinterpretation; what changes is the role each object
plays. Under strategy B:

- $V_\ell$ is a morphism in the TQFT+interface category — specifically
  which one is open question **1** above.
- $|B^{\text{open}}\rangle$ is a vector in the interface's state space
  — a natural "boundary of the interface" candidate.
- The level truncation is a finite-dim approximation of the CFT-side
  interface data, not of the TQFT.

## 8. Where this leaves the project narrative

[`project_overview.md`](./project_overview.md) frames the project as
producing a sequence of honest finite-dim Frobenius algebras
approximating the CFT. Strategy B is a refinement of that: the
Frobenius algebra $A$ is fixed, and the approximation happens on an
interface built on top of $A$. Finite-dim, locality-preserving
approximation is still the theme; the locus of approximation has moved
from "the algebra itself" to "the interface decorating the algebra".
Strategy A, if it can be made to work, would realise the stronger
original statement. For now we work in B.
