# Finite-Dimensional Truncation of 2d CFT Preserving Locality

## 1. Philosophy

A fully extended 2d TQFT is specified (via the cobordism hypothesis) by a finite-dimensional algebra — concretely, a dagger-special-Frobenius algebra, equivalently a finite-dimensional 2-Hilbert space. A 2d CFT should analogously be determined by some "infinite" algebraic structure at this level, but the project deliberately avoids making that infinite notion precise.

Instead, the goal is to construct a **sequence of finite-dimensional 2-Hilbert spaces** (i.e., dagger-special-Frobenius algebras) that **approximate a given CFT** — paralleling how we approximate an infinite-dimensional Hilbert space by finite-dimensional ones in ordinary QM. Each truncation level gives an honest fully extended 2d TQFT.

This is related to Hamiltonian truncation but differs in a crucial respect: **locality is preserved** at every truncation level, because the truncation acts on local state spaces rather than on the global Hilbert space.

## 2. Construction

### 2.1 Input

A 2d CFT with known operator content and three-point functions (OPE coefficients). The framework is meant to be general, but concrete calculations require explicit three-point function data.

### 2.2 MPS/MPO from CFT Data

From the CFT's three- and four-point functions, construct MPS and MPO tensors. This is analogous to how Brehm–Runkel build interaction vertices from open channel factorisation on surfaces with holes. At this stage, the MPS/MPO tensors have:

- **Infinite physical (on-site) dimension** — indexed by the full set of states in the CFT.
- **Infinite bond dimension** — similarly indexed.

### 2.3 Level Truncation

Impose a cutoff at conformal weight $h_{\max}$: discard all states with $h > h_{\max}$. This renders both bond and physical dimensions finite, producing finite MPS/MPO tensors. The truncation acts locally (on each tensor individually), so the resulting model remains composed of local pieces.

### 2.4 The Resulting Structure

At each truncation level $h_{\max}$, the finite MPS/MPO data should assemble into a **dagger-special-Frobenius algebra**, specifying a fully extended 2d TQFT. As $h_{\max} \to \infty$, this sequence of TQFTs should converge (in a sense to be made precise) to the original CFT.

## 3. Core Question

> **Does the level-truncated data at each $h_{\max}$ form a dagger-special-Frobenius algebra?**

Establishing this is the central conceptual challenge of the project. In particular:

- The multiplication (from three-point functions) must remain associative after truncation.
- The Frobenius condition (compatibility of multiplication and comultiplication) must hold.
- The dagger-special conditions must be satisfied.

This is nontrivial because level truncation generically breaks algebraic relations. The question is whether the specific structure of CFT correlation functions ensures these relations survive.

## 4. Conceptual Background

### 4.1 Open String Field Theory

In open SFT (e.g., Witten's cubic theory), the star product gives an associative algebra structure on the space of open string states. This provides conceptual precedent: CFT data naturally organises into an algebraic structure, and level truncation of this algebra (widely used in numerical SFT) is a key source of intuition.

### 4.2 Brehm–Runkel: Lattice Models from CFT on Surfaces with Holes

**Key references:**

- [arXiv:2112.01563](https://arxiv.org/abs/2112.01563) — *Lattice models from CFT on surfaces with holes I* (Brehm, Runkel, 2021)
- [arXiv:2410.19938](https://arxiv.org/abs/2410.19938) — *Lattice models from CFT on surfaces with holes II* (Brehm, Runkel, 2024)

**Summary of their construction:**

- Start with a 2d RCFT on a surface with a regular lattice of holes, each equipped with a conformal boundary condition.
- Cut into triangles with clipped edges using open channel factorisation.
- The hole radius $R$ is a continuous parameter: $R \to 0$ recovers the CFT exactly; touching holes give a TQFT.
- The **cloaking boundary condition** ensures the full fusion category of topological line defects is realised exactly, even after truncation to finite state spaces.
- They obtain a two-parameter family of lattice models (hole radius $R$ and energy cutoff $h_{\max}$) that preserves topological symmetry at every truncation level.

**Relation to the present project:** The current project is a Hamiltonian-picture counterpart of the Brehm–Runkel Euclidean/partition-function construction. The MPS/MPO tensors play the role of Brehm–Runkel's interaction vertices, and level truncation plays the role of their energy cutoff. However, the present project builds its own regularisation rather than directly using Brehm–Runkel.

### 4.3 MPS/MPO

Matrix product states and matrix product operators provide the tensor network language for the construction. They are both intermediate derivables (objects to be computed from CFT data) and computational tools (for numerical evaluation of entanglement entropy, spectra, etc.).

## 5. Locality

Locality is preserved in two related senses:

1. **Extended field theory sense:** The truncated theory assigns data to points, intervals, and surfaces in a compatible way — the value on small local pieces determines the whole. This is the 2d extended TQFT structure specified by the Frobenius algebra.

2. **Lattice Hamiltonian sense:** The resulting lattice model should have local (e.g., nearest-neighbour) interactions, rather than the non-local truncation that arises in Hamiltonian truncation of the global Hilbert space.

The key mechanism is that truncation acts on **local state spaces** (bond/on-site dimensions of individual tensors), not on the global Hilbert space.

## 6. Examples and Diagnostics

### 6.1 Target Examples

- **Ising CFT / Virasoro minimal models** — matches the examples studied by Brehm–Runkel, providing benchmarks.
- **Free boson** — a non-rational example with continuous spectrum; tests the framework beyond the rational case.

### 6.2 Convergence Diagnostics

The precise notion of convergence (of the sequence of TQFTs to the CFT) is part of what the project aims to establish. Testable diagnostics include:

- **Entanglement entropy** — compare the EE of truncated states/operators against known CFT results (e.g., the $c/3 \log L$ scaling).
- **Energy spectrum** — compare the spectrum of the truncated lattice Hamiltonian against the CFT spectrum (conformal weights, degeneracies, tower structure).

## 7. Open Questions

- Under what conditions on the CFT data does level truncation produce a dagger-special-Frobenius algebra?
- Is there a natural notion of "distance" between the truncated TQFT and the target CFT?
- How does the truncation parameter $h_{\max}$ relate to the hole radius $R$ in Brehm–Runkel?
- Can the framework handle non-rational or non-unitary CFTs?
- What is the relationship between the Frobenius algebra at finite $h_{\max}$ and the fusion category symmetry preserved in Brehm–Runkel?
