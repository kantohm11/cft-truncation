# Project Instructions for Claude

## Physics conventions

### |0⟩ vs open Cardy state |B⟩⟩
- **|0⟩** = SL(2)-invariant vacuum = identity local operator. Inserting it at an arm does NOT remove the arm.
- **|B⟩⟩** = open Cardy state = caps the arm (arm disappears). This is what gives the strip propagator.
- Never confuse the two. "Capping an arm" = |B⟩⟩, NOT |0⟩.

### BPZ sign convention
For the U(1) current J (weight h=1), the BPZ map acts on modes as:
  `bpz(J_{-k}) = (-1)^{k+1} J_k`
So odd modes (k=1,3,5,...) get **no sign flip**, even modes (k=2,4,6,...) get **-1**.
The BPZ sign for a Fock state with partition λ = [k₁, k₂, ...] is `∏ (-1)^{kᵢ+1} = (-1)^{#even_parts}`.
This is NOT (-1)^N (that would be the Virasoro/weight-2 convention).

## Caching: bump CACHE_VERSION

When editing any of the following, bump `CACHE_VERSION` in `src/Cache.jl` and add a history entry:

- `_compute_vertex_raw`, `_recurse_entry`, `_apply_Jk_on_arm_sparse` (VertexRecursion.jl)
- `_compute_vertex_raw_cross`, `_recurse_entry_cross`, `_apply_Jk_on_arm_sparse_cross` (VertexRecursionCross.jl)
- `compute_geometry` (LocalCoordinates.jl), `compute_geometry_cross` (LocalCoordinatesCross.jl)
- `compute_neumann` (NeumannCoefficients.jl)
- `primary_vertex` (PrimaryVertex.jl)
- FockBasis state ordering or normalization (FockSpace.jl)
- `modified_vertex` propagator factor formula (VertexProjections.jl)

BPZ.jl is NOT in this list — BPZ is not used in the vertex recursion or caching.

## Pluto notebooks: verify before claiming success

Always run `experiments/scripts/check_notebook.jl <notebook>` before declaring a notebook finished. HTML export can succeed even when cells error.

## TensorKit notes

### @tensor with (0,N) TensorMaps
The output syntax `result[; -1 -2 -3]` (empty codomain before semicolon) does NOT parse in TensorOperations v5. Workaround: write `result[-1 -2 -3]` (puts free legs in codomain as V'), then `permute(result, ((), (1,2,3)))` to move to domain (double-dualization V' → V'' = V recovers original space).

### Charged contractions
TensorKit CAN handle charged TensorMaps. Use a 1D codomain at the appropriate charge sector: `Vect[U1Irrep](U1Irrep(-n) => 1)`. The `permute` from domain to codomain dualizes (V → V'), so selectors must act on V' with charge -n.

### DiagonalTensorMap
Exists and constructible via `DiagonalTensorMap(vec, V)`, but `⊗` does NOT preserve diagonal structure — materializes to dense. Use `@tensor` for leg-by-leg contraction with diagonal operators.
