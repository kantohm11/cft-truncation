# Project Instructions for Claude

## Physics conventions

### |0⟩ vs open Cardy state |B⟩⟩
- **|0⟩** = SL(2)-invariant vacuum = identity local operator. Inserting it at an arm does NOT remove the arm.
- **|B⟩⟩** = open Cardy state = caps the arm (arm disappears). This is what gives the strip propagator.
- Never confuse the two. "Capping an arm" = |B⟩⟩, NOT |0⟩.

## Caching: bump CACHE_VERSION

When editing any of the following, bump `CACHE_VERSION` in `src/Cache.jl` and add a history entry:

- `_compute_vertex_raw`, `_recurse_entry`, `_apply_Jk_on_arm_sparse` (Recursion.jl)
- `compute_geometry` (LocalCoordinates.jl)
- `compute_neumann` (NeumannCoefficients.jl)
- `primary_vertex` (PrimaryVertex.jl)
- BPZ conventions (BPZ.jl)
- FockBasis state ordering or normalization (FockSpace.jl)
- `modified_vertex` propagator factor formula

## Pluto notebooks: verify before claiming success

Always run `experiments/scripts/check_notebook.jl <notebook>` before declaring a notebook export finished. PlutoSliderServer writes HTML even when cells error — cell-level errors are invisible in the export success message. The `cell.errored` flag is on the `Cell` struct, NOT on `cell.output`.

## TensorKit notes

### @tensor with (0,N) TensorMaps
The output syntax `result[; -1 -2 -3]` (empty codomain before semicolon) does NOT parse in TensorOperations v5. Workaround: write `result[-1 -2 -3]` (puts free legs in codomain as V'), then `permute(result, ((), (1,2,3)))` to move to domain (double-dualization V' → V'' = V recovers original space).

### Charged contractions
TensorKit CAN handle charged TensorMaps. Use a 1D codomain at the appropriate charge sector: `Vect[U1Irrep](U1Irrep(-n) => 1)`. The `permute` from domain to codomain dualizes (V → V'), so selectors must act on V' with charge -n.

### DiagonalTensorMap
Exists and constructible via `DiagonalTensorMap(vec, V)`, but `⊗` does NOT preserve diagonal structure — materializes to dense. Use `@tensor` for leg-by-leg contraction with diagonal operators.
