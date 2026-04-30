"""
Post-recursion analysis primitives for the T-shape vertex.

Split off from `Recursion.jl`. Contains the propagator factor
`build_propagator_factor`, the modified vertex
`modified_vertex` / `modified_vertex_cache`, the h-cut projector
`project_to_hcut`, and the contracted-norm helpers
`projected_norm_after_contract_T(L)` / `full_norm_after_contract_T(L)`.

Must be `include`d after `VertexRecursion.jl` (uses `VertexData` and
`conformal_dim`).
"""

# ============================================================
# TensorMap-native analysis primitives
#
# The recommended way to do analysis is to compose these primitives
# directly. For example, to compute the projected norm after contracting
# the T leg with a charge-n selector:
#
#   V_reshaped = permute(Vm, ((1,), (2, 3)))
#   sel = build_selector(bp, n, alpha)
#   contracted = sel * V_reshaped          # V_{-n} ← V_bond²
#   projected = project_to_hcut(contracted, [bb, bb], h_cut)
#   ratio = norm(projected) / norm(contracted)
#
# The specialized functions (projected_norm_after_contract_T, etc.) are
# kept for backward compat but the compositional form is preferred.
# ============================================================

"""
    build_propagator_factor(basis::FockBasis, ell, c) -> DiagonalTensorMap

Diagonal endomorphism V -> V with eigenvalue exp(pi*ell/2*(h_n+N-c/24))
on each basis state. This is e^{H*ell/2} where H = pi(L_0 - c/24) on
a strip of unit width. Returned as a DiagonalTensorMap for efficient
storage and composition.
"""
function build_propagator_factor(basis::FockBasis, ell::Float64, c::Float64)
    D = TensorKit.DiagonalTensorMap(undef, basis.V)
    for (f1, f2) in fusiontrees(D)
        n = Int(f2.uncoupled[1].charge)
        haskey(basis.levels, n) || continue
        blk = D[f1, f2]  # Diagonal matrix view
        d = blk.diag      # the underlying diagonal vector
        for a in 1:length(d)
            d[a] = exp(pi * ell / 2 * (conformal_dim(basis, n, a) - c / 24))
        end
    end
    D
end

"""
    modified_vertex(vd::VertexData) -> TensorMap

Compute the modified vertex Ṽ = V ∘ (id_phys ⊗ D ⊗ D) where
D = e^{H ℓ/2} is the propagator factor on each bond arm.
The central charge `c` is read from `vd.cft.c`.

Uses `@tensor` to contract D into the two bond legs of the vertex,
then `permute` to restore the original (0,3) codomain/domain structure.

## Note on the @tensor + permute pattern

The ideal one-liner `vertex * (id(V_phys) ⊗ D ⊗ D)` is correct but
impractical: TensorKit's `⊗` on TensorMaps materializes the full dense
tensor product (dim³ × dim³), even when D is diagonal. This is an
upstream limitation — `DiagonalTensorMap` exists, but `⊗` does not
preserve it.

Instead, `@tensor` contracts D into individual legs efficiently
(O(entries), no dense intermediate). However, @tensor has its own
quirk: for a (0,N) TensorMap (trivial codomain), the output syntax
`result[; -1 -2 -3]` (empty codomain, semicolon, domain indices)
**does not parse** in TensorOperations v5. The parser misinterprets
`-1` after `;` as subtraction rather than a negative index label.

The workaround: write `@tensor result[-1 -2 -3] := ...` (no semicolons),
which puts all free legs in the **codomain** (producing a (3,0) Tensor
with V' legs), then call `permute(result, ((), (1,2,3)))` to move them
to the domain. The double dualization V → V' (from @tensor) → V'' = V
(from permute) recovers the original space.
"""
function modified_vertex(vd::VertexData)
    V = vd.vertex
    D = build_propagator_factor(vd.cft.basis_bond, vd.ell, vd.cft.c)
    @tensor tmp[-1 -2 -3] := V[-1 1 2] * D[1; -2] * D[2; -3]
    permute(tmp, ((), (1, 2, 3)))
end

"""
    modified_vertex_cache(cft, ells; series_order=20, cache=:auto) -> Dict{Float64, TensorMap}

Precompute and cache modified vertices for a sweep over ell values.
Central charge c is read from cft.c.
"""
function modified_vertex_cache(cft::CompactBosonCFT, ells;
                               series_order::Int=20, cache::Symbol=:auto)
    Dict(Float64(l) => modified_vertex(compute_vertex(cft, l; series_order, cache))
         for l in ells)
end

"""
    project_to_hcut(tm::TensorMap, bases, h_cut) -> TensorMap

Zero out entries where any tensor factor has conformal dimension > h_cut.
`bases` is a tuple/vector of FockBasis, one per domain leg (in order).
"""
function project_to_hcut(tm::TensorMap, bases, h_cut::Float64)
    result = copy(tm)
    for (f1, f2) in fusiontrees(result)
        charges = [Int(f2.uncoupled[i].charge) for i in 1:length(f2.uncoupled)]
        blk = result[f1, f2]
        for idx in CartesianIndices(blk)
            keep = true
            for (leg, a) in enumerate(Tuple(idx))
                n = charges[leg]
                if !haskey(bases[leg].levels, n) || conformal_dim(bases[leg], n, a) > h_cut + 1e-10
                    keep = false; break
                end
            end
            keep || (blk[idx] = 0.0)
        end
        result[f1, f2] = blk
    end
    result
end

"""
    projected_norm_after_contract_T(Vm, basis_phys, basis_bond, vec_T, h_cut) -> Float64

Compute the projected norm (at h_cut) of the modified vertex Vm after
contracting the T (phys) leg with a linear combination of basis states.

`vec_T` is a vector of ((n_T, αT), coefficient) pairs.

Works directly on Vm's TensorMap blocks, avoiding intermediate TensorMap
construction (which would fail for charged contractions due to TensorKit's
charge-conservation constraint on the result space).
"""
function projected_norm_after_contract_T(Vm::TensorMap, basis_phys::FockBasis,
                                          basis_bond::FockBasis,
                                          vec_T::Vector, h_cut::Float64)
    # Accumulate: result[(n_L,n_R,αL,αR)] = Σ_i c_i * Vm[αT_i, αL, αR]
    contracted = Dict{NTuple{4,Int}, Float64}()
    for ((n_T, αT), coeff) in vec_T
        coeff == 0.0 && continue
        for (f1, f2) in fusiontrees(Vm)
            Int(f2.uncoupled[1].charge) == n_T || continue
            n_L = Int(f2.uncoupled[2].charge)
            n_R = Int(f2.uncoupled[3].charge)
            blk = Vm[f1, f2]
            for αL in axes(blk, 2), αR in axes(blk, 3)
                key = (n_L, n_R, αL, αR)
                contracted[key] = get(contracted, key, 0.0) + coeff * blk[αT, αL, αR]
            end
        end
    end
    # Compute projected norm
    sq = 0.0
    for ((n_L, n_R, αL, αR), val) in contracted
        conformal_dim(basis_bond, n_L, αL) > h_cut + 1e-10 && continue
        conformal_dim(basis_bond, n_R, αR) > h_cut + 1e-10 && continue
        sq += val^2
    end
    sqrt(sq)
end

"""
    projected_norm_after_contract_TL(Vm, basis_phys, basis_bond, vec_T, vec_L, h_cut) -> Float64

Same as above but contracting both the T (phys) and L (first bond) legs.
`vec_T` and `vec_L` are vectors of ((n, α), coefficient) pairs.
"""
function projected_norm_after_contract_TL(Vm::TensorMap, basis_phys::FockBasis,
                                           basis_bond::FockBasis,
                                           vec_T::Vector, vec_L::Vector, h_cut::Float64)
    contracted = Dict{NTuple{2,Int}, Float64}()
    for ((n_T, αT), cT) in vec_T
        cT == 0.0 && continue
        for ((n_L, αL), cL) in vec_L
            cL == 0.0 && continue
            for (f1, f2) in fusiontrees(Vm)
                Int(f2.uncoupled[1].charge) == n_T || continue
                Int(f2.uncoupled[2].charge) == n_L || continue
                n_R = Int(f2.uncoupled[3].charge)
                blk = Vm[f1, f2]
                for αR in axes(blk, 3)
                    key = (n_R, αR)
                    contracted[key] = get(contracted, key, 0.0) + cT * cL * blk[αT, αL, αR]
                end
            end
        end
    end
    sq = 0.0
    for ((n_R, αR), val) in contracted
        conformal_dim(basis_bond, n_R, αR) > h_cut + 1e-10 && continue
        sq += val^2
    end
    sqrt(sq)
end

"""
    full_norm_after_contract_T(Vm, vec_T) -> Float64

Full (unprojected) norm after contracting T leg.
"""
function full_norm_after_contract_T(Vm::TensorMap, vec_T::Vector)
    contracted = Dict{NTuple{4,Int}, Float64}()
    for ((n_T, αT), coeff) in vec_T
        coeff == 0.0 && continue
        for (f1, f2) in fusiontrees(Vm)
            Int(f2.uncoupled[1].charge) == n_T || continue
            n_L = Int(f2.uncoupled[2].charge)
            n_R = Int(f2.uncoupled[3].charge)
            blk = Vm[f1, f2]
            for αL in axes(blk, 2), αR in axes(blk, 3)
                key = (n_L, n_R, αL, αR)
                contracted[key] = get(contracted, key, 0.0) + coeff * blk[αT, αL, αR]
            end
        end
    end
    sqrt(sum(v^2 for (_, v) in contracted; init=0.0))
end

"""
    full_norm_after_contract_TL(Vm, vec_T, vec_L) -> Float64

Full (unprojected) norm after contracting both T and L legs.
"""
function full_norm_after_contract_TL(Vm::TensorMap, vec_T::Vector, vec_L::Vector)
    contracted = Dict{NTuple{2,Int}, Float64}()
    for ((n_T, αT), cT) in vec_T
        cT == 0.0 && continue
        for ((n_L, αL), cL) in vec_L
            cL == 0.0 && continue
            for (f1, f2) in fusiontrees(Vm)
                Int(f2.uncoupled[1].charge) == n_T || continue
                Int(f2.uncoupled[2].charge) == n_L || continue
                n_R = Int(f2.uncoupled[3].charge)
                blk = Vm[f1, f2]
                for αR in axes(blk, 3)
                    key = (n_R, αR)
                    contracted[key] = get(contracted, key, 0.0) + cT * cL * blk[αT, αL, αR]
                end
            end
        end
    end
    sqrt(sum(v^2 for (_, v) in contracted; init=0.0))
end
