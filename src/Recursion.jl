"""
Build the T-vertex tensor via Ward identity recursion.

The primary entry point is the layered form
    compute_vertex(cft::CompactBosonCFT, ell; series_order=20)
which reuses the ℓ-independent CFT data (basis, J matrices, BPZ form)
and only does the per-ℓ work (geometry + Neumann + Ward recursion).

A backward-compatible kwarg form
    compute_vertex(; R, ell, trunc, ...)
builds the CFT data on the fly and is preserved so existing tests don't churn.
"""

struct VertexData
    cft::CompactBosonCFT
    vertex::TensorMap       # ℂ ← V_phys ⊗ V_bond ⊗ V_bond
    geom::Geometry
    neumann::NeumannData
    ell::Float64
end

# Backward-compat field forwarding so existing tests that read
# vd.basis_bond / vd.basis_phys / vd.R continue to work.
function Base.getproperty(vd::VertexData, name::Symbol)
    if name === :basis_bond
        return getfield(vd, :cft).basis_bond
    elseif name === :basis_phys
        return getfield(vd, :cft).basis_phys
    elseif name === :R
        return getfield(vd, :cft).R
    else
        return getfield(vd, name)
    end
end

"""
    compute_vertex(cft::CompactBosonCFT, ell::Real; series_order=20) -> VertexData

Compute the T-vertex for the given fixed CFT data at the given ℓ. This is
the primary entry point for ℓ-sweeps: build `cft` once with
`CompactBosonCFT(; R, trunc=...)`, then call this for each ℓ.
"""
function compute_vertex(cft::CompactBosonCFT, ell::Real; series_order::Int=20)
    geom = compute_geometry(ell, series_order)
    neumann = compute_neumann(geom, cft.m_max)
    _build_vertex(cft, geom, neumann, Float64(ell))
end

"""
    compute_vertex(cft::CompactBosonCFT, geom::Geometry, neumann::NeumannData;
                   ell::Real) -> VertexData

Lower-level: assemble the vertex using pre-computed geometry and Neumann data.
Useful when the same geometry is shared across CFTs, or for testing.
"""
function compute_vertex(cft::CompactBosonCFT, geom::Geometry, neumann::NeumannData;
                        ell::Real)
    _build_vertex(cft, geom, neumann, Float64(ell))
end

"""
    compute_vertex(; R, ell, trunc, geom=nothing, neumann=nothing, series_order=20)

Backward-compatible one-shot form that builds a fresh `CompactBosonCFT` per call.
Prefer the layered form for ℓ-sweeps.
"""
function compute_vertex(; R::Real, ell::Real, trunc::TruncationSpec,
                        geom=nothing, neumann=nothing, series_order::Int=20)
    cft = CompactBosonCFT(R=R, trunc=trunc)
    if geom !== nothing
        neum = neumann === nothing ? compute_neumann(geom, cft.m_max) : neumann
        return compute_vertex(cft, geom, neum; ell=ell)
    else
        return compute_vertex(cft, ell; series_order=series_order)
    end
end

"""
    vertex_sweep(cft::CompactBosonCFT, ells; series_order=20) -> Vector{VertexData}

Convenience: compute vertices for all ℓ values, reusing the CFT data.
"""
function vertex_sweep(cft::CompactBosonCFT, ells; series_order::Int=20)
    [compute_vertex(cft, ell; series_order=series_order) for ell in ells]
end

# Internal: actually run the recursion and assembly given all three layers.
function _build_vertex(cft::CompactBosonCFT, geom::Geometry, neumann::NeumannData,
                       ell::Float64)
    raw = _compute_vertex_raw(cft.basis_bond, cft.basis_phys, geom, neumann,
                              cft.J_bond, cft.J_phys, cft.R)
    vertex = _assemble_vertex(raw, cft.basis_bond, cft.basis_phys)
    VertexData(cft, vertex, geom, neumann, ell)
end

"""
Compute all vertex entries by sweeping total level.
Returns a dict keyed by (n_T, n_L, n_R, α_T, α_L, α_R) → value.
Here α_i is the 1-based index of the state within sector n_i.
"""
function _compute_vertex_raw(basis_bond::FockBasis, basis_phys::FockBasis,
                             geom::Geometry, neumann::NeumannData,
                             J_bond, J_phys, R::Float64)
    raw = Dict{NTuple{6, Int}, Float64}()

    # Iterate over all valid (n_L, n_R, n_T) triples (with n_L + n_R + n_T = 0)
    # and all state indices, sweeping by total level
    sectors_bond = collect(keys(basis_bond.states))
    sectors_phys = collect(keys(basis_phys.states))

    # Collect all valid triples
    triples = NTuple{6, Int}[]
    for n_L in sectors_bond, n_R in sectors_bond
        n_T = -(n_L + n_R)
        n_T in sectors_phys || continue
        for αL in eachindex(basis_bond.states[n_L])
            for αR in eachindex(basis_bond.states[n_R])
                for αT in eachindex(basis_phys.states[n_T])
                    push!(triples, (n_T, n_L, n_R, αT, αL, αR))
                end
            end
        end
    end

    # Sort by total level (N_L + N_R + N_T)
    total_level(t) = basis_bond.levels[t[2]][t[5]] +
                     basis_bond.levels[t[3]][t[6]] +
                     basis_phys.levels[t[1]][t[4]]
    sort!(triples, by=total_level)

    # Primary vertex at level 0
    for t in triples
        N_tot = total_level(t)
        N_tot > 0 && break
        n_T, n_L, n_R, _, _, _ = t
        raw[t] = primary_vertex(n_L, n_R, n_T, geom, R)
    end

    # Recursion for level ≥ 1
    for t in triples
        N_tot = total_level(t)
        N_tot == 0 && continue
        raw[t] = _recurse_entry(t, raw, basis_bond, basis_phys,
                                neumann, J_bond, J_phys)
    end

    raw
end

"""
Compute a single vertex entry by applying the Ward identity once.
Pick an arm with a nonempty partition, apply J_{-m} recursion,
reducing to already-computed lower-level entries.
"""
function _recurse_entry(t::NTuple{6, Int},
                        raw::Dict{NTuple{6, Int}, Float64},
                        basis_bond::FockBasis, basis_phys::FockBasis,
                        neumann::NeumannData,
                        J_bond, J_phys)
    n_T, n_L, n_R, αT, αL, αR = t

    λ_T = basis_phys.states[n_T][αT]
    λ_L = basis_bond.states[n_L][αL]
    λ_R = basis_bond.states[n_R][αR]

    # Pick an arm with a nonempty partition; prefer L → R → T
    arm::Symbol = :none
    m::Int = 0
    if !isempty(λ_L)
        arm = :L; m = λ_L[1]  # largest part (parts are weakly decreasing)
    elseif !isempty(λ_R)
        arm = :R; m = λ_R[1]
    elseif !isempty(λ_T)
        arm = :T; m = λ_T[1]
    else
        error("All partitions empty but total_level > 0")
    end

    # Residual state on arm `arm` after removing one copy of `m`
    # In the normalized basis: |λ̂⟩ = (1/√(m·m_k(λ))) J_{-m} |μ̂⟩
    # where μ = λ \ {m} and m_k = multiplicity of m in λ.
    # So |λ̂⟩ = (1/√(m·m_k)) J_{-m} |μ̂⟩
    # ⟹ V(..., |λ̂_arm⟩, ...) = (1/√(m·m_k)) V(..., J_{-m}|μ̂_arm⟩, ...)
    # And V(..., J_{-m}|μ̂_arm⟩, ...) = -Σ_j Σ_{k≥0} N^{arm→j}_{m,k} V(..., J_k|state_j⟩, ...)

    mk = count(==(m), _arm_partition(arm, λ_L, λ_R, λ_T))
    norm_factor = 1.0 / sqrt(m * mk)

    # Remove one copy of m from the arm's partition to get μ
    if arm == :L
        μ = _remove_part(λ_L, m)
        αL_new = _find_partition(basis_bond.states[n_L], μ)
        @assert αL_new !== nothing
        t_residual = (n_T, n_L, n_R, αT, αL_new, αR)
    elseif arm == :R
        μ = _remove_part(λ_R, m)
        αR_new = _find_partition(basis_bond.states[n_R], μ)
        @assert αR_new !== nothing
        t_residual = (n_T, n_L, n_R, αT, αL, αR_new)
    else # :T
        μ = _remove_part(λ_T, m)
        αT_new = _find_partition(basis_phys.states[n_T], μ)
        @assert αT_new !== nothing
        t_residual = (n_T, n_L, n_R, αT_new, αL, αR)
    end

    # Apply Ward identity: sum over arms j and mode k ≥ 0
    # V(..., J_{-m}|residual⟩, ...) = -Σ_j Σ_k N^{arm→j}_{m,k} · V(..., J_k acting on arm j, ...)

    result_ward = 0.0
    for j in (:L, :R, :T), k in 0:_k_max_for(arm, j, neumann, m)
        N_coeff = _neumann_coeff(neumann, arm, j, m, k)
        N_coeff == 0.0 && continue

        # Apply J_k on arm j to the appropriate state in t_residual
        # Returns a dict: state_index -> coefficient
        contrib = _apply_Jk_on_arm(t_residual, j, k, basis_bond, basis_phys,
                                   J_bond, J_phys, raw)
        result_ward += N_coeff * contrib
    end

    value = -norm_factor * result_ward
    value
end

function _arm_partition(arm::Symbol, λ_L, λ_R, λ_T)
    arm == :L ? λ_L : arm == :R ? λ_R : λ_T
end

function _k_max_for(arm::Symbol, j::Symbol, neumann::NeumannData, m::Int)
    key = Symbol(arm, j)
    mat = getfield(neumann.𝒩, key)
    size(mat, 2) - 1  # k index is 0, 1, ..., size(mat,2)-1
end

function _neumann_coeff(neumann::NeumannData, i::Symbol, j::Symbol, m::Int, k::Int)
    key = Symbol(i, j)
    mat = getfield(neumann.𝒩, key)
    (m > size(mat, 1) || k + 1 > size(mat, 2)) && return 0.0
    mat[m, k + 1]
end

"""
Apply J_k on arm `j` to the state in `t`, then look up the resulting
vertex values (weighted by the J_k matrix coefficients).
Returns: sum over resulting states of (J_k coefficient) · V(resulting state).
"""
function _apply_Jk_on_arm(t::NTuple{6, Int}, j::Symbol, k::Int,
                          basis_bond::FockBasis, basis_phys::FockBasis,
                          J_bond, J_phys,
                          raw::Dict{NTuple{6, Int}, Float64})
    n_T, n_L, n_R, αT, αL, αR = t
    result = 0.0

    if j == :L
        Jk_mat = J_bond[n_L][k + 1]  # index k+1 for stored J_k
        d = size(Jk_mat, 1)
        for αL_new in 1:d
            coeff = Jk_mat[αL_new, αL]
            coeff == 0.0 && continue
            t_new = (n_T, n_L, n_R, αT, αL_new, αR)
            haskey(raw, t_new) || continue
            result += coeff * raw[t_new]
        end
    elseif j == :R
        Jk_mat = J_bond[n_R][k + 1]
        d = size(Jk_mat, 1)
        for αR_new in 1:d
            coeff = Jk_mat[αR_new, αR]
            coeff == 0.0 && continue
            t_new = (n_T, n_L, n_R, αT, αL, αR_new)
            haskey(raw, t_new) || continue
            result += coeff * raw[t_new]
        end
    else # :T
        Jk_mat = J_phys[n_T][k + 1]
        d = size(Jk_mat, 1)
        for αT_new in 1:d
            coeff = Jk_mat[αT_new, αT]
            coeff == 0.0 && continue
            t_new = (n_T, n_L, n_R, αT_new, αL, αR)
            haskey(raw, t_new) || continue
            result += coeff * raw[t_new]
        end
    end
    result
end

"""
Assemble the raw vertex values into a TensorMap:
vertex : ℂ ← V_phys ⊗ V_bond ⊗ V_bond  (trilinear form)
"""
function _assemble_vertex(raw::Dict{NTuple{6, Int}, Float64},
                          basis_bond::FockBasis, basis_phys::FockBasis)
    V_b = basis_bond.V
    V_p = basis_phys.V

    vertex = zeros(Float64, one(V_p), V_p ⊗ V_b ⊗ V_b)

    for (f₁, f₂) in fusiontrees(vertex)
        # f₂ has 3 domain legs: V_phys, V_bond, V_bond
        n_T = Int(f₂.uncoupled[1].charge)
        n_L = Int(f₂.uncoupled[2].charge)
        n_R = Int(f₂.uncoupled[3].charge)

        (haskey(basis_phys.states, n_T) &&
         haskey(basis_bond.states, n_L) &&
         haskey(basis_bond.states, n_R)) || continue

        blk = vertex[f₁, f₂]
        # blk is 3-dimensional: (d_T, d_L, d_R)
        for αT in 1:size(blk, 1), αL in 1:size(blk, 2), αR in 1:size(blk, 3)
            key = (n_T, n_L, n_R, αT, αL, αR)
            if haskey(raw, key)
                blk[αT, αL, αR] = raw[key]
            end
        end
        vertex[f₁, f₂] = blk
    end

    vertex
end

"""
    charge_block(vd::VertexData, n_L, n_R, n_T) -> Array{Float64, 3}

Extract the block of the vertex for a specific charge triple.
Returns a 3D array indexed by [α_L, α_R, α_T] (spec convention).
Reads directly from the TensorMap.
"""
function charge_block(vd::VertexData, n_L::Int, n_R::Int, n_T::Int)
    V = vd.vertex
    for (f1, f2) in fusiontrees(V)
        fn_T = Int(f2.uncoupled[1].charge)
        fn_L = Int(f2.uncoupled[2].charge)
        fn_R = Int(f2.uncoupled[3].charge)
        (fn_T == n_T && fn_L == n_L && fn_R == n_R) || continue
        blk = V[f1, f2]
        # blk is (d_T, d_L, d_R); permute to spec convention (d_L, d_R, d_T)
        return permutedims(Array(blk), (2, 3, 1))
    end
    return zeros(Float64, 0, 0, 0)
end

"""
    conformal_dim(basis::FockBasis, n::Int, α::Int) -> Float64

Total conformal dimension h_n + N of state α in sector n.
"""
function conformal_dim(basis::FockBasis, n::Int, α::Int)
    (n / basis.R)^2 / 2 + basis.levels[n][α]
end

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
    build_propagator_factor(basis::FockBasis, ell, c) -> TensorMap

Diagonal endomorphism V -> V with eigenvalue exp(pi*ell/2*(h_n+N-c/24))
on each basis state. This is e^{H*ell/2} where H = pi(L_0 - c/24) on
a strip of unit width.
"""
function build_propagator_factor(basis::FockBasis, ell::Float64, c::Float64)
    D = zeros(Float64, basis.V, basis.V)
    for (f1, f2) in fusiontrees(D)
        n = Int(f2.uncoupled[1].charge)
        haskey(basis.levels, n) || continue
        blk = D[f1, f2]
        for a in axes(blk, 1)
            blk[a, a] = exp(pi * ell / 2 * (conformal_dim(basis, n, a) - c / 24))
        end
        D[f1, f2] = blk
    end
    D
end

"""
    modified_vertex(vd::VertexData; c=1.0) -> TensorMap

Compute the modified vertex Ṽ = V ∘ (id_phys ⊗ D ⊗ D) where
D = e^{H ℓ/2} is the propagator factor on each bond arm.

This is a pure TensorKit composition — no block iteration.
"""
function modified_vertex(vd::VertexData; c::Float64=1.0)
    D = build_propagator_factor(vd.basis_bond, vd.ell, c)
    vd.vertex * (id(vd.basis_phys.V) ⊗ D ⊗ D)
end

"""
    modified_vertex_cache(cft, ells; c=1.0, series_order=20) -> Dict{Float64, TensorMap}

Precompute and cache modified vertices for a sweep over ell values.
"""
function modified_vertex_cache(cft::CompactBosonCFT, ells;
                               c::Float64=1.0, series_order::Int=20)
    Dict(Float64(l) => modified_vertex(compute_vertex(cft, l; series_order); c=c)
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

"""
    build_selector(basis::FockBasis, n::Int, alpha::Int) -> TensorMap

Build a charge selector: a TensorMap `V_{-n} ← V'` that picks out the
alpha-th basis state at charge n.

Used for charged contractions: `selector * permute(vertex, ((1,),(2,3)))`
gives a charged TensorMap representing the vertex with one leg fixed.

The codomain V_{-n} is a 1D space at charge -n (the dual-charge convention
from the permute-induced dualization). The domain is V' (the dual space).
"""
function build_selector(basis::FockBasis, n::Int, alpha::Int)
    V_neg_n = Vect[U1Irrep](U1Irrep(-n) => 1)
    sel = zeros(Float64, V_neg_n, basis.V')
    for (f1, f2) in fusiontrees(sel)
        fn = Int(f2.uncoupled[1].charge)
        fn == -n || continue
        blk = sel[f1, f2]
        alpha <= size(blk, 2) || continue
        blk[1, alpha] = 1.0
        sel[f1, f2] = blk
    end
    sel
end

"""
    weight_shells(basis::FockBasis) -> Vector{@NamedTuple{h::Float64, states::Vector{Tuple{Int,Int}}}}

Enumerate all distinct conformal weights in the basis with their (n, α) pairs.
"""
function weight_shells(basis::FockBasis)
    h_map = Dict{Float64, Vector{Tuple{Int,Int}}}()
    for n in keys(basis.states), a in eachindex(basis.states[n])
        h = round(conformal_dim(basis, n, a); digits=6)
        push!(get!(h_map, h, []), (n, a))
    end
    [(h=h, states=states) for (h, states) in sort(collect(h_map); by=first)]
end

"""
    random_unit_vec(states; rng) -> Vector{Tuple{Tuple{Int,Int}, Float64}}

Build a random unit vector over the given (n, α) states.
Returns a list of ((n, α), coefficient) pairs.
"""
function random_unit_vec(states::Vector{Tuple{Int,Int}}; rng=Random.GLOBAL_RNG)
    coeffs = randn(rng, length(states))
    nrm = norm(coeffs)
    nrm > 0 && (coeffs ./= nrm)
    collect(zip(states, coeffs))
end
