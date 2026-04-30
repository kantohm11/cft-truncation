"""
Build the cross-shape (4-arm) vertex tensor via Ward identity recursion.

Split off from `Recursion.jl`. The T-shape recursion lives in
`VertexRecursion.jl`, and the public dispatcher
`compute_vertex(cft, ell; shape=:cross)` is defined there. This file
provides `_build_vertex_cross`, the (geom, neumann)-keyed
`compute_vertex(cft, geom::GeometryCross, neumann; ell)` overload, and
`charge_block(::VertexDataCross, …)`.

Must be `include`d after `VertexRecursion.jl` and `LocalCoordinatesCross.jl`.
"""

# ============================================================================
# Cross-shape (4-arm) vertex: Ward recursion for the cross geometry.
# ============================================================================
#
# Leg convention: ℂ ← V_phys ⊗ V_phys ⊗ V_bond ⊗ V_bond, ordered as (T, B, L, R).
# Conservation: n_L + n_R + n_T + n_B = 0.
# T and B share the physical basis (basis_phys); L and R share the bond basis.

struct VertexDataCross
    cft::CompactBosonCFT
    vertex::TensorMap                 # ℂ ← V_phys ⊗ V_phys ⊗ V_bond ⊗ V_bond
    geom::GeometryCross
    neumann::NeumannDataCross
    ell::Float64
end

"""
    VertexArrayCross

Dense storage for cross-vertex entries indexed by
`(n_T, n_B, n_L, n_R, αT, αB, αL, αR)`. Mirrors `VertexArray` but with
4 charge axes and conservation `n_L + n_R + n_T + n_B = 0`.

Memory layout: offsets is a 4D array over `(n_T, n_B, n_L, n_R)`; each
valid quadruple addresses a contiguous `d_T · d_B · d_L · d_R` block,
stored row-major as
  flat = base + (αT-1)·d_B·d_L·d_R + (αB-1)·d_L·d_R + (αL-1)·d_R + αR.
"""
struct VertexArrayCross
    data::Vector{Float64}
    offsets::Array{Int, 4}
    dims::Array{NTuple{4,Int}, 4}
    n_off::Int
end

@inline function Base.getindex(va::VertexArrayCross,
                               n_T::Int, n_B::Int, n_L::Int, n_R::Int,
                               αT::Int, αB::Int, αL::Int, αR::Int)
    @inbounds begin
        off = va.offsets[n_T + va.n_off, n_B + va.n_off,
                         n_L + va.n_off, n_R + va.n_off]
        off == 0 && return 0.0
        d_T, d_B, d_L, d_R = va.dims[n_T + va.n_off, n_B + va.n_off,
                                     n_L + va.n_off, n_R + va.n_off]
        va.data[off + (αT-1)*d_B*d_L*d_R + (αB-1)*d_L*d_R + (αL-1)*d_R + αR]
    end
end

@inline function Base.setindex!(va::VertexArrayCross, val::Float64,
                                n_T::Int, n_B::Int, n_L::Int, n_R::Int,
                                αT::Int, αB::Int, αL::Int, αR::Int)
    @inbounds begin
        off = va.offsets[n_T + va.n_off, n_B + va.n_off,
                         n_L + va.n_off, n_R + va.n_off]
        d_T, d_B, d_L, d_R = va.dims[n_T + va.n_off, n_B + va.n_off,
                                     n_L + va.n_off, n_R + va.n_off]
        va.data[off + (αT-1)*d_B*d_L*d_R + (αB-1)*d_L*d_R + (αL-1)*d_R + αR] = val
    end
end

function _build_vertex_array_cross(basis_bond::FockBasis, basis_phys::FockBasis)
    sectors_bond = sort(collect(keys(basis_bond.states)))
    sectors_phys = sort(collect(keys(basis_phys.states)))
    n_min = min(minimum(sectors_bond), minimum(sectors_phys))
    n_max = max(maximum(sectors_bond), maximum(sectors_phys))
    n_off = 1 - n_min
    n_range = n_max - n_min + 1

    offsets = zeros(Int, n_range, n_range, n_range, n_range)
    dims = fill((0,0,0,0), n_range, n_range, n_range, n_range)
    total = 0
    for n_L in sectors_bond, n_R in sectors_bond, n_T in sectors_phys
        n_B = -(n_L + n_R + n_T)
        n_B in sectors_phys || continue
        d_T = length(basis_phys.states[n_T])
        d_B = length(basis_phys.states[n_B])
        d_L = length(basis_bond.states[n_L])
        d_R = length(basis_bond.states[n_R])
        offsets[n_T + n_off, n_B + n_off, n_L + n_off, n_R + n_off] = total
        dims[n_T + n_off, n_B + n_off, n_L + n_off, n_R + n_off] =
            (d_T, d_B, d_L, d_R)
        total += d_T * d_B * d_L * d_R
    end
    VertexArrayCross(zeros(Float64, total), offsets, dims, n_off)
end

"""Compute all cross-vertex entries by sweeping total level."""
function _compute_vertex_raw_cross(basis_bond::FockBasis, basis_phys::FockBasis,
                                   geom::GeometryCross, neumann::NeumannDataCross,
                                   J_bond_sp, J_phys_sp, R::Float64)
    raw = _build_vertex_array_cross(basis_bond, basis_phys)

    sectors_bond = collect(keys(basis_bond.states))
    sectors_phys = collect(keys(basis_phys.states))

    # Enumerate valid 8-tuples (n_T, n_B, n_L, n_R, αT, αB, αL, αR).
    tuples = NTuple{8, Int}[]
    for n_L in sectors_bond, n_R in sectors_bond, n_T in sectors_phys
        n_B = -(n_L + n_R + n_T)
        n_B in sectors_phys || continue
        for αL in eachindex(basis_bond.states[n_L]),
            αR in eachindex(basis_bond.states[n_R]),
            αT in eachindex(basis_phys.states[n_T]),
            αB in eachindex(basis_phys.states[n_B])
            push!(tuples, (n_T, n_B, n_L, n_R, αT, αB, αL, αR))
        end
    end

    total_level(t) = basis_bond.levels[t[3]][t[7]] +
                     basis_bond.levels[t[4]][t[8]] +
                     basis_phys.levels[t[1]][t[5]] +
                     basis_phys.levels[t[2]][t[6]]
    sort!(tuples, by=total_level)

    # Level 0: primary vertex on the all-vacuum (partitions empty) states.
    for t in tuples
        total_level(t) == 0 || break
        n_T, n_B, n_L, n_R, αT, αB, αL, αR = t
        raw[n_T, n_B, n_L, n_R, αT, αB, αL, αR] =
            primary_vertex(n_L, n_R, n_T, n_B, geom, R)
    end

    # Level ≥ 1: Ward recursion.
    for t in tuples
        total_level(t) == 0 && continue
        n_T, n_B, n_L, n_R, αT, αB, αL, αR = t
        raw[n_T, n_B, n_L, n_R, αT, αB, αL, αR] = _recurse_entry_cross(
            t, raw, basis_bond, basis_phys, neumann, J_bond_sp, J_phys_sp)
    end

    raw
end

"""Apply the Ward identity once for a single cross-vertex entry."""
function _recurse_entry_cross(t::NTuple{8, Int},
                              raw::VertexArrayCross,
                              basis_bond::FockBasis, basis_phys::FockBasis,
                              neumann::NeumannDataCross,
                              J_bond_sp, J_phys_sp)
    n_T, n_B, n_L, n_R, αT, αB, αL, αR = t
    λ_T = basis_phys.states[n_T][αT]
    λ_B = basis_phys.states[n_B][αB]
    λ_L = basis_bond.states[n_L][αL]
    λ_R = basis_bond.states[n_R][αR]

    arm::Symbol = :none
    m::Int = 0
    if !isempty(λ_L)
        arm = :L; m = λ_L[1]
    elseif !isempty(λ_R)
        arm = :R; m = λ_R[1]
    elseif !isempty(λ_T)
        arm = :T; m = λ_T[1]
    elseif !isempty(λ_B)
        arm = :B; m = λ_B[1]
    else
        error("All partitions empty but total_level > 0")
    end

    mk = count(==(m), _arm_partition_cross(arm, λ_L, λ_R, λ_T, λ_B))
    norm_factor = 1.0 / sqrt(m * mk)

    # Remove one copy of `m` from the arm's partition to get the residual state.
    if arm == :L
        μ = _remove_part(λ_L, m)
        αL_new = basis_bond.partition_index[n_L][μ]
        t_residual = (n_T, n_B, n_L, n_R, αT, αB, αL_new, αR)
    elseif arm == :R
        μ = _remove_part(λ_R, m)
        αR_new = basis_bond.partition_index[n_R][μ]
        t_residual = (n_T, n_B, n_L, n_R, αT, αB, αL, αR_new)
    elseif arm == :T
        μ = _remove_part(λ_T, m)
        αT_new = basis_phys.partition_index[n_T][μ]
        t_residual = (n_T, n_B, n_L, n_R, αT_new, αB, αL, αR)
    else  # :B
        μ = _remove_part(λ_B, m)
        αB_new = basis_phys.partition_index[n_B][μ]
        t_residual = (n_T, n_B, n_L, n_R, αT, αB_new, αL, αR)
    end

    result_ward = 0.0
    for j in (:L, :R, :T, :B), k in 0:_k_max_for_cross(arm, j, neumann, m)
        N_coeff = _neumann_coeff_cross(neumann, arm, j, m, k)
        N_coeff == 0.0 && continue
        contrib = _apply_Jk_on_arm_sparse_cross(t_residual, j, k,
                                                J_bond_sp, J_phys_sp, raw)
        result_ward += N_coeff * contrib
    end

    -norm_factor * result_ward
end

function _arm_partition_cross(arm::Symbol, λ_L, λ_R, λ_T, λ_B)
    arm == :L ? λ_L :
    arm == :R ? λ_R :
    arm == :T ? λ_T : λ_B
end

function _k_max_for_cross(arm::Symbol, j::Symbol, neumann::NeumannDataCross, m::Int)
    key = Symbol(arm, j)
    mat = getfield(neumann.𝒩, key)
    size(mat, 2) - 1
end

function _neumann_coeff_cross(neumann::NeumannDataCross,
                              i::Symbol, j::Symbol, m::Int, k::Int)
    key = Symbol(i, j)
    mat = getfield(neumann.𝒩, key)
    (m > size(mat, 1) || k + 1 > size(mat, 2)) && return 0.0
    mat[m, k + 1]
end

"""Apply J_k on arm j to the state in `t`, returning J_k coefficient × V(resulting)."""
function _apply_Jk_on_arm_sparse_cross(t::NTuple{8, Int}, j::Symbol, k::Int,
                                       J_bond_sp, J_phys_sp,
                                       raw::VertexArrayCross)
    n_T, n_B, n_L, n_R, αT, αB, αL, αR = t
    if j == :L
        (target, coeff) = J_bond_sp[n_L][k + 1][αL]
        target == 0 && return 0.0
        return coeff * raw[n_T, n_B, n_L, n_R, αT, αB, target, αR]
    elseif j == :R
        (target, coeff) = J_bond_sp[n_R][k + 1][αR]
        target == 0 && return 0.0
        return coeff * raw[n_T, n_B, n_L, n_R, αT, αB, αL, target]
    elseif j == :T
        (target, coeff) = J_phys_sp[n_T][k + 1][αT]
        target == 0 && return 0.0
        return coeff * raw[n_T, n_B, n_L, n_R, target, αB, αL, αR]
    else  # :B
        (target, coeff) = J_phys_sp[n_B][k + 1][αB]
        target == 0 && return 0.0
        return coeff * raw[n_T, n_B, n_L, n_R, αT, target, αL, αR]
    end
end

"""Assemble the raw cross-vertex values into a 4-leg TensorMap
`ℂ ← V_phys ⊗ V_phys ⊗ V_bond ⊗ V_bond`, legs ordered as (T, B, L, R)."""
function _assemble_vertex_cross(raw::VertexArrayCross,
                                basis_bond::FockBasis, basis_phys::FockBasis)
    V_b = basis_bond.V
    V_p = basis_phys.V
    vertex = zeros(Float64, one(V_p), V_p ⊗ V_p ⊗ V_b ⊗ V_b)

    for (f₁, f₂) in fusiontrees(vertex)
        n_T = Int(f₂.uncoupled[1].charge)
        n_B = Int(f₂.uncoupled[2].charge)
        n_L = Int(f₂.uncoupled[3].charge)
        n_R = Int(f₂.uncoupled[4].charge)

        (haskey(basis_phys.states, n_T) && haskey(basis_phys.states, n_B) &&
         haskey(basis_bond.states, n_L) && haskey(basis_bond.states, n_R)) || continue

        blk = vertex[f₁, f₂]
        for αT in 1:size(blk, 1), αB in 1:size(blk, 2),
            αL in 1:size(blk, 3), αR in 1:size(blk, 4)
            blk[αT, αB, αL, αR] = raw[n_T, n_B, n_L, n_R, αT, αB, αL, αR]
        end
        vertex[f₁, f₂] = blk
    end
    vertex
end

function _build_vertex_cross(cft::CompactBosonCFT, geom::GeometryCross,
                             neumann::NeumannDataCross, ell::Float64)
    raw = _compute_vertex_raw_cross(cft.basis_bond, cft.basis_phys, geom, neumann,
                                    cft.J_bond_sp, cft.J_phys_sp, cft.R)
    vertex = _assemble_vertex_cross(raw, cft.basis_bond, cft.basis_phys)
    VertexDataCross(cft, vertex, geom, neumann, ell)
end

"""
    compute_vertex(cft, geom::GeometryCross, neumann::NeumannDataCross; ell) -> VertexDataCross

Build the 4-arm cross vertex using pre-computed cross geometry and
Neumann coefficients. Caching for this path is not yet supported — call
without relying on a cache dir (use the lower-level entry directly).
"""
function compute_vertex(cft::CompactBosonCFT, geom::GeometryCross,
                        neumann::NeumannDataCross; ell::Real)
    _build_vertex_cross(cft, geom, neumann, Float64(ell))
end

"""
    charge_block(vd::VertexDataCross, n_L, n_R, n_T, n_B) -> Array{Float64, 4}

Extract the block of the cross vertex for a specific charge quadruple.
Returns a 4D array indexed by [α_L, α_R, α_T, α_B] (spec convention).
"""
function charge_block(vd::VertexDataCross,
                      n_L::Int, n_R::Int, n_T::Int, n_B::Int)
    V = vd.vertex
    for (f1, f2) in fusiontrees(V)
        fn_T = Int(f2.uncoupled[1].charge)
        fn_B = Int(f2.uncoupled[2].charge)
        fn_L = Int(f2.uncoupled[3].charge)
        fn_R = Int(f2.uncoupled[4].charge)
        (fn_T == n_T && fn_B == n_B && fn_L == n_L && fn_R == n_R) || continue
        blk = V[f1, f2]
        # blk is (d_T, d_B, d_L, d_R); permute to spec convention (d_L, d_R, d_T, d_B).
        return permutedims(Array(blk), (3, 4, 1, 2))
    end
    return zeros(Float64, 0, 0, 0, 0)
end
