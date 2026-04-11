"""
Build the T-vertex tensor via Ward identity recursion.
"""

struct TruncationSpec
    h_bond::Float64
    h_phys::Float64
end
TruncationSpec(h::Real) = TruncationSpec(Float64(h), Float64(h))
TruncationSpec(; h_bond::Real, h_phys::Real) = TruncationSpec(Float64(h_bond), Float64(h_phys))

struct VertexData
    vertex::TensorMap       # V_phys ⊗ V_bond ⊗ V_bond → ℂ
    geom::Geometry
    neumann::NeumannData
    basis_bond::FockBasis
    basis_phys::FockBasis
    R::Float64
    ell::Float64
    # Raw vertex values, keyed by (n_T, n_L, n_R, α_T, α_L, α_R)
    raw::Dict{NTuple{6, Int}, Float64}
end

"""
    compute_vertex(; R, ℓ, trunc::TruncationSpec, geom=nothing, neumann=nothing)

Compute the T-vertex for the compact boson at radius R, geometry parameter ℓ,
and truncation spec trunc. Returns VertexData.
"""
function compute_vertex(; R::Real, ell::Real, trunc::TruncationSpec,
                        geom=nothing, neumann=nothing, series_order::Int=20)
    R = Float64(R); ell = Float64(ell)

    if geom === nothing
        geom = compute_geometry(ell, series_order)
    end
    if neumann === nothing
        # Need modes up to max descendant level
        m_max = max(floor(Int, trunc.h_bond), floor(Int, trunc.h_phys)) + 2
        neumann = compute_neumann(geom, m_max)
    end

    basis_bond = build_fock_basis(R, trunc.h_bond)
    basis_phys = build_fock_basis(R, trunc.h_phys)

    # Build J-matrices for both bases
    k_max = max(floor(Int, trunc.h_bond), floor(Int, trunc.h_phys)) + 2
    J_bond = build_J_matrices(basis_bond, k_max)
    J_phys = build_J_matrices(basis_phys, k_max)

    # Compute the vertex values by sweeping total level
    raw = _compute_vertex_raw(basis_bond, basis_phys, geom, neumann,
                              J_bond, J_phys, R)

    # Assemble into a TensorMap: V_phys ⊗ V_bond ⊗ V_bond → ℂ
    # Codomain is trivial, domain is V_phys ⊗ V_bond ⊗ V_bond
    V_b = basis_bond.V
    V_p = basis_phys.V
    vertex = _assemble_vertex(raw, basis_bond, basis_phys)

    VertexData(vertex, geom, neumann, basis_bond, basis_phys, R, ell, raw)
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
"""
function charge_block(vd::VertexData, n_L::Int, n_R::Int, n_T::Int)
    (haskey(vd.basis_bond.states, n_L) &&
     haskey(vd.basis_bond.states, n_R) &&
     haskey(vd.basis_phys.states, n_T)) || return zeros(Float64, 0, 0, 0)

    d_L = length(vd.basis_bond.states[n_L])
    d_R = length(vd.basis_bond.states[n_R])
    d_T = length(vd.basis_phys.states[n_T])
    block = zeros(Float64, d_L, d_R, d_T)
    for αL in 1:d_L, αR in 1:d_R, αT in 1:d_T
        key = (n_T, n_L, n_R, αT, αL, αR)
        if haskey(vd.raw, key)
            block[αL, αR, αT] = vd.raw[key]
        end
    end
    block
end
