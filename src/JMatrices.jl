"""
J_k mode action matrices in the normalised Fock basis.

Two representations are built:
- Dense matrices (for tests, commutation checks, general use)
- Sparse column lists `J_sparse`: for each (sector, k, source_col),
  stores (target_row, coefficient). Since J_k removes exactly one part,
  each column has at most 1 nonzero entry. The recursion uses this
  for O(1) per (arm, k, state) instead of O(d) row scan.
"""

"""
    build_J_matrices(basis::FockBasis, k_max::Int)

Build annihilation mode matrices J_k (k = 0, 1, ..., k_max) for each momentum sector.

Returns `(dense, sparse)` where:
- `dense`: Dict{Int, Vector{Matrix{Float64}}} — sector n -> [J_0, ..., J_{k_max}]
- `sparse`: Dict{Int, Vector{Vector{Tuple{Int,Float64}}}} — sector n -> [cols_0, ..., cols_{k_max}]
  where cols_k[col] = (target_row, coefficient) or (0, 0.0) if zero column.

The sparse form exploits that J_k (k≥1) has at most 1 nonzero per column.
"""
function build_J_matrices(basis::FockBasis, k_max::Int)
    dense = Dict{Int, Vector{Matrix{Float64}}}()
    sparse = Dict{Int, Vector{Vector{Tuple{Int,Float64}}}}()

    for n in keys(basis.states)
        parts = basis.states[n]
        d = length(parts)
        matrices = Vector{Matrix{Float64}}()
        sparse_cols = Vector{Vector{Tuple{Int,Float64}}}()

        # J_0 = (n/R) · I — diagonal, every column has one nonzero
        J0_val = n / basis.R
        push!(matrices, J0_val * Matrix{Float64}(I, d, d))
        push!(sparse_cols, [(i, J0_val) for i in 1:d])

        # J_k for k = 1, ..., k_max
        for k in 1:k_max
            Jk = zeros(Float64, d, d)
            cols_k = Vector{Tuple{Int,Float64}}(undef, d)
            for col in 1:d
                cols_k[col] = (0, 0.0)  # default: zero column
            end
            for (col, lambda) in enumerate(parts)
                mk = count(==(k), lambda)
                mk == 0 && continue
                mu = _remove_part(lambda, k)
                row = _find_partition(parts, mu)
                row === nothing && continue
                coeff = sqrt(k * mk)
                Jk[row, col] = coeff
                cols_k[col] = (row, coeff)
            end
            push!(matrices, Jk)
            push!(sparse_cols, cols_k)
        end

        dense[n] = matrices
        sparse[n] = sparse_cols
    end
    dense, sparse
end

"""
    build_creation_matrix(basis::FockBasis, n::Int, k::Int) -> Matrix{Float64}

Build the creation mode matrix J_{-k} in sector n.
In the normalised basis: J_{-k}|λ̂⟩ = √(k · (m_k(λ)+1)) |λ∪{k}⟩_hat
"""
function build_creation_matrix(basis::FockBasis, n::Int, k::Int)
    parts = basis.states[n]
    d = length(parts)
    Jmk = zeros(Float64, d, d)
    for (col, lambda) in enumerate(parts)
        mk = count(==(k), lambda)
        # Add part k to lambda
        mu = _add_part(lambda, k)
        # Find mu in the basis
        row = _find_partition(parts, mu)
        row === nothing && continue
        # Coefficient: √(k · (m_k(λ) + 1))
        Jmk[row, col] = sqrt(k * (mk + 1))
    end
    Jmk
end

function _remove_part(lambda::Vector{Int}, k::Int)
    mu = copy(lambda)
    idx = findfirst(==(k), mu)
    idx === nothing && error("Part $k not found in $lambda")
    deleteat!(mu, idx)
    mu
end

function _add_part(lambda::Vector{Int}, k::Int)
    mu = copy(lambda)
    # Insert k in the right position to maintain weakly decreasing order
    pos = searchsortedfirst(mu, k; rev=true)
    insert!(mu, pos, k)
    mu
end

function _find_partition(parts::Vector{Vector{Int}}, target::Vector{Int})
    for (i, p) in enumerate(parts)
        p == target && return i
    end
    nothing
end
