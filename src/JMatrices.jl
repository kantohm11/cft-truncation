"""
J_k mode action matrices in the normalised Fock basis.
"""

"""
    build_J_matrices(basis::FockBasis, k_max::Int) -> Dict{Int, Vector{Matrix{Float64}}}

Build annihilation mode matrices J_k (k = 0, 1, ..., k_max) for each momentum sector.
Returns dict: sector n -> [J_0, J_1, ..., J_{k_max}] as dense matrices.

In the normalised basis |λ̂; n⟩ = J_{-λ}|n⟩ / √z_λ:
- J_0 = (n/R) · I
- J_k (k ≥ 1): removes a part k from λ. Coefficient √(k · m_k(λ)).
"""
function build_J_matrices(basis::FockBasis, k_max::Int)
    result = Dict{Int, Vector{Matrix{Float64}}}()
    for n in keys(basis.states)
        parts = basis.states[n]
        d = length(parts)
        matrices = Vector{Matrix{Float64}}()

        # J_0 = (n/R) · I
        push!(matrices, (n / basis.R) * Matrix{Float64}(I, d, d))

        # J_k for k = 1, ..., k_max
        for k in 1:k_max
            Jk = zeros(Float64, d, d)
            for (col, lambda) in enumerate(parts)
                # Check if lambda contains part k
                mk = count(==(k), lambda)
                mk == 0 && continue
                # Remove one copy of k from lambda to get mu
                mu = _remove_part(lambda, k)
                # Find mu in the basis
                row = _find_partition(parts, mu)
                row === nothing && continue
                # Coefficient: √(k · m_k(λ))
                Jk[row, col] = sqrt(k * mk)
            end
            push!(matrices, Jk)
        end

        result[n] = matrices
    end
    result
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
