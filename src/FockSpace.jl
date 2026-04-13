"""
U(1)-graded Fock space for the compact boson at radius R with cutoff h_max.
"""

struct FockBasis
    R::Float64
    h_max::Float64
    states::Dict{Int, Vector{Vector{Int}}}   # sector n -> partitions (weakly decreasing)
    levels::Dict{Int, Vector{Int}}           # sector n -> level of each state
    z_lambda::Dict{Int, Vector{Float64}}     # sector n -> normalization factors
    V::GradedSpace                           # U1-graded TensorKit space
    partition_index::Dict{Int, Dict{Vector{Int}, Int}}  # sector n -> (partition -> basis index)
end

"""
    build_fock_basis(R, h_max) -> FockBasis

Enumerate the truncated Fock space for the compact boson at radius R.
Momentum sectors n with h_n = (n/R)²/2 ≤ h_max are included.
Within each sector, descendant states with level |λ| ≤ h_max - h_n.
"""
function build_fock_basis(R::Real, h_max::Real)
    R = Float64(R)
    h_max = Float64(h_max)

    # Determine momentum sectors: |n| ≤ R√(2·h_max)
    n_max = floor(Int, R * sqrt(2 * h_max) + 1e-10)

    states = Dict{Int, Vector{Vector{Int}}}()
    levels = Dict{Int, Vector{Int}}()
    z_lambda = Dict{Int, Vector{Float64}}()

    for n in -n_max:n_max
        h_n = (n / R)^2 / 2
        h_n > h_max + 1e-10 && continue
        max_level = floor(Int, h_max - h_n + 1e-10)

        parts = _partitions_up_to(max_level)
        lvls = [sum(p; init=0) for p in parts]
        zls = [_compute_z_lambda(p) for p in parts]

        states[n] = parts
        levels[n] = lvls
        z_lambda[n] = zls
    end

    # Build TensorKit GradedSpace
    sector_dims = Pair{U1Irrep, Int}[]
    for n in sort(collect(keys(states)))
        push!(sector_dims, U1Irrep(n) => length(states[n]))
    end
    V = Vect[U1Irrep](sector_dims...)

    # Pre-build partition → index hashmap for O(1) lookup
    partition_index = Dict{Int, Dict{Vector{Int}, Int}}()
    for n in keys(states)
        pidx = Dict{Vector{Int}, Int}()
        for (i, p) in enumerate(states[n])
            pidx[p] = i
        end
        partition_index[n] = pidx
    end

    FockBasis(R, h_max, states, levels, z_lambda, V, partition_index)
end

"""
Enumerate all integer partitions with sum ≤ max_level.
Ordered by level (sum), then reverse-lexicographic within each level.
Parts are in weakly decreasing order.
"""
function _partitions_up_to(max_level::Int)
    result = Vector{Vector{Int}}()
    push!(result, Int[])  # empty partition (level 0)
    for level in 1:max_level
        append!(result, _partitions_of(level))
    end
    result
end

"""
All partitions of integer n, in reverse-lexicographic order.
Each partition is a vector of parts in weakly decreasing order.
"""
function _partitions_of(n::Int)
    n == 0 && return [Int[]]
    result = Vector{Vector{Int}}()
    _partition_helper!(result, Int[], n, n)
    result
end

function _partition_helper!(result::Vector{Vector{Int}}, current::Vector{Int},
                           remaining::Int, max_part::Int)
    if remaining == 0
        push!(result, copy(current))
        return
    end
    for part in min(remaining, max_part):-1:1
        push!(current, part)
        _partition_helper!(result, current, remaining - part, part)
        pop!(current)
    end
end

"""
Compute z_λ = ∏_j j^{m_j} · m_j! where m_j is the multiplicity of part j in λ.
"""
function _compute_z_lambda(lambda::Vector{Int})
    isempty(lambda) && return 1.0
    z = 1.0
    j = lambda[1]
    count = 1
    for i in 2:length(lambda)
        if lambda[i] == j
            count += 1
        else
            z *= Float64(j)^count * factorial(count)
            j = lambda[i]
            count = 1
        end
    end
    z *= Float64(j)^count * factorial(count)
    z
end
