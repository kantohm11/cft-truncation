"""
U(1)-graded Fock space for the compact boson at radius R with per-charge
descendant-level cutoffs.
"""

struct FockBasis
    R::Float64
    h_max::Float64                                 # max h actually included (= max over charges)
    cutoffs::Dict{Int, Int}                        # sector n -> max descendant level
    states::Dict{Int, Vector{Vector{Int}}}         # sector n -> partitions (weakly decreasing)
    levels::Dict{Int, Vector{Int}}                 # sector n -> level of each state
    V::GradedSpace                                 # U1-graded TensorKit space
    partition_index::Dict{Int, Dict{Vector{Int}, Int}}  # sector n -> (partition -> basis index)
end

"""
    uniform_cutoffs(h_max, R) -> Dict{Int, Int}

Per-charge descendant-level cutoffs equivalent to the uniform-`h_max`
spec: for each charge n with h_n = (n/R)² / 2 ≤ h_max, set the level
cap to `floor(h_max − h_n)`. The backward-compat path for the old
`build_fock_basis(R, h_max)` and `TruncationSpec(h)` constructors.
"""
function uniform_cutoffs(h_max::Real, R::Real)
    R_f = Float64(R)
    h_max_f = Float64(h_max)
    n_max = floor(Int, R_f * sqrt(2 * h_max_f) + 1e-10)
    cutoffs = Dict{Int, Int}()
    for n in -n_max:n_max
        h_n = (n / R_f)^2 / 2
        h_n > h_max_f + 1e-10 && continue
        cutoffs[n] = floor(Int, h_max_f - h_n + 1e-10)
    end
    cutoffs
end

"""
    build_fock_basis(R, cutoffs::Dict{Int, Int}) -> FockBasis

Enumerate the truncated Fock space at radius `R` with per-charge
descendant-level caps `cutoffs[n]`. Charge sector `n` is included iff
`n ∈ keys(cutoffs)`, with partitions of total level `≤ cutoffs[n]`.

For backward compatibility, `build_fock_basis(R, h_max::Real)` is also
accepted: it constructs `cutoffs = uniform_cutoffs(h_max, R)` and calls
this method.
"""
function build_fock_basis(R::Real, cutoffs::Dict{Int, Int})
    R_f = Float64(R)

    states = Dict{Int, Vector{Vector{Int}}}()
    levels = Dict{Int, Vector{Int}}()

    for (n, max_level) in cutoffs
        max_level >= 0 || continue
        parts = _partitions_up_to(max_level)
        lvls = [sum(p; init=0) for p in parts]

        states[n] = parts
        levels[n] = lvls
    end

    # Effective h_max for downstream code that reads it.
    h_max_eff = isempty(states) ? 0.0 :
        maximum((n/R_f)^2/2 + cutoffs[n] for n in keys(states))

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

    FockBasis(R_f, h_max_eff, cutoffs, states, levels, V, partition_index)
end

# Backward-compat: uniform h_max cap.
build_fock_basis(R::Real, h_max::Real) = build_fock_basis(R, uniform_cutoffs(h_max, R))

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
