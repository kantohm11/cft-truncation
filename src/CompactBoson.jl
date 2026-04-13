"""
Compact-boson CFT data: a bundle of the truncated state spaces, mode-action
matrices, and BPZ form needed to compute T-vertices for a fixed CFT and
truncation. ℓ-independent — build once and reuse across an ℓ-sweep.
"""

"""
    TruncationSpec(h_bond, h_phys)
    TruncationSpec(h)               # symmetric: h_bond = h_phys = h
    TruncationSpec(; h_bond, h_phys)

Conformal-weight cutoff applied to the bond (L/R arms) and physical (T arm)
state spaces. Generic across CFT families: lives here for now because
`CompactBosonCFT` is currently the only consumer.
"""
struct TruncationSpec
    h_bond::Float64
    h_phys::Float64
end
TruncationSpec(h::Real) = TruncationSpec(Float64(h), Float64(h))
TruncationSpec(; h_bond::Real, h_phys::Real) = TruncationSpec(Float64(h_bond), Float64(h_phys))

"""
    CompactBosonCFT

Truncated boundary CFT data for the compact boson at radius R, cut off at
conformal weights `trunc.h_bond` (bond arms) and `trunc.h_phys` (physical arm).
Bundles the Fock bases, U(1) current modes `J_k`, and BPZ bilinear form on
the bond space. ℓ-independent.

Build once with `CompactBosonCFT(R=..., trunc=...)`, then reuse across calls
to `compute_vertex(cft, ell)` for different ℓ values.
"""
struct CompactBosonCFT
    R::Float64
    c::Float64                                     # central charge (c=1 for compact boson)
    trunc::TruncationSpec
    basis_bond::FockBasis
    basis_phys::FockBasis
    J_bond::Dict{Int, Vector{Matrix{Float64}}}     # dense J_k per sector (for tests)
    J_phys::Dict{Int, Vector{Matrix{Float64}}}
    J_bond_sp::Dict{Int, Vector{Vector{Tuple{Int,Float64}}}}  # sparse J_k (for recursion)
    J_phys_sp::Dict{Int, Vector{Vector{Tuple{Int,Float64}}}}
    bpz_bond_form::TensorMap                       # ℂ ← V_bond ⊗ V_bond
    m_max::Int                                     # max mode index for Ward recursion
end

"""
    CompactBosonCFT(; R, trunc) -> CompactBosonCFT
    CompactBosonCFT(; R, h_bond, h_phys) -> CompactBosonCFT     # convenience

Build the compact-boson CFT data at radius R with the given truncation.
You may pass either a `trunc::TruncationSpec`, or `h_bond` and `h_phys`
directly. (Julia methods only dispatch on positional args, so we use a
single kwarg method with optional fields rather than two overloads.)
"""
function CompactBosonCFT(; R::Real, c::Real=1.0,
                        trunc::Union{TruncationSpec,Nothing}=nothing,
                        h_bond::Union{Real,Nothing}=nothing,
                        h_phys::Union{Real,Nothing}=nothing)
    if trunc === nothing
        (h_bond === nothing || h_phys === nothing) &&
            error("CompactBosonCFT: provide either `trunc` or both `h_bond` and `h_phys`")
        trunc = TruncationSpec(h_bond=h_bond, h_phys=h_phys)
    end

    R_f = Float64(R)
    basis_bond = build_fock_basis(R_f, trunc.h_bond)
    basis_phys = build_fock_basis(R_f, trunc.h_phys)

    # m_max bound: largest mode index that the Ward recursion can encounter.
    # The Ward identity peels J_{-m} where m is the largest part of any
    # partition in either basis, plus a small buffer.
    m_max = max(floor(Int, trunc.h_bond), floor(Int, trunc.h_phys)) + 2

    J_bond, J_bond_sp = build_J_matrices(basis_bond, m_max)
    J_phys, J_phys_sp = build_J_matrices(basis_phys, m_max)

    bpz_bond_form = build_bpz_form(basis_bond)

    CompactBosonCFT(R_f, Float64(c), trunc, basis_bond, basis_phys,
                    J_bond, J_phys, J_bond_sp, J_phys_sp, bpz_bond_form, m_max)
end
