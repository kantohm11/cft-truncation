"""
Compact-boson CFT data: a bundle of the truncated state spaces, mode-action
matrices, and BPZ form needed to compute T-vertices for a fixed CFT and
truncation. ℓ-independent — build once and reuse across an ℓ-sweep.
"""

"""
    TruncationSpec(bond_cutoffs::Dict{Int,Int}, phys_cutoffs::Dict{Int,Int})
    TruncationSpec(h_bond::Real, h_phys::Real; R = 1.0)
    TruncationSpec(h::Real; R = 1.0)
    TruncationSpec(; h_bond, h_phys, R = 1.0)
    TruncationSpec(; bond_cutoffs, phys_cutoffs)

Per-charge descendant-level cutoffs for the bond (L/R arms) and
physical (T or T/B arms) state spaces. `bond_cutoffs[n] = N_max`
means: include charge `n` with partitions of total level `≤ N_max`.

Backward-compat constructors accept a uniform conformal-weight cap
`h_max` and convert via `uniform_cutoffs(h_max, R)` (per-charge cap
`floor(h_max − n²/(2R²))`). For backward-readability, `h_bond` and
`h_phys` are kept as fields and set to the maximum effective h
included in each cutoffs dict.
"""
struct TruncationSpec
    bond_cutoffs::Dict{Int, Int}
    phys_cutoffs::Dict{Int, Int}
    h_bond::Float64        # effective: max h included in bond
    h_phys::Float64        # effective: max h included in phys
end

function TruncationSpec(bond_cutoffs::Dict{Int,Int}, phys_cutoffs::Dict{Int,Int};
                        R::Real = 1.0)
    R_f = Float64(R)
    h_eff(c) = isempty(c) ? 0.0 :
        maximum((n/R_f)^2/2 + c[n] for n in keys(c))
    TruncationSpec(bond_cutoffs, phys_cutoffs, h_eff(bond_cutoffs), h_eff(phys_cutoffs))
end

# Uniform-h backward-compat constructors.
function TruncationSpec(h_bond::Real, h_phys::Real; R::Real = 1.0)
    bc = uniform_cutoffs(h_bond, R)
    pc = uniform_cutoffs(h_phys, R)
    TruncationSpec(bc, pc; R = R)
end

TruncationSpec(h::Real; R::Real = 1.0) = TruncationSpec(h, h; R = R)

TruncationSpec(; h_bond::Union{Real,Nothing} = nothing,
               h_phys::Union{Real,Nothing} = nothing,
               bond_cutoffs::Union{Dict{Int,Int},Nothing} = nothing,
               phys_cutoffs::Union{Dict{Int,Int},Nothing} = nothing,
               R::Real = 1.0) = begin
    if bond_cutoffs !== nothing && phys_cutoffs !== nothing
        TruncationSpec(bond_cutoffs, phys_cutoffs; R = R)
    elseif h_bond !== nothing && h_phys !== nothing
        TruncationSpec(h_bond, h_phys; R = R)
    else
        error("TruncationSpec: provide either (bond_cutoffs, phys_cutoffs) or (h_bond, h_phys).")
    end
end

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
        trunc = TruncationSpec(h_bond, h_phys; R = R)
    end

    R_f = Float64(R)
    basis_bond = build_fock_basis(R_f, trunc.bond_cutoffs)
    basis_phys = build_fock_basis(R_f, trunc.phys_cutoffs)

    # m_max bound: largest mode index that the Ward recursion can encounter.
    # The Ward identity peels J_{-m} where m is the largest part of any
    # partition in either basis, plus a small buffer.
    max_level = max(maximum(values(trunc.bond_cutoffs); init=0),
                    maximum(values(trunc.phys_cutoffs); init=0))
    m_max = max_level + 2

    J_bond, J_bond_sp = build_J_matrices(basis_bond, m_max)
    J_phys, J_phys_sp = build_J_matrices(basis_phys, m_max)

    bpz_bond_form = build_bpz_form(basis_bond)

    CompactBosonCFT(R_f, Float64(c), trunc, basis_bond, basis_phys,
                    J_bond, J_phys, J_bond_sp, J_phys_sp, bpz_bond_form, m_max)
end
