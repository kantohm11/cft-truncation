#!/usr/bin/env julia
# Check the Virasoro-primary Jacobian convention by reading it off from
# `V(J_{-1}|0⟩_L, J_{-1}|0⟩_R, |0⟩_T)` (and the two cyclic permutations).
#
# Argument:
#  • J(z) is a Virasoro primary of weight h = 1, lying in the u(1) vacuum module.
#  • The recursion computes V(J·,J·,vac) using only Ward/Neumann coefficients
#    plus `primary_vertex(0,0,0) = 1`. The `|α|^{2h}` formula is dormant at h=0.
#  • CFT prediction:
#       V(J(ξ_i=0) J(ξ_j=0)) = (Jac_i)·(Jac_j) · ⟨J(x_i) J(x_j)⟩_UHP-Cardy
#                            = (Jac_i)(Jac_j) · c_J / (x_i − x_j)²
#  • If we ratio out the boundary 2-pt coefficient `c_J` and the geometric factor
#    `1/(x_i − x_j)²`, the remainder reads off the Jacobian.
#
# By Virasoro-primary universality, the same rule (with h replaced) applies to
# any primary, including charged primaries `V_n` — so this pins down the
# convention `primary_vertex` should be using.

using CFTTruncation
using TensorKit: fusiontrees
using LinearAlgebra: I
using Printf

const R_VAL  = 1.0
const H_BOND = 6.0
const H_PHYS = 3.0

function charge_block_T(vd::CFTTruncation.VertexData, n_L::Int, n_R::Int, n_T::Int)
    V = vd.vertex
    for (f1, f2) in fusiontrees(V)
        Int(f2.uncoupled[1].charge) == n_T || continue
        Int(f2.uncoupled[2].charge) == n_L || continue
        Int(f2.uncoupled[3].charge) == n_R || continue
        return permutedims(Array(V[f1, f2]), (2, 3, 1))   # [αL, αR, αT]
    end
    return zeros(Float64, 0, 0, 0)
end

function partition_index(basis::CFTTruncation.FockBasis, n::Int, λ::Vector{Int})
    # Find the index `α` of partition λ in basis.states[n].
    for (i, λi) in enumerate(basis.states[n])
        λi == λ && return i
    end
    error("Partition $λ not in basis at charge $n")
end

function probe(ell::Float64)
    cft = CompactBosonCFT(R = R_VAL, h_bond = H_BOND, h_phys = H_PHYS)
    vd  = compute_vertex(cft, ell; cache = :off, series_order = 20)
    geom = vd.geom

    αL = abs(geom.arms.L.α)
    αR = abs(geom.arms.R.α)
    αT = abs(geom.arms.T.α)

    # Indices: vacuum |∅⟩ has partition [] (level 0); J_{-1}|0⟩ has partition [1].
    α_vac_b = partition_index(cft.basis_bond, 0, Int[])
    α_J_b   = partition_index(cft.basis_bond, 0, [1])
    α_vac_p = partition_index(cft.basis_phys, 0, Int[])
    α_J_p   = partition_index(cft.basis_phys, 0, [1])

    blk = charge_block_T(vd, 0, 0, 0)

    V_JJV = blk[α_J_b,   α_J_b,   α_vac_p]   # V(J|0⟩_L, J|0⟩_R, |0⟩_T)
    V_JVJ = blk[α_J_b,   α_vac_b, α_J_p]     # V(J|0⟩_L, |0⟩_R, J|0⟩_T)
    V_VJJ = blk[α_vac_b, α_J_b,   α_J_p]     # V(|0⟩_L, J|0⟩_R, J|0⟩_T)

    (; ell, αL, αR, αT, V_JJV, V_JVJ, V_VJJ)
end

# Geometric factors:
# x_L = -1, x_R = +1, x_T = 0  =>  (x_L-x_R)² = 4, (x_L-x_T)² = 1, (x_R-x_T)² = 1.
#
# Predicted:
#   V_JJV = c_J · Jac_L · Jac_R · 1/4
#   V_JVJ = c_J · Jac_L · Jac_T · 1
#   V_VJJ = c_J · Jac_R · Jac_T · 1
# under hypothesis Jac_i = (1/α_i)^1 = 1/α_i:
#   V_JJV · (α_L α_R)     = c_J / 4  (constant in ℓ)
#   V_JVJ · (α_L α_T)     = c_J      (constant in ℓ)
#   V_VJJ · (α_R α_T)     = c_J      (constant in ℓ)
# under WRONG hypothesis Jac_i = α_i (positive exponent):
#   V_JJV / (α_L α_R)     = c_J / 4  (constant)
#   etc.  Either is testable; only one will be ℓ-independent.

ells = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
results = [probe(ell) for ell in ells]

println("=== Reading off the J-current Jacobian convention ===")
println("h_bond = $(H_BOND), h_phys = $(H_PHYS), R = $(R_VAL)")
println()
println(@sprintf("%6s | %10s %10s %10s | %12s %12s %12s",
        "ℓ", "|α_L|", "|α_R|", "|α_T|", "V_JJV", "V_JVJ", "V_VJJ"))
println("-"^120)
for r in results
    println(@sprintf("%6.2f | %10.5f %10.5f %10.5f | %12.5e %12.5e %12.5e",
        r.ell, r.αL, r.αR, r.αT, r.V_JJV, r.V_JVJ, r.V_VJJ))
end

println("\n--- Hypothesis A: Jac = (1/α)^h  (so V·αα should be constant) ---")
println(@sprintf("%6s | %18s %18s %18s",
        "ℓ", "V_JJV·αLαR", "V_JVJ·αLαT", "V_VJJ·αRαT"))
println("-"^80)
for r in results
    p1 = r.V_JJV * r.αL * r.αR
    p2 = r.V_JVJ * r.αL * r.αT
    p3 = r.V_VJJ * r.αR * r.αT
    println(@sprintf("%6.2f | %18.10e %18.10e %18.10e", r.ell, p1, p2, p3))
end

println("\n--- Hypothesis B: Jac = α^h  (so V/(αα) should be constant) ---")
println(@sprintf("%6s | %18s %18s %18s",
        "ℓ", "V_JJV/(αLαR)", "V_JVJ/(αLαT)", "V_VJJ/(αRαT)"))
println("-"^80)
for r in results
    p1 = r.V_JJV / (r.αL * r.αR)
    p2 = r.V_JVJ / (r.αL * r.αT)
    p3 = r.V_VJJ / (r.αR * r.αT)
    println(@sprintf("%6.2f | %18.10e %18.10e %18.10e", r.ell, p1, p2, p3))
end

println("\nIf hypothesis A's columns are flat in ℓ AND in ratio 1/4 : 1 : 1,")
println("the Jacobian rule is (1/α)^h (sign opposite to primary_vertex's |α|^{+2h}).")
