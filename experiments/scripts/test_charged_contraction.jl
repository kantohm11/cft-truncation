#!/usr/bin/env julia
# Test: charged contraction via permute + compose with a charge selector

using TensorKit
using CFTTruncation
using LinearAlgebra: norm

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(3.0))
vd = compute_vertex(cft, 1.0)
V = vd.vertex
bp = cft.basis_phys; bb = cft.basis_bond

println("V: codomain=$(codomain(V)), domain=$(domain(V))")

# Step 1: permute V to expose V_phys as codomain
# V is (0,3): ℂ ← V_phys ⊗ V_bond ⊗ V_bond
# After permute: V_phys ← V_bond ⊗ V_bond
V_reshaped = permute(V, ((1,), (2, 3)))
println("V_reshaped: codomain=$(codomain(V_reshaped)), domain=$(domain(V_reshaped))")

# Step 2: permute dualizes the leg: codomain(V_reshaped) = V_phys' (dual).
# Build selector as V_{-n} ← V_phys' (the dual sector charge -n in V_phys'
# corresponds to charge n in V_phys).
function build_selector(basis, n, alpha)
    V_neg_n = Vect[U1Irrep](U1Irrep(-n) => 1)
    sel = zeros(Float64, V_neg_n, basis.V')  # note: V' (dual space)
    for (f1, f2) in fusiontrees(sel)
        # domain is V_phys', sector charge = -n (dual of charge n in V_phys)
        fn = Int(f2.uncoupled[1].charge)
        fn == -n || continue
        blk = sel[f1, f2]
        alpha <= size(blk, 2) || continue
        blk[1, alpha] = 1.0
        sel[f1, f2] = blk
    end
    sel
end

# Test charged contraction: n_T = 1, alpha = 1
sel1 = build_selector(bp, 1, 1)
println("\nselector(n=1): codomain=$(codomain(sel1)), domain=$(domain(sel1))")
result1 = sel1 * V_reshaped  # V_n ← V_bond²
println("result(n=1): codomain=$(codomain(result1)), domain=$(domain(result1))")
println("result(n=1) norm = ", norm(result1))

# Cross-check vs raw dict
c1_raw = contract_T(vd.raw, vd, 1, 1)
println("raw cross-check: norm = ", sqrt(sum(v^2 for (_, v) in c1_raw)))

# Test neutral contraction: n_T = 0, alpha = 1
sel0 = build_selector(bp, 0, 1)
result0 = sel0 * V_reshaped
println("\nresult(n=0) norm = ", norm(result0))
c0_raw = contract_T(vd.raw, vd, 0, 1)
println("raw cross-check: norm = ", sqrt(sum(v^2 for (_, v) in c0_raw)))

# Test n_T = -1
sel_m1 = build_selector(bp, -1, 1)
result_m1 = sel_m1 * V_reshaped
println("\nresult(n=-1) norm = ", norm(result_m1))
cm1_raw = contract_T(vd.raw, vd, -1, 1)
println("raw cross-check: norm = ", sqrt(sum(v^2 for (_, v) in cm1_raw)))

println("\nCharged contraction via permute+selector works!")
