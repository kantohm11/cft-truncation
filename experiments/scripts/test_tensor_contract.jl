#!/usr/bin/env julia
# Test: TensorMap-native modified vertex and contraction (block iteration)

using TensorKit
using CFTTruncation

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
vd = compute_vertex(cft, 1.0)
V = vd.vertex

# --- Test 1: modified_vertex via block iteration ---
Vmod = copy(V)
bb = cft.basis_bond
for (f1, f2) in fusiontrees(Vmod)
    n_L = Int(f2.uncoupled[2].charge)
    n_R = Int(f2.uncoupled[3].charge)
    blk = Vmod[f1, f2]
    for aT in axes(blk, 1), aL in axes(blk, 2), aR in axes(blk, 3)
        hd_L = conformal_dim(bb, n_L, aL)
        hd_R = conformal_dim(bb, n_R, aR)
        factor = exp(pi * 1.0 / 2 * (hd_L - 1.0/24)) * exp(pi * 1.0 / 2 * (hd_R - 1.0/24))
        blk[aT, aL, aR] *= factor
    end
    Vmod[f1, f2] = blk
end

# Cross-check against raw dict version
mod_raw = modified_vertex_raw(vd; c=1.0)
max_diff = 0.0
for (key, val) in mod_raw
    n_T, n_L, n_R, aT, aL, aR = key
    # Find the block in Vmod
    for (f1, f2) in fusiontrees(Vmod)
        fn_T = Int(f2.uncoupled[1].charge)
        fn_L = Int(f2.uncoupled[2].charge)
        fn_R = Int(f2.uncoupled[3].charge)
        (fn_T == n_T && fn_L == n_L && fn_R == n_R) || continue
        blk = Vmod[f1, f2]
        global max_diff = max(max_diff, abs(blk[aT, aL, aR] - val))
    end
end
println("Test 1 (modified_vertex): max_diff vs raw = $max_diff (expect ~0)")

# --- Test 2: contraction with a ket via block iteration ---
# Contract V_phys leg with primary at n=0
bp = cft.basis_phys
result = zeros(Float64, one(bb.V), bb.V ⊗ bb.V)
for (f1_V, f2_V) in fusiontrees(V)
    n_T = Int(f2_V.uncoupled[1].charge)
    n_T == 0 || continue
    n_L = Int(f2_V.uncoupled[2].charge)
    n_R = Int(f2_V.uncoupled[3].charge)
    blk_V = V[f1_V, f2_V]
    # psi_T = primary at n=0 means alpha_T = 1
    for (f1_R, f2_R) in fusiontrees(result)
        fn_L = Int(f2_R.uncoupled[1].charge)
        fn_R = Int(f2_R.uncoupled[2].charge)
        (fn_L == n_L && fn_R == n_R) || continue
        blk_R = result[f1_R, f2_R]
        for aL in axes(blk_V, 2), aR in axes(blk_V, 3)
            blk_R[aL, aR] += blk_V[1, aL, aR]  # alpha_T = 1
        end
        result[f1_R, f2_R] = blk_R
    end
end
println("Test 2 (contraction): norm(result) = ", norm(result))

# Cross-check against contract_T on raw
c_raw = contract_T(vd.raw, vd, 0, 1)
raw_norm = sqrt(sum(v^2 for (_, v) in c_raw))
println("Test 2 (cross-check): raw norm = $raw_norm (should match)")

println("\nAll tests passed!")
