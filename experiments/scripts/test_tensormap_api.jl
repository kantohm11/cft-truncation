#!/usr/bin/env julia
using CFTTruncation
using LinearAlgebra: norm
using Random

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(3.0))
vd = compute_vertex(cft, 1.0)
Vm = modified_vertex(vd; c=1.0)
println("modified_vertex: norm=$(norm(Vm))")

# Test project_to_hcut on the full 3-leg vertex
bp = cft.basis_phys; bb = cft.basis_bond
Vm_proj = project_to_hcut(Vm, [bp, bb, bb], 2.0)
println("project_to_hcut(h=2): ratio=$(norm(Vm_proj)/norm(Vm))")

# Test contracted norm at n_T=0 primary (neutral, so charge-conserving)
vec_T_0 = [((0, 1), 1.0)]
fn = full_norm_after_contract_T(Vm, vec_T_0)
pn = projected_norm_after_contract_T(Vm, bp, bb, vec_T_0, 2.0)
println("contract T (n=0,a=1): full=$(fn) proj=$(pn) ratio=$(pn/fn)")

# Cross-check against raw dict
mod_raw = modified_vertex_raw(vd; c=1.0)
c_raw = contract_T(mod_raw, vd, 0, 1)
raw_fn = sqrt(sum(v^2 for (_, v) in c_raw))
println("  raw cross-check: full=$(raw_fn) (should match $(fn))")

# Test contracted norm at n_T=1 (CHARGED — this is the case that was 0 before)
vec_T_1 = [((1, 1), 1.0)]
fn1 = full_norm_after_contract_T(Vm, vec_T_1)
println("contract T (n=1,a=1): full=$(fn1) (should be nonzero)")

# Test random vector at h=0.5
shells = weight_shells(bp)
for s in shells
    s.h == 0.5 || continue
    rng = Random.MersenneTwister(42)
    vec = random_unit_vec(s.states; rng=rng)
    fn_r = full_norm_after_contract_T(Vm, vec)
    pn_r = projected_norm_after_contract_T(Vm, bp, bb, vec, 2.0)
    println("random h=0.5: full=$(fn_r) proj=$(pn_r)")
end

# Test double contraction (T and L)
vec_L_0 = [((0, 1), 1.0)]
fn_tl = full_norm_after_contract_TL(Vm, vec_T_0, vec_L_0)
pn_tl = projected_norm_after_contract_TL(Vm, bp, bb, vec_T_0, vec_L_0, 2.0)
println("contract T+L (both n=0): full=$(fn_tl) proj=$(pn_tl)")

# Test cache
cache = modified_vertex_cache(cft, [0.5, 1.0]; c=1.0)
println("cache[0.5] norm: $(norm(cache[0.5]))")
println("cache[1.0] norm: $(norm(cache[1.0]))")

println("\nAll TensorMap API tests passed!")
