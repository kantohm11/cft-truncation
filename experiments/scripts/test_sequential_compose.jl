#!/usr/bin/env julia
# Test: apply D to individual legs via permute+compose.

using TensorKit
using CFTTruncation

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(6.0))
vd = compute_vertex(cft, 1.0)
V = vd.vertex
bb = cft.basis_bond

# Dense D (endomorphism V_bond → V_bond)
D = build_propagator_factor(bb, vd.ell, 1.0)

# D' (adjoint) should act on V_bond'
println("D:  $(codomain(D)) ← $(domain(D))")
println("D': $(codomain(D')) ← $(domain(D'))")

# V is (0,3): ℂ ← V_phys ⊗ V_bond ⊗ V_bond
# Expose leg 2 as codomain:
println("\nStep 1: permute(V, ((2,), (1,3)))")
@time V1 = permute(V, ((2,), (1, 3)))
println("  V1: $(codomain(V1)) ← $(domain(V1))")

# Compose D' (V_bond' → V_bond') with V1 (V_bond' ← V_phys ⊗ V_bond)
println("\nStep 2: D' * V1")
@time V2 = D' * V1
println("  V2: $(codomain(V2)) ← $(domain(V2))")

# Now expose leg 3 (the remaining V_bond in domain position 2)
println("\nStep 3: permute(V2, ((2,), (1,)))")
@time V3 = permute(V2, ((2,), (1,)))
println("  V3: $(codomain(V3)) ← $(domain(V3))")

println("\nStep 4: D' * V3")
@time V4 = D' * V3
println("  V4: $(codomain(V4)) ← $(domain(V4))")

# Permute back to (0,3) with original leg order
println("\nStep 5: permute back to (0,3)")
@time Vm_compose = permute(V4, ((), (2, 1)))
# This gives ℂ ← V_phys ⊗ V_bond'... hmm, legs are dualized

println("  Vm: $(codomain(Vm_compose)) ← $(domain(Vm_compose))")

# Compare with block-iteration
println("\nBlock iteration:")
@time Vm_block = modified_vertex(vd; c=1.0)
println("  norm(block) = ", norm(Vm_block))
println("  norm(compose) = ", norm(Vm_compose))
