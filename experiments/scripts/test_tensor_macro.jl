#!/usr/bin/env julia
using TensorKit
using CFTTruncation

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(4.0))
vd = compute_vertex(cft, 1.0)
V = vd.vertex
bb = cft.basis_bond
D = build_propagator_factor(bb, vd.ell, 1.0)

println("Original V:")
println("  codomain = $(codomain(V))")
println("  domain   = $(domain(V))")

# @tensor gives (3,0) with V' legs in codomain
@tensor Vm_raw[-1 -2 -3] := V[-1 1 2] * D[1; -2] * D[2; -3]
println("\n@tensor result (3,0):")
println("  codomain = $(codomain(Vm_raw))")
println("  domain   = $(domain(Vm_raw))")

# Permute (3,0) → (0,3): codomain V' → domain V'' = V
Vm = permute(Vm_raw, ((), (1, 2, 3)))
println("\nAfter permute (0,3):")
println("  codomain = $(codomain(Vm))")
println("  domain   = $(domain(Vm))")

# Compare with block-iteration
Vm_block = modified_vertex(vd; c=1.0)
println("\nBlock iteration:")
println("  codomain = $(codomain(Vm_block))")
println("  domain   = $(domain(Vm_block))")

# Same spaces?
println("\nSame spaces? ", codomain(Vm) == codomain(Vm_block) && domain(Vm) == domain(Vm_block))

# Same values?
if codomain(Vm) == codomain(Vm_block) && domain(Vm) == domain(Vm_block)
    println("norm diff = ", norm(Vm - Vm_block))
else
    println("Spaces still differ!")
    println("norm(@tensor) = ", norm(Vm))
    println("norm(block)   = ", norm(Vm_block))
end

# Timing at h_max=4
println("\nTiming @tensor + permute:")
@time begin
    @tensor tmp[-1 -2 -3] := V[-1 1 2] * D[1; -2] * D[2; -3]
    permute(tmp, ((), (1, 2, 3)))
end
println("Timing block iteration:")
@time modified_vertex(vd; c=1.0)
