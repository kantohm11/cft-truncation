#!/usr/bin/env julia
# Test: can we express modified_vertex as pure TensorKit composition?
# Vm = vertex * (id_phys ⊗ D ⊗ D)

using TensorKit
using CFTTruncation
using LinearAlgebra: norm

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(3.0))
vd = compute_vertex(cft, 1.0)
V = vd.vertex
bp = cft.basis_phys; bb = cft.basis_bond

# Build propagator D (diagonal, V_bond → V_bond)
D = build_propagator_factor(bb, vd.ell, 1.0)
println("D: $(codomain(D)) ← $(domain(D))")

# Compose: Vm = V * (id ⊗ D ⊗ D)
insertion = id(bp.V) ⊗ D ⊗ D
println("insertion: $(codomain(insertion)) ← $(domain(insertion))")

Vm_compose = V * insertion
println("Vm_compose: $(codomain(Vm_compose)) ← $(domain(Vm_compose))")

# Compare with block-iteration modified_vertex
Vm_block = modified_vertex(vd; c=1.0)
println("\nnorm(compose) = ", norm(Vm_compose))
println("norm(block)   = ", norm(Vm_block))
println("max diff      = ", norm(Vm_compose - Vm_block))

# --- Now test contraction via composition ---
# Contract T leg with charge-1 selector:
V_reshaped = permute(Vm_compose, ((1,), (2, 3)))
sel = build_selector(bp, 1, 1)
contracted = sel * V_reshaped
println("\ncontracted(n=1,a=1): $(codomain(contracted)) ← $(domain(contracted))")
println("norm = ", norm(contracted))

# Cross-check
fn = full_norm_after_contract_T(Vm_compose, [((1, 1), 1.0)])
println("full_norm_after_contract_T cross-check: ", fn)

# --- project_to_hcut on contracted ---
proj = project_to_hcut(contracted, [bb, bb], 2.0)
println("\nprojected(h_cut=2): norm = ", norm(proj))
println("ratio = ", norm(proj) / norm(contracted))

println("\nAll composition tests passed!")
