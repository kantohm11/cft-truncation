#!/usr/bin/env julia
# Test: use ncon to apply D to specific legs of the vertex.

using TensorKit
using CFTTruncation

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(6.0))
vd = compute_vertex(cft, 1.0)
V = vd.vertex
bb = cft.basis_bond; bp = cft.basis_phys

D = build_propagator_factor(bb, vd.ell, 1.0)

# ncon convention: positive indices = contracted, negative = free.
# V has 0 codomain legs and 3 domain legs.
# D has 1 codomain leg and 1 domain leg.
#
# We want: Vm[;a,b,c] = V[;a,b',c'] * D[b';b] * D[c';c]
#
# In ncon: tensors = [V, D, D], and index lists specify contractions.
# V: domain legs get indices [-1, 1, 2] (a=free, b'=contracted, c'=contracted)
# D1: codomain=1 (contracts with V's b'), domain=-2 (free, new b)
# D2: codomain=2 (contracts with V's c'), domain=-3 (free, new c)
#
# But ncon needs to know codomain vs domain split. Let me check the syntax.

println("V: N_cod=$(length(codomain(V).spaces)), N_dom=$(length(domain(V).spaces))")
println("D: N_cod=$(length(codomain(D).spaces)), N_dom=$(length(domain(D).spaces))")

# ncon syntax: ncon([T1, T2, ...], [idx1, idx2, ...])
# where idx_i lists the indices of T_i (positive=contract, negative=free)
# For TensorKit: indices include codomain and domain.
# T with N_cod=0, N_dom=3: indices = [a, b, c] (all domain)
# T with N_cod=1, N_dom=1: indices = [x, y] (codomain, domain)

println("\nTrying ncon...")
@time Vm_ncon = ncon([V, D, D],
                     [[-1, 1, 2],    # V: domain legs (a=free, b'=contract, c'=contract)
                      [1, -2],       # D1: [codomain=contract with b', domain=free new b]
                      [2, -3]])      # D2: [codomain=contract with c', domain=free new c]

println("Vm_ncon type: ", typeof(Vm_ncon))
println("Vm_ncon: $(codomain(Vm_ncon)) ← $(domain(Vm_ncon))")
println("norm(Vm_ncon) = ", norm(Vm_ncon))

# Compare with block-iteration
println("\nBlock iteration:")
@time Vm_block = modified_vertex(vd; c=1.0)
println("norm(Vm_block) = ", norm(Vm_block))

# Can't subtract directly (different codomain/domain split) — compare norms
println("\nnorm match: ", isapprox(norm(Vm_ncon), norm(Vm_block); rtol=1e-12))
