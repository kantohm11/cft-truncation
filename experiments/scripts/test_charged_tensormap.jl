#!/usr/bin/env julia
# Investigate: can TensorKit represent "charged" TensorMaps?
# i.e., a TensorMap where the total charge in codomain ≠ total in domain?

using TensorKit

V = Vect[U1Irrep](U1Irrep(0) => 2, U1Irrep(1) => 1, U1Irrep(-1) => 1)

println("=== Test 1: standard charge-conserving map V → V ===")
t = zeros(Float64, V, V)
println("  t: codomain=$(codomain(t)), domain=$(domain(t))")
println("  sectors: ", collect(blocksectors(t)))

println("\n=== Test 2: ket |n=1⟩ as TensorMap V ← ℂ ===")
psi = zeros(Float64, V, one(V))
# Set the n=1, α=1 component
for (f1, f2) in fusiontrees(psi)
    n = Int(f1.uncoupled[1].charge)
    println("  fusion tree: codomain charge=$n")
    if n == 1
        blk = psi[f1, f2]
        blk[1, 1] = 1.0
        psi[f1, f2] = blk
    end
end
println("  psi norm = ", norm(psi))

println("\n=== Test 3: can we make a 'charged' map? ===")
# Try: a map from V_bond ⊗ V_bond to some charge-1 sector
# In TensorKit, the codomain could carry a different grading
# Try: make codomain a single irrep space

V1 = Vect[U1Irrep](U1Irrep(1) => 1)  # 1D space at charge 1
println("  V1 = $V1")
try
    t1 = zeros(Float64, V1, V ⊗ V)
    println("  t1: codomain=$(codomain(t1)), domain=$(domain(t1))")
    println("  t1 sectors: ", collect(blocksectors(t1)))
    # This map has charge conservation: charge(codomain) = charge(domain)
    # i.e., 1 = n_1 + n_2.  So it naturally selects n_1+n_2=1 blocks.
    for (f1, f2) in fusiontrees(t1)
        n_cod = Int(f1.uncoupled[1].charge)
        n_d1 = Int(f2.uncoupled[1].charge)
        n_d2 = Int(f2.uncoupled[2].charge)
        println("  block: cod=$n_cod  dom=($n_d1, $n_d2)")
    end
catch e
    println("  ERROR: $e")
end

println("\n=== Test 4: contract vertex (ℂ ← V⊗V⊗V) with ket (V←ℂ) ===")
# Build a toy (0,3) vertex: ℂ ← V ⊗ V ⊗ V
vertex = zeros(Float64, one(V), V ⊗ V ⊗ V)
for (f1, f2) in fusiontrees(vertex)
    blk = vertex[f1, f2]
    fill!(blk, 1.0)
    vertex[f1, f2] = blk
end
println("  vertex norm = ", norm(vertex))

# Try: vertex * (psi ⊗ id(V) ⊗ id(V))
# psi: V ← ℂ
# psi ⊗ id ⊗ id: V ⊗ V ⊗ V ← ℂ ⊗ V ⊗ V
# But ℂ ⊗ V ⊗ V is not the same as V ⊗ V...
println("  Trying vertex * (psi ⊗ id(V) ⊗ id(V))...")
try
    insertion = psi ⊗ id(V) ⊗ id(V)
    println("  insertion: codomain=$(codomain(insertion)), domain=$(domain(insertion))")
    result = vertex * insertion
    println("  result: codomain=$(codomain(result)), domain=$(domain(result))")
    println("  result norm = ", norm(result))
catch e
    println("  ERROR: $e")
end

println("\n=== Test 5: alternative — build result as charged TensorMap ===")
# The result of contracting vertex with |n_T=1⟩ should be:
# a map V1 ← V ⊗ V  where V1 = Vect[U1Irrep](U1Irrep(-1) => 1)
# (charge -1 because vertex has total charge 0, we removed charge +1)
# Actually: vertex has charge_cod = 0, charge_dom = n_T + n_L + n_R = 0.
# Removing n_T = 1 leaves n_L + n_R = -1.
# The result should have codomain charge 0 and domain charge -1? No...
# The result is a LINEAR FUNCTIONAL on (n_L+n_R=-1) states.
# As a TensorMap: it maps V⊗V to ℂ but only on the charge(-1) subspace.
# Equivalently: it's ℂ ← V⊗V but with a "sector shift" of -1.

# In TensorKit: this can be a TensorMap with codomain = V(-1) (1D at charge -1)
# and domain = V ⊗ V. Then charge conservation gives: -1 = n_L + n_R, which
# is exactly the constraint we want.
V_neg1 = Vect[U1Irrep](U1Irrep(-1) => 1)
try
    result = zeros(Float64, V_neg1, V ⊗ V)
    println("  result: codomain=$(codomain(result)), domain=$(domain(result))")
    for (f1, f2) in fusiontrees(result)
        n_cod = Int(f1.uncoupled[1].charge)
        n_d1 = Int(f2.uncoupled[1].charge)
        n_d2 = Int(f2.uncoupled[2].charge)
        println("  block: cod=$n_cod  dom=($n_d1, $n_d2)  sum=$(n_d1+n_d2)")
    end
    println("  -> This works! Codomain charge -1 selects n_L+n_R = -1 blocks.")
catch e
    println("  ERROR: $e")
end

println("\nDone!")
