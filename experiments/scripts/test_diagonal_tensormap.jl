#!/usr/bin/env julia
using TensorKit

V = Vect[U1Irrep](U1Irrep(0) => 3, U1Irrep(1) => 2)

# Check DiagonalTensorMap constructors
println("Methods for DiagonalTensorMap:")
for m in methods(TensorKit.DiagonalTensorMap)
    println("  ", m)
end

# Try SVD to get a DiagonalTensorMap
t = randn(Float64, V, V)
U, S, Vd = svd_compact(t)
println("\nsvd_compact S type: ", typeof(S))

# Try constructing directly
println("\nTrying direct construction...")
try
    D = TensorKit.DiagonalTensorMap(undef, Float64, V)
    println("  DiagonalTensorMap(undef, Float64, V) → ", typeof(D))
    println("  codomain=$(codomain(D)), domain=$(domain(D))")
    # Fill it
    for (f1, f2) in fusiontrees(D)
        blk = D[f1, f2]
        println("  block type: ", typeof(blk), " size: ", size(blk))
    end
catch e
    println("  Failed: $e")
end

# Try another construction form
try
    D = TensorKit.DiagonalTensorMap(ones(Float64, dim(V)), V)
    println("  DiagonalTensorMap(vec, V) → ", typeof(D))
catch e
    println("  Failed: $e")
end

# Check if ⊗ preserves DiagonalTensorMap
if S isa TensorKit.DiagonalTensorMap
    println("\nS is DiagonalTensorMap!")
    println("Testing ⊗...")
    prod = id(V) ⊗ S ⊗ S
    println("  id⊗S⊗S type: ", typeof(prod))

    # Test composition with a regular TensorMap
    vtx = randn(Float64, one(V), V ⊗ V ⊗ V)
    insertion = id(V) ⊗ S ⊗ S
    println("  vertex * insertion...")
    result = vtx * insertion
    println("  result type: ", typeof(result))
end
