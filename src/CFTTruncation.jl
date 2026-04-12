module CFTTruncation

using TensorKit
using LinearAlgebra
using SparseArrays

include("TruncLaurent.jl")
include("SCMap.jl")
include("LocalCoordinates.jl")
include("NeumannCoefficients.jl")
include("FockSpace.jl")
include("JMatrices.jl")
include("BPZ.jl")
include("PrimaryVertex.jl")
include("CompactBoson.jl")
include("Recursion.jl")

# Public API: enough for `using CFTTruncation` to give a notebook
# everything it needs to build a CFT and compute vertices.
# Lower-level types (Geometry, NeumannData, FockBasis, TruncLaurent, ...)
# remain reachable via the qualified `CFTTruncation.X` form.
export TruncationSpec, CompactBosonCFT
export compute_vertex, vertex_sweep, charge_block, VertexData

end
