module CFTTruncation

using TensorKit
using TensorKit: @tensor
using LinearAlgebra

include("TruncLaurent.jl")
include("SCMap.jl")
include("LocalCoordinates.jl")
include("NeumannCoefficients.jl")
include("FockSpace.jl")
include("JMatrices.jl")
include("BPZ.jl")
include("PrimaryVertex.jl")
include("CompactBoson.jl")
include("Cache.jl")
include("Recursion.jl")

# Public API: enough for `using CFTTruncation` to give a notebook
# everything it needs to build a CFT and compute vertices.
# Lower-level types (Geometry, NeumannData, FockBasis, TruncLaurent, ...)
# remain reachable via the qualified `CFTTruncation.X` form.
export TruncationSpec, CompactBosonCFT
export compute_vertex, vertex_sweep, charge_block, VertexData, VertexDataCross
export set_cache_dir
export conformal_dim
export modified_vertex, modified_vertex_cache, build_propagator_factor
export project_to_hcut
export projected_norm_after_contract_T, projected_norm_after_contract_TL
export full_norm_after_contract_T, full_norm_after_contract_TL

end
