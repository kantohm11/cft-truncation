using Test
using CFTTruncation

@testset "CFTTruncation" begin
    include("test_trunclaurent.jl")
    include("test_scmap.jl")
    include("test_scmap_cross.jl")
    include("test_localcoordinates.jl")
    include("test_neumann.jl")
    include("test_fockspace.jl")
    include("test_jmatrices.jl")
    include("test_bpz.jl")
    include("test_primaryvertex.jl")
    include("test_compact_boson.jl")
    include("test_recursion.jl")
    # include("test_integration.jl")
end
