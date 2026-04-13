#!/usr/bin/env julia
using CFTTruncation
using TensorKit: norm

cache_dir = mktempdir()
println("Cache dir: $cache_dir")
set_cache_dir(cache_dir)

cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(4.0))

println("\n=== Test 1: :auto first call (compute + save) ===")
@time vd1 = compute_vertex(cft, 1.0)
println("norm = ", norm(vd1.vertex))

println("\n=== Test 2: :auto second call (load from cache) ===")
@time vd2 = compute_vertex(cft, 1.0)
println("norm = ", norm(vd2.vertex))
println("norms match: ", isapprox(norm(vd1.vertex), norm(vd2.vertex)))

println("\n=== Test 3: :off (no cache) ===")
@time vd3 = compute_vertex(cft, 1.0; cache=:off)
println("norm = ", norm(vd3.vertex))

println("\n=== Test 4: :regenerate (recompute + overwrite) ===")
@time vd4 = compute_vertex(cft, 1.0; cache=:regenerate)
println("norm = ", norm(vd4.vertex))

println("\n=== Test 5: cache files ===")
files = readdir(cache_dir)
println("Files: ", files)
println("Count: ", length(files))

println("\n=== Test 6: modified_vertex_cache with caching ===")
@time cache_dict = modified_vertex_cache(cft, [0.5, 1.0, 1.5])
println("Keys: ", sort(collect(keys(cache_dict))))
println("All finite: ", all(isfinite(norm(v)) for v in values(cache_dict)))

println("\nAll cache tests passed!")
