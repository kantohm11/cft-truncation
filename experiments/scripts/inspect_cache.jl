#!/usr/bin/env julia
# List vertex-cache files in a directory and decode each truncation hash
# via its JSON sidecar. Useful for auditing what's cached on disk.
#
# Usage:
#   julia --project=. experiments/scripts/inspect_cache.jl <cache_dir>

using JSON
using Printf

if length(ARGS) != 1
    println(stderr, "usage: julia inspect_cache.jl <cache_dir>")
    exit(1)
end

cache_dir = ARGS[1]
isdir(cache_dir) || error("not a directory: $cache_dir")

# Group cache files by their (cache_version, shape, R, so, trunc_hash) stem.
function parse_filename(name::AbstractString)
    # Pattern: vertex_<cache_version>_<shape>_R<R>_so<so>_t<hash>_ell<ell>.jld2
    m = match(r"^vertex_(?<v>v\d+(?:_[\w]+)*?)_(?<shape>T|cross)_R(?<R>[\d.-]+)_so(?<so>\d+)_t(?<th>[0-9a-f]{8})_ell(?<ell>[\d.+-]+)\.jld2$", name)
    m === nothing && return nothing
    (; cache_version = String(m[:v]),
       shape = String(m[:shape]),
       R = parse(Float64, m[:R]),
       so = parse(Int, m[:so]),
       trunc_hash = String(m[:th]),
       ell = parse(Float64, m[:ell]))
end

function sidecar_filename(meta)
    "vertex_$(meta.cache_version)_$(meta.shape)_R$(meta.R)_so$(meta.so)_t$(meta.trunc_hash).json"
end

vertex_files = filter(f -> endswith(f, ".jld2"), readdir(cache_dir))
sidecar_files = filter(f -> endswith(f, ".json"), readdir(cache_dir))

# Group cache files by stem.
groups = Dict{String, Vector}()
metas  = Dict{String, Any}()
sizes  = Dict{String, Int}()
for f in vertex_files
    meta = parse_filename(f)
    meta === nothing && continue
    sc = sidecar_filename(meta)
    push!(get!(groups, sc, String[]), f)
    metas[sc] = meta
    sizes[sc] = get(sizes, sc, 0) + filesize(joinpath(cache_dir, f))
end

println("=== Cache audit: $cache_dir ===")
println("Files: $(length(vertex_files)) .jld2, $(length(sidecar_files)) sidecars")
println()

if isempty(groups)
    println("(no recognised cache files)")
    exit(0)
end

# For each group, print the spec from sidecar (or a warning if missing).
for (sc, files) in sort(collect(groups); by = first)
    meta = metas[sc]
    sc_path = joinpath(cache_dir, sc)
    println("─" ^ 78)
    println(@sprintf("[%s] %s, R=%g, so=%d, hash=%s",
        meta.cache_version, meta.shape, meta.R, meta.so, meta.trunc_hash))
    println(@sprintf("  files: %d ℓ values, total %.1f MB",
        length(files), sizes[sc] / 1e6))
    print("  ℓs: [")
    for (i, f) in enumerate(sort(files))
        m = parse_filename(f)
        i > 1 && print(", ")
        print(m === nothing ? "?" : m.ell)
    end
    println("]")

    if isfile(sc_path)
        sd = JSON.parsefile(sc_path)
        bond = sd["trunc"]["bond_cutoffs"]
        phys = sd["trunc"]["phys_cutoffs"]
        h_b = sd["trunc"]["h_bond_eff"]
        h_p = sd["trunc"]["h_phys_eff"]
        println(@sprintf("  bond cutoffs (h_eff=%g): %s",
            h_b, sort(["$n => $v" for (n, v) in bond])))
        println(@sprintf("  phys cutoffs (h_eff=%g): %s",
            h_p, sort(["$n => $v" for (n, v) in phys])))
    else
        println("  (sidecar missing; spec is unrecoverable)")
    end
end

# Orphan sidecars: sidecar files whose stem has no .jld2 files.
orphan_sidecars = setdiff(sidecar_files, keys(groups))
if !isempty(orphan_sidecars)
    println("─" ^ 78)
    println("Orphan sidecars (no matching .jld2):")
    for f in orphan_sidecars
        println("  $f")
    end
end
