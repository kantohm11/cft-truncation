using Test
using CFTTruncation
using CFTTruncation: _cache_path
using TensorKit: norm

@testset "Cache (T-shape and cross)" begin

    cache_dir = mktempdir()
    set_cache_dir(cache_dir)

    cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(1.0))
    ell = 1.0

    @testset "T-shape: compute → cache load round-trip" begin
        # First call: compute and save.
        vd_fresh = compute_vertex(cft, ell; shape=:T, series_order=15, cache=:regenerate)
        @test isfile(_cache_path(cft, ell, 15; shape=:T))
        # Second call: must load from disk and match.
        vd_loaded = compute_vertex(cft, ell; shape=:T, series_order=15, cache=:auto)
        @test norm(vd_loaded.vertex) ≈ norm(vd_fresh.vertex)
    end

    @testset "Cross: compute → cache load round-trip" begin
        vd_fresh = compute_vertex(cft, ell; shape=:cross, series_order=15, cache=:regenerate)
        @test isfile(_cache_path(cft, ell, 15; shape=:cross))
        vd_loaded = compute_vertex(cft, ell; shape=:cross, series_order=15, cache=:auto)
        @test norm(vd_loaded.vertex) ≈ norm(vd_fresh.vertex)
    end

    @testset "T-shape and cross caches coexist (different files)" begin
        path_T = _cache_path(cft, ell, 15; shape=:T)
        path_cross = _cache_path(cft, ell, 15; shape=:cross)
        @test path_T != path_cross
        @test occursin("_T_", path_T)
        @test occursin("_cross_", path_cross)
    end

    @testset ":off skips disk IO" begin
        # Build a fresh path that doesn't exist.
        vd_off = compute_vertex(cft, 0.5; shape=:cross, series_order=15, cache=:off)
        @test !isfile(_cache_path(cft, 0.5, 15; shape=:cross))
    end

end
