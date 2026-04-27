using Test
using CFTTruncation
using CFTTruncation: _cache_path, _sidecar_path, truncation_hash, write_sidecar,
                     truncation_spec_dict, uniform_cutoffs
using TensorKit: norm
using JSON

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

    @testset "Cache filename embeds truncation hash" begin
        # The hash should appear as `_t<8 hex chars>` in the filename.
        path = _cache_path(cft, ell, 15; shape=:T)
        m = match(r"_t([0-9a-f]{8})_", basename(path))
        @test m !== nothing
        # Hash should match what truncation_hash returns directly.
        @test m.captures[1] == truncation_hash(cft.trunc)
    end

    @testset "Sidecar written on save and decodes back to spec" begin
        # The T-shape compute step above should have produced a sidecar.
        sc_path = _sidecar_path(cft, 15; shape=:T)
        @test isfile(sc_path)
        sc = JSON.parsefile(sc_path)
        @test sc["cache_version"] == CFTTruncation.CACHE_VERSION
        @test sc["R"] == cft.R
        @test sc["shape"] == "T"
        @test sc["trunc_hash"] == truncation_hash(cft.trunc)
        # The trunc dict should round-trip the cutoffs.
        bond = sc["trunc"]["bond_cutoffs"]
        for (n, v) in cft.trunc.bond_cutoffs
            @test bond[string(n)] == v
        end
    end

    @testset "Different truncation specs produce distinct hashes" begin
        cft_a = CompactBosonCFT(R=1.0, trunc=TruncationSpec(1.0))
        cft_b = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        @test truncation_hash(cft_a.trunc) != truncation_hash(cft_b.trunc)
    end

    @testset "Per-charge spec from custom dict" begin
        # Build a non-uniform spec: more momentum sectors but fewer descendants.
        custom = TruncationSpec(
            bond_cutoffs = Dict(0 => 1, 1 => 0, -1 => 0, 2 => 0, -2 => 0),
            phys_cutoffs = Dict(0 => 1, 1 => 0, -1 => 0),
            R = 1.0,
        )
        cft_c = CompactBosonCFT(R=1.0, trunc=custom)
        @test cft_c.basis_bond.cutoffs == custom.bond_cutoffs
        @test cft_c.basis_phys.cutoffs == custom.phys_cutoffs
        # h_max effective for bond: max over charges of h_n + N_max(n).
        # bond: n=±2 → h_n=2, N=0 ⇒ 2; n=0 → h=0, N=1 ⇒ 1; max = 2.
        # phys: n=0 → h=0, N=1 ⇒ 1; n=±1 → h=0.5, N=0 ⇒ 0.5; max = 1.
        @test cft_c.trunc.h_bond ≈ 2.0
        @test cft_c.trunc.h_phys ≈ 1.0
    end

    @testset "Backward-compat: uniform_cutoffs equivalent to uniform h" begin
        # Old TruncationSpec(2.0) should produce the same cutoffs as
        # uniform_cutoffs(2.0, R=1).
        spec_old = TruncationSpec(2.0)
        @test spec_old.bond_cutoffs == uniform_cutoffs(2.0, 1.0)
        @test spec_old.phys_cutoffs == uniform_cutoffs(2.0, 1.0)
    end

end
