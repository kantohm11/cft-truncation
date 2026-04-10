using Test
using CFTTruncation: compute_geometry, compute_sc_params, fprime_exact,
                     TruncLaurent, evaluate, compose, series_precision

@testset "LocalCoordinates" begin

    @testset "3.1 Residue check" begin
        geom = compute_geometry(1.0, 20)
        sc = geom.sc

        # Residue at x_R = 1: C√(1-p²)/2 = 1/π
        expected_res_R = sc.C * sqrt(1 - sc.p^2) / 2
        @test expected_res_R ≈ 1/π

        # Residue at x_T = 0: |Res| = Cp = ℓ/π
        @test sc.C * sc.p ≈ 1/π   # at ℓ = 1

        # Check arm widths and directions are consistent
        @test geom.arms.R.w ≈ 1.0
        @test geom.arms.L.w ≈ 1.0
        @test geom.arms.T.w ≈ 1.0  # ℓ = 1
    end

    @testset "3.2 Round-trip: f_i(g_i(ξ)) = ξ" begin
        geom = compute_geometry(1.0, 30)

        for arm in (:L, :R, :T)
            f = getfield(geom.arms, arm).f_series
            g = getfield(geom.arms, arm).g_series

            h = compose(f, g)
            @test h[1] ≈ 1.0 atol=1e-12
            for k in 2:20
                @test abs(h[k]) < 1e-10
            end
        end
    end

    @testset "3.3 Leading coefficient (Jacobian)" begin
        geom = compute_geometry(1.0, 20)

        for arm in (:L, :R, :T)
            a = getfield(geom.arms, arm)
            @test a.f_series[1] ≈ a.α
            @test abs(a.α) > 0
        end
    end

    @testset "3.4 Z₂ symmetry: L vs R" begin
        geom = compute_geometry(1.0, 20)
        α_L = geom.arms.L.α
        α_R = geom.arms.R.α
        @test abs(α_L) ≈ abs(α_R)
    end

    @testset "3.5 Series vs direct evaluation" begin
        geom = compute_geometry(1.0, 20)
        sc = geom.sc
        ζ_test = 0.01 + 0.02im

        for arm in (:L, :R, :T)
            a = getfield(geom.arms, arm)
            z = a.x + ζ_test

            # Direct evaluation of f'
            fp_direct = fprime_exact(z, sc)

            # From the stored fprime Laurent series
            fp_series = evaluate(a.fprime_series, ζ_test)

            @test fp_direct ≈ fp_series rtol=1e-8
        end
    end

    @testset "Multiple ℓ values" begin
        for ℓ in [0.5, 1.0, 2.0]
            geom = compute_geometry(ℓ, 20)
            @test geom.arms.T.w ≈ ℓ

            # Round-trip still works
            for arm in (:L, :R, :T)
                a = getfield(geom.arms, arm)
                h = compose(a.f_series, a.g_series)
                @test h[1] ≈ 1.0 atol=1e-10
                for k in 2:10
                    @test abs(h[k]) < 1e-8
                end
            end
        end
    end

end
