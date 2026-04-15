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
            # T arm has reduced precision due to small |α_T| at ℓ = 1
            tol = arm == :T ? 1e-4 : 1e-10
            @test abs(h[1] - 1.0) < tol
            for k in 2:10
                @test abs(h[k]) < tol
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

    @testset "3.4 Z₂ symmetry: |α_L| = |α_R|" begin
        for ℓ in [0.5, 1.0, 2.0]
            geom = compute_geometry(ℓ, 20)
            @test abs(geom.arms.L.α) ≈ abs(geom.arms.R.α) rtol=1e-10
        end
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

    # ξ_R(p) = +1 (ρ₀^R shifted by +i to fix propagator sign).
    # ξ_L(-p) = -1 (L arm ρ₀ unchanged; Z₂ gives α_L = α_R).
    @testset "3.6 Corners at unit semicircle (R_conv = 1)" begin
        for ℓ in [0.5, 1.0, 2.0]
            geom = compute_geometry(ℓ, 40)
            p = geom.sc.p

            ξ_R = evaluate(geom.arms.R.f_series, p - 1)
            ξ_L = evaluate(geom.arms.L.f_series, 1 - p)
            ξ_Tp = evaluate(geom.arms.T.f_series, p)
            ξ_Tm = evaluate(geom.arms.T.f_series, -p)

            # Magnitude: all corners on the unit circle
            @test abs(ξ_R) ≈ 1  atol=5e-2
            @test abs(ξ_L) ≈ 1  atol=5e-2
            @test abs(ξ_Tp) ≈ 1  atol=5e-2
            @test abs(ξ_Tm) ≈ 1  atol=5e-2

            # Sign: R corner at +1, L corner at -1, T corners at ±1
            @test real(ξ_R) ≈ +1  atol=5e-2
            @test real(ξ_L) ≈ -1  atol=5e-2
            @test real(ξ_Tp) ≈ +1  atol=5e-2
            @test real(ξ_Tm) ≈ -1  atol=5e-2
        end
    end

    @testset "Multiple ℓ values" begin
        for ℓ in [1.0, 2.0]  # skip ℓ=0.5: |α_T| too small for order 20
            geom = compute_geometry(ℓ, 20)
            @test geom.arms.T.w ≈ ℓ

            for arm in (:L, :R, :T)
                a = getfield(geom.arms, arm)
                h = compose(a.f_series, a.g_series)
                tol = arm == :T ? 1e-4 : 1e-8
                @test abs(h[1] - 1.0) < tol
                for k in 2:8
                    @test abs(h[k]) < tol
                end
            end

            # Z₂: |α_L| = |α_R|
            @test abs(geom.arms.L.α) ≈ abs(geom.arms.R.α) rtol=1e-10
        end
    end

end
