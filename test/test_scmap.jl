using Test
using CFTTruncation: compute_sc_params, fprime_exact

@testset "SCMap" begin

    @testset "2.1 Closed-form values" begin
        sc = compute_sc_params(1.0)
        @test sc.p ≈ 1/√5
        @test sc.C ≈ √5/π

        sc = compute_sc_params(2.0)
        @test sc.p ≈ 1/√2
        @test sc.C ≈ 2√2/π
    end

    @testset "2.2 Limits" begin
        # ℓ → 0: p → 0
        sc = compute_sc_params(1e-10)
        @test sc.p < 1e-10

        # ℓ → ∞: p → 1
        sc = compute_sc_params(1e6)
        @test sc.p > 1 - 1e-6

        # Identity: Cp = ℓ/π always
        for ℓ in [0.1, 0.5, 1.0, 2.0, 5.0]
            sc = compute_sc_params(ℓ)
            @test sc.C * sc.p ≈ ℓ/π
        end
    end

    @testset "2.3 f' evaluation" begin
        sc = compute_sc_params(1.0)
        z = 0.5 + 0.1im
        fp = fprime_exact(z, sc)
        @test fp isa ComplexF64
        @test isfinite(fp)

        # Check residue at z=1: Res = C√(1-p²)/2 = 1/π
        # f'(z) ≈ (1/π)/(z-1) near z=1
        δ = 1e-8
        res_approx = fprime_exact(1.0 + δ*im, sc) * (δ*im)
        @test abs(res_approx - 1/π) < 1e-4
    end

end
