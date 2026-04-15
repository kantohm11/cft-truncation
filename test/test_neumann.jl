using Test
using CFTTruncation: compute_geometry, compute_neumann, compose,
                     TruncLaurent, singular_part

@testset "NeumannCoefficients" begin

    @testset "4.1 Diagonal self-consistency" begin
        geom = compute_geometry(1.0, 15)
        neum = compute_neumann(geom, 10)

        for arm in (:L, :R, :T)
            Fm_polys = neum.F_polys[arm]
            a = getfield(geom.arms, arm)
            for m in 1:5
                composed = CFTTruncation._compose_Fm_with_g(
                    Fm_polys[m], a, a, geom.order)
                @test abs(composed[-m] - 1.0) < 1e-6
                for k in (-m+1):(-1)
                    @test abs(composed[k]) < 1e-6
                end
            end
        end
    end

    @testset "4.2 Z₂ symmetry: N^{LL} = (-1)^{m+k} N^{RR}" begin
        # With α_L = α_R (ρ₀^R shifted by +i): f_L(ζ) = -f_R(-ζ),
        # giving N^{LL}_{m,k} = (-1)^{m+k} N^{RR}_{m,k}.
        for ℓ in [0.5, 1.0, 2.0]
            geom = compute_geometry(ℓ, 15)
            neum = compute_neumann(geom, 6)

            for m in 1:4, k in 0:4
                @test neum.𝒩.LL[m, k+1] ≈ (-1)^(m+k) * neum.𝒩.RR[m, k+1] atol=1e-10
            end
        end
    end

    @testset "4.3 Neumann coefficients are real and finite" begin
        for ℓ in [0.5, 1.0, 2.0]
            geom = compute_geometry(ℓ, 15)
            neum = compute_neumann(geom, 6)

            for arm_key in (:LL, :LR, :LT, :RL, :RR, :RT, :TL, :TR, :TT)
                mat = getfield(neum.𝒩, arm_key)
                @test all(isfinite, mat)
            end
        end
    end

    @testset "4.4 Convergence with series order" begin
        geom_12 = compute_geometry(1.0, 12)
        neum_12 = compute_neumann(geom_12, 5)

        geom_22 = compute_geometry(1.0, 22)
        neum_22 = compute_neumann(geom_22, 5)

        for m in 1:3, k in 1:3
            @test neum_12.𝒩.LR[m, k] ≈ neum_22.𝒩.LR[m, k] rtol=1e-4
        end
    end

end
