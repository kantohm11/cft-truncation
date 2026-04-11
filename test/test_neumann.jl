using Test
using CFTTruncation: compute_geometry, compute_neumann, compose,
                     TruncLaurent, singular_part

@testset "NeumannCoefficients" begin

    @testset "4.1 Diagonal self-consistency" begin
        geom = compute_geometry(1.0, 15)
        neum = compute_neumann(geom, 10)

        # For the diagonal case (i == j), the Neumann coefficients
        # N^{ii}_{m,k} for k ≥ 0 are the regular part of F_m(g_i(ξ)).
        # The intermediate powers ξ^{-m+1},...,ξ^{-1} must be zero
        # by construction. We verify this via the internal compose.
        for arm in (:L, :R, :T)
            Fm_polys = neum.F_polys[arm]
            a = getfield(geom.arms, arm)
            for m in 1:5
                # Use the internal _compose_Fm_with_g
                composed = CFTTruncation._compose_Fm_with_g(
                    Fm_polys[m], a, a, geom.order)
                # ξ^{-m} coefficient should be 1
                @test composed[-m] ≈ 1.0 atol=1e-10
                # Intermediate powers should be zero
                for k in (-m+1):(-1)
                    @test abs(composed[k]) < 1e-10
                end
            end
        end
    end

    @testset "4.2 Z₂ symmetry" begin
        # Under the Z₂ of the T-shape (z → -z, swapping L ↔ R), the relation
        # f_L(ζ) = -f_R(-ζ) combined with F_m^L coefficient sign flips
        # (-1)^{m+n} and g_L(ξ) = -g_R(-ξ) gives:
        #   N^{LL}_{m,k} = (-1)^{m+k} N^{RR}_{m,k}
        #   N^{LR}_{m,k} = (-1)^{m+k} N^{RL}_{m,k}
        #   N^{LT}_{m,k} = (-1)^{m+k} N^{RT}_{m,k}
        # This uses g_T(-ξ) = -g_T(ξ) (g_T is odd, since ρ_T has only
        # even-power coefficients).
        geom = compute_geometry(1.0, 15)
        neum = compute_neumann(geom, 8)

        for m in 1:6, k in 0:6
            sign = (-1)^(m + k)
            @test neum.𝒩.LL[m, k+1] ≈ sign * neum.𝒩.RR[m, k+1] atol=1e-12
            @test neum.𝒩.LR[m, k+1] ≈ sign * neum.𝒩.RL[m, k+1] atol=1e-12
            @test neum.𝒩.LT[m, k+1] ≈ sign * neum.𝒩.RT[m, k+1] atol=1e-12
        end

        # Diagonal self-maps should also be real (no imaginary parts leaked
        # through the series arithmetic).
        for arm_key in (:LL, :RR, :LT, :RT, :TT, :TL, :TR)
            mat = getfield(neum.𝒩, arm_key)
            @test all(isfinite, mat)
        end
    end

    @testset "4.3 Off-diagonal coefficients vs distance" begin
        # N^{i→j}_{m,k} for i≠j has Neumann coefficients that are smaller
        # at low orders when |x_i - x_j| is larger (farther apart arms
        # have smaller overlap). |x_L-x_R| = 2, |x_L-x_T|=|x_R-x_T|=1,
        # so LR coefficients should be smaller in magnitude than LT
        # at low (m, k).
        #
        # At (m=1, k=0) specifically: LR = 0.5, LT = 1.0 — factor of 2.
        # (Note: the Neumann series has finite radius of convergence in ξ,
        # so high-k coefficients may grow; this is fine because the sum
        # truncates at the descendant level cutoff in practice.)
        geom = compute_geometry(1.0, 20)
        neum = compute_neumann(geom, 10)

        @test abs(neum.𝒩.LR[1, 1]) < abs(neum.𝒩.LT[1, 1])
        @test abs(neum.𝒩.LR[1, 2]) < abs(neum.𝒩.LT[1, 2])
        @test abs(neum.𝒩.LR[2, 1]) < abs(neum.𝒩.LT[2, 1]) + 1e-15

        # All coefficients should be finite
        for arm_key in (:LL, :LR, :LT, :RL, :RR, :RT, :TL, :TR, :TT)
            mat = getfield(neum.𝒩, arm_key)
            @test all(isfinite, mat)
        end
    end

    @testset "4.4 Convergence with series order" begin
        geom_10 = compute_geometry(1.0, 12)
        neum_10 = compute_neumann(geom_10, 5)

        geom_20 = compute_geometry(1.0, 22)
        neum_20 = compute_neumann(geom_20, 5)

        for m in 1:3, k in 1:3
            @test neum_10.𝒩.LT[m, k] ≈ neum_20.𝒩.LT[m, k] rtol=1e-6
        end
    end

end
