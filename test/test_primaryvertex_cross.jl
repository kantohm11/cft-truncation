using Test
using CFTTruncation: compute_geometry_cross, primary_vertex

@testset "PrimaryVertex cross" begin

    @testset "Selection rule (4-charge conservation)" begin
        geom = compute_geometry_cross(1.0, 5)
        R = 1.0
        # Σn = 0 required
        @test primary_vertex(1, -1, 0, 0, geom, R) != 0
        @test primary_vertex(1, 1, -1, -1, geom, R) != 0
        @test primary_vertex(0, 0, 0, 0, geom, R) != 0
        # Σn ≠ 0 must return 0
        @test primary_vertex(1, 0, 0, 0, geom, R) == 0
        @test primary_vertex(1, 1, 1, 0, geom, R) == 0
        @test primary_vertex(2, -1, 0, 0, geom, R) == 0
    end

    @testset "All-zero momenta return 1" begin
        geom = compute_geometry_cross(1.0, 5)
        @test primary_vertex(0, 0, 0, 0, geom, 1.0) ≈ 1.0
    end

    @testset "Reduces to T-shape formula when n_B = 0" begin
        # With n_B = 0, h_B = 0, so |α_B|^{2h_B} = 1 and B contributes
        # nothing to the finite-pair distance product. The cross answer
        # for (n_L, n_R, n_T, 0) should match what the T-shape formula
        # gives for (n_L, n_R, n_T), evaluated with the cross geometry's
        # α_L, α_R, α_T.
        geom = compute_geometry_cross(1.0, 10)
        α_L = abs(geom.arms.L.α)
        α_R = abs(geom.arms.R.α)
        # (n_L=1, n_R=-1, n_T=0, n_B=0): h_L=h_R=1/2.
        # Jacobian = |α_L|·|α_R|, finite-pair factor |x_L-x_R|^{p_L·p_R} = 2^{-1}.
        expected = α_L * α_R * 2.0^(-1)
        @test primary_vertex(1, -1, 0, 0, geom, 1.0) ≈ expected  rtol=1e-12
    end

    @testset "Four-charge case with explicit formula" begin
        # (n_L, n_R, n_T, n_B) = (1, 1, -1, -1) at R=1:
        #   h_i = 1/2 for all four, p_i = ±1.
        #   Jacobian = ∏|α_i|.
        #   Finite pairs (L,R), (L,T), (R,T):
        #     |x_L-x_R|^{p_L·p_R} = 2^{+1} = 2
        #     |x_L-x_T|^{p_L·p_T} = 1^{-1} = 1
        #     |x_R-x_T|^{p_R·p_T} = 1^{-1} = 1
        #   V = 2 · ∏|α_i|.
        geom = compute_geometry_cross(1.0, 10)
        α_prod = abs(geom.arms.L.α) * abs(geom.arms.R.α) *
                 abs(geom.arms.T.α) * abs(geom.arms.B.α)
        @test primary_vertex(1, 1, -1, -1, geom, 1.0) ≈ 2 * α_prod  rtol=1e-12
    end

    @testset "Horizontal Z₂: L↔R symmetry" begin
        geom = compute_geometry_cross(1.0, 10)
        # |x_L - x_T| = |x_R - x_T| = 1 and |α_L| = |α_R|, so swapping
        # n_L and n_R leaves V invariant.
        for (nL, nR, nT, nB) in ((1, -1, 0, 0), (2, -2, 0, 0), (1, 1, -1, -1),
                                 (3, -1, -1, -1), (2, 0, -1, -1))
            @test primary_vertex(nL, nR, nT, nB, geom, 1.0) ≈
                  primary_vertex(nR, nL, nT, nB, geom, 1.0)  rtol=1e-12
        end
    end

    @testset "No dependence on x_B = Inf (handled via α_B)" begin
        # Check that the implementation doesn't produce NaN or Inf from
        # reading x_B = Inf directly — it should only use |α_B|^{2h_B}.
        geom = compute_geometry_cross(1.0, 10)
        for (nL, nR, nT, nB) in ((0, 0, 0, 0), (1, -1, 1, -1), (1, 1, -1, -1),
                                 (2, -1, -1, 0), (3, -2, -1, 0))
            v = primary_vertex(nL, nR, nT, nB, geom, 1.0)
            @test isfinite(v)
        end
    end

    @testset "Radius scaling" begin
        # At R → ∞, all weights h_i → 0, so V → 1 for any conserved charges.
        geom = compute_geometry_cross(1.0, 10)
        for (nL, nR, nT, nB) in ((1, -1, 0, 0), (1, 1, -1, -1), (2, -1, -1, 0))
            @test primary_vertex(nL, nR, nT, nB, geom, 1e10) ≈ 1.0  atol=1e-6
        end
    end

end
