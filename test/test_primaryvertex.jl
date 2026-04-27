using Test
using CFTTruncation: compute_geometry, primary_vertex

@testset "PrimaryVertex" begin

    @testset "8.1 Selection rule" begin
        geom = compute_geometry(1.0, 5)
        R = 1.0
        @test primary_vertex(1, -1, 0, geom, R) != 0
        @test primary_vertex(0, 0, 0, geom, R) != 0
        @test primary_vertex(1, 0, 0, geom, R) == 0
        @test primary_vertex(1, 1, 1, geom, R) == 0
    end

    @testset "8.2 All-zero momenta" begin
        geom = compute_geometry(1.0, 5)
        @test primary_vertex(0, 0, 0, geom, 1.0) ≈ 1.0
    end

    @testset "8.3 Non-trivial primary vertex" begin
        geom = compute_geometry(1.0, 5)
        α_L = geom.arms.L.α
        α_R = geom.arms.R.α
        # V(1,-1,0) at R=1: charges are boundary-doubled, Δ_L = Δ_R = 1, Δ_T = 0.
        # Jacobian = (1/|α_L|)·(1/|α_R|), distance factor = 1/|x_L − x_R|² = 1/4.
        # ⇒ V = 1 / (4 · |α_L| · |α_R|).
        @test primary_vertex(1, -1, 0, geom, 1.0) ≈ 1.0 / (4 * abs(α_L) * abs(α_R))
    end

    @testset "8.4 L↔R symmetry" begin
        geom = compute_geometry(1.0, 5)
        @test primary_vertex(1, -1, 0, geom, 1.0) ≈
              primary_vertex(-1, 1, 0, geom, 1.0)
    end

end
