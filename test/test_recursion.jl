using Test
using CFTTruncation: compute_vertex, charge_block, TruncationSpec, primary_vertex
using TensorKit

@testset "Recursion" begin

    @testset "9.1 Level 0 matches primary vertex" begin
        vd = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))

        # V(|0;0⟩, |0;0⟩, |0;0⟩) = 1
        B = charge_block(vd, 0, 0, 0)
        @test B[1, 1, 1] ≈ 1.0

        # Another primary triple: (1,-1,0)
        B2 = charge_block(vd, 1, -1, 0)
        @test B2[1, 1, 1] ≈ primary_vertex(1, -1, 0, vd.geom, 1.0)
    end

    @testset "9.2 V(J₋₁|0⟩, |0⟩, |0⟩) = 0" begin
        vd = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        B = charge_block(vd, 0, 0, 0)
        # State 2 in sector 0 is [1] (J_{-1}|0⟩ normalized)
        # All J_k on |0;0⟩ give zero, so Ward identity sum vanishes
        @test abs(B[2, 1, 1]) < 1e-12
    end

    @testset "9.6 Norm grows with truncation" begin
        vd1 = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(1.0))
        vd2 = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        @test norm(vd1.vertex) > 0
        @test norm(vd2.vertex) > norm(vd1.vertex)
    end

    @testset "9.5 Tensor structure" begin
        vd = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        # Trilinear form: ℂ ← V_phys ⊗ V_bond ⊗ V_bond
        @test codomain(vd.vertex) == one(vd.basis_phys.V)
        @test domain(vd.vertex) == vd.basis_phys.V ⊗ vd.basis_bond.V ⊗ vd.basis_bond.V
    end

end
