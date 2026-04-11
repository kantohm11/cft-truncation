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

    @testset "9.3 V(J₋₁|0⟩, J₋₁|0⟩, |0⟩) = -N^{LR}_{1,1}" begin
        vd = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        B = charge_block(vd, 0, 0, 0)
        N_LR_11 = vd.neumann.𝒩.LR[1, 2]
        @test B[2, 2, 1] ≈ -N_LR_11 atol=1e-12
    end

    @testset "9.4 V(J₋₁|1⟩, |-1⟩, |0⟩) Ward identity" begin
        # Peel J₋₁ from L. Residual on L: |1⟩ (primary).
        # Ward: V = -[N^{LL}_{1,0}·J₀ on L - N^{LR}_{1,0}·J₀ on R]·V_prim(1,-1,0)/norm
        # With J₀|n⟩ = (n/R)|n⟩ and R=1:
        #   L contributes: -N^{LL}_{1,0}·(1/1)·V_prim
        #   R contributes: -N^{LR}_{1,0}·(-1/1)·V_prim
        #   T contributes: 0 (J₀|0⟩ = 0)
        vd = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        R = 1.0
        V_prim = primary_vertex(1, -1, 0, vd.geom, R)
        N_LL_10 = vd.neumann.𝒩.LL[1, 1]
        N_LR_10 = vd.neumann.𝒩.LR[1, 1]
        expected = -(N_LL_10 / R - N_LR_10 / R) * V_prim

        B = charge_block(vd, 1, -1, 0)
        # α_L=2 (state [1] in sector 1), α_R=1 (state ∅ in sector -1), α_T=1 (state ∅)
        @test B[2, 1, 1] ≈ expected atol=1e-12
    end

    @testset "9.6 Norm grows with truncation" begin
        vd1 = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(1.0))
        vd2 = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        @test norm(vd1.vertex) > 0
        @test norm(vd2.vertex) > norm(vd1.vertex)
    end

    @testset "10.2 Z₂ symmetry of vertex (L ↔ R)" begin
        # V(n_L, n_R, n_T)[α_L, α_R, α_T] should equal V(n_R, n_L, n_T)[α_R, α_L, α_T]
        # in norm (exact equality depends on basis ordering in sectors ±n).
        for ell in [0.5, 1.0, 2.0]
            vd = compute_vertex(R=1.0, ell=ell, trunc=TruncationSpec(2.0))
            B1 = charge_block(vd, 1, -1, 0)
            B2 = charge_block(vd, -1, 1, 0)
            @test norm(B1) ≈ norm(B2) atol=1e-10
        end
    end

    @testset "10.1 ell-dependence of primary vertex" begin
        # For the free boson, V_prim should vary smoothly with ℓ
        vals = Float64[]
        for ell in [0.5, 1.0, 1.5, 2.0]
            vd = compute_vertex(R=1.0, ell=ell, trunc=TruncationSpec(2.0))
            B = charge_block(vd, 1, -1, 0)
            push!(vals, B[1, 1, 1])
        end
        @test all(isfinite, vals)
        @test all(!iszero, vals)
    end

    @testset "9.5 Tensor structure" begin
        vd = compute_vertex(R=1.0, ell=1.0, trunc=TruncationSpec(2.0))
        # Trilinear form: ℂ ← V_phys ⊗ V_bond ⊗ V_bond
        @test codomain(vd.vertex) == one(vd.basis_phys.V)
        @test domain(vd.vertex) == vd.basis_phys.V ⊗ vd.basis_bond.V ⊗ vd.basis_bond.V
    end

end
