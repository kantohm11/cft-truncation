using Test
using CFTTruncation: build_fock_basis, build_bpz_form, build_bpz_map
using TensorKit

@testset "BPZ" begin

    @testset "7.0 BPZ bilinear form η : V⊗V → ℂ" begin
        basis = build_fock_basis(1.0, 3.0)
        η_form = build_bpz_form(basis)
        # Codomain is trivial, domain is V ⊗ V
        @test codomain(η_form) == one(basis.V)
        @test domain(η_form) == basis.V ⊗ basis.V
    end

    @testset "7.1 Conjugation map: V' ← V, diagonal with ∏(-1)^{k+1}" begin
        basis = build_fock_basis(1.0, 3.0)
        η = build_bpz_map(basis)

        # η is V' ← V (not V ← V)
        @test codomain(η) == ProductSpace(basis.V')
        @test domain(η) == ProductSpace(basis.V)

        for (f₁, f₂) in fusiontrees(η)
            n = Int(f₂.uncoupled[1].charge)
            blk = η[f₁, f₂]
            d = size(blk, 1)
            for α in 1:d
                # BPZ sign for U(1) current (h=1): (-1)^{k+1} per mode
                λ = basis.states[n][α]
                expected = isempty(λ) ? 1.0 : prod(iseven(k) ? -1.0 : 1.0 for k in λ)
                @test blk[α, α] ≈ expected
            end
            for α in 1:d, β in 1:d
                α == β && continue
                @test abs(blk[α, β]) < 1e-15
            end
        end
    end

    @testset "7.2 Specific values at n = 0" begin
        basis = build_fock_basis(1.0, 3.0)
        η = build_bpz_map(basis)
        blk = block(η, U1Irrep(0))
        # BPZ sign = ∏(-1)^{k+1}: odd modes → +1, even modes → -1
        @test blk[1,1] ≈  1.0   # level 0, []
        @test blk[2,2] ≈  1.0   # level 1, [1]     (odd mode → +1)
        @test blk[3,3] ≈ -1.0   # level 2, [2]     (even mode → -1)
        @test blk[4,4] ≈  1.0   # level 2, [1,1]   (odd×2 → +1)
        @test blk[5,5] ≈  1.0   # level 3, [3]     (odd mode → +1)
        @test blk[6,6] ≈ -1.0   # level 3, [2,1]   (even×odd → -1)
        @test blk[7,7] ≈  1.0   # level 3, [1,1,1] (odd×3 → +1)
    end

    @testset "7.3 Involution: η' ∘ η = id on V" begin
        basis = build_fock_basis(1.0, 3.0)
        η = build_bpz_map(basis)
        # η : V' ← V.  η' (adjoint) : V ← V''.  V'' ≅ V canonically.
        # So η' ∘ η : V → V'' ≅ V should be identity (since (±1)² = 1).
        η2 = η' * η
        @test η2 ≈ id(basis.V) atol=1e-14
    end

end
