using Test
using CFTTruncation: build_fock_basis, FockBasis
using TensorKit
using TensorKit: dim, U1Irrep, GradedSpace, fusiontrees, Vect
using LinearAlgebra: norm

@testset "FockSpace" begin

    @testset "5.1 State counting" begin
        basis = build_fock_basis(1.0, 2.0)
        @test length(keys(basis.states)) == 5
        @test Set(keys(basis.states)) == Set([-2, -1, 0, 1, 2])
        @test length(basis.states[0]) == 4
        @test length(basis.states[1]) == 2
        @test length(basis.states[-1]) == 2
        @test length(basis.states[2]) == 1
        @test length(basis.states[-2]) == 1
        @test dim(basis.V) == 10
    end

    @testset "5.1b Higher truncation" begin
        basis = build_fock_basis(1.0, 3.0)
        @test length(basis.states[0]) == 7
        @test length(basis.states[1]) == 4
        @test length(basis.states[2]) == 2
        @test dim(basis.V) == 7 + 4 + 4 + 2 + 2  # = 19
    end

    @testset "5.2 Partition ordering" begin
        basis = build_fock_basis(1.0, 3.0)
        @test basis.states[0][1] == Int[]       # primary, level 0
        @test basis.states[0][2] == [1]         # level 1
        @test basis.states[0][3] == [2]         # level 2, revlex: [2] before [1,1]
        @test basis.states[0][4] == [1, 1]      # level 2
        @test basis.states[0][5] == [3]         # level 3
        @test basis.states[0][6] == [2, 1]      # level 3
        @test basis.states[0][7] == [1, 1, 1]   # level 3
    end

    @testset "5.3 Levels" begin
        basis = build_fock_basis(1.0, 3.0)
        @test basis.levels[0] == [0, 1, 2, 2, 3, 3, 3]
        @test basis.levels[1] == [0, 1, 2, 2]
    end

    @testset "5.4 Normalisation factors z_λ" begin
        basis = build_fock_basis(1.0, 3.0)
        zl = basis.z_lambda[0]
        @test zl[1] ≈ 1.0      # z_{∅} = 1
        @test zl[2] ≈ 1.0      # z_{[1]} = 1
        @test zl[3] ≈ 2.0      # z_{[2]} = 2
        @test zl[4] ≈ 2.0      # z_{[1,1]} = 1²·2! = 2
        @test zl[5] ≈ 3.0      # z_{[3]} = 3
        @test zl[6] ≈ 2.0      # z_{[2,1]} = 2
        @test zl[7] ≈ 6.0      # z_{[1,1,1]} = 1³·3! = 6
    end

    @testset "5.5 Graded space" begin
        basis = build_fock_basis(1.0, 2.0)
        V = basis.V
        @test V isa GradedSpace
        @test dim(V, U1Irrep(0)) == 4
        @test dim(V, U1Irrep(1)) == 2
        @test dim(V, U1Irrep(3)) == 0
    end

    # The unit-normalised Fock convention requires that TensorKit's `norm`
    # and `dot` on Vect[U1Irrep](...) reduce to the standard Euclidean
    # inner product on block-coefficient vectors. If a future TensorKit
    # version applied any non-trivial metric (e.g. quantum-dim weights),
    # the Heisenberg algebra coefficients in JMatrices.jl would silently
    # disagree with TensorKit's inner product. These tests guard against
    # that.
    @testset "5.6 TensorKit norm = Euclidean on block coefficients" begin
        basis = build_fock_basis(1.0, 3.0)
        V = basis.V

        # Random covector V → 1.
        for _ in 1:3
            t = zeros(Float64, V, one(V))
            raw_sum = 0.0
            for (f₁, f₂) in fusiontrees(t)
                blk = t[f₁, f₂]
                for i in eachindex(blk)
                    blk[i] = randn()
                    raw_sum += blk[i]^2
                end
                t[f₁, f₂] = blk
            end
            @test norm(t) ≈ sqrt(raw_sum) atol=1e-12
        end

        # Trilinear vertex shape 1 → V ⊗ V ⊗ V.
        for _ in 1:3
            t = zeros(Float64, one(V), V ⊗ V ⊗ V)
            raw_sum = 0.0
            for (f₁, f₂) in fusiontrees(t)
                blk = t[f₁, f₂]
                for i in eachindex(blk)
                    blk[i] = randn()
                    raw_sum += blk[i]^2
                end
                t[f₁, f₂] = blk
            end
            @test norm(t) ≈ sqrt(raw_sum) atol=1e-12
        end
    end

    @testset "5.7 Single-entry kets: norm = |entry| in every charge sector" begin
        basis = build_fock_basis(1.0, 3.0)
        V = basis.V
        for n in keys(basis.states)
            aux = Vect[U1Irrep](U1Irrep(n) => 1)
            for α in 1:length(basis.states[n])
                t = zeros(Float64, V, aux)
                for (f₁, f₂) in fusiontrees(t)
                    Int(f₁.uncoupled[1].charge) == n || continue
                    blk = t[f₁, f₂]
                    blk[α, 1] = 1.5
                    t[f₁, f₂] = blk
                end
                @test norm(t) ≈ 1.5 atol=1e-14
            end
        end
    end

end
