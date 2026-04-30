using Test
using CFTTruncation: build_fock_basis, build_J_matrices, build_creation_matrix
using LinearAlgebra: I as eye
using SparseArrays: sparse, nnz

@testset "JMatrices" begin

    @testset "6.1 J₀ diagonal with eigenvalue n/R" begin
        basis = build_fock_basis(1.0, 3.0)
        J, _ = build_J_matrices(basis, 3)
        for n in keys(basis.states)
            d = length(basis.states[n])
            J0 = J[n][1]   # k=0 stored at index 1
            @test J0 ≈ (n / 1.0) * eye(d)
        end
    end

    @testset "6.2 J₁ matrix elements at n = 0" begin
        basis = build_fock_basis(1.0, 3.0)
        J, _ = build_J_matrices(basis, 3)
        J1 = J[0][2]   # k=1 at index 2, sector n=0
        @test size(J1) == (7, 7)
        # Check nonzero entries
        @test J1[1, 2] ≈ 1.0             # [1] → ∅
        @test J1[2, 4] ≈ √2              # [1,1] → [1]
        @test J1[3, 6] ≈ 1.0             # [2,1] → [2]
        @test J1[4, 7] ≈ √3              # [1,1,1] → [1,1]
        @test nnz(sparse(J1)) == 4
    end

    @testset "6.3 J₂ matrix elements at n = 0" begin
        basis = build_fock_basis(1.0, 3.0)
        J, _ = build_J_matrices(basis, 3)
        J2 = J[0][3]   # k=2 at index 3
        @test J2[1, 3] ≈ √2              # [2] → ∅
        @test J2[2, 6] ≈ √2              # [2,1] → [1]
    end

    @testset "6.4 Commutation [J₁, J₋₁] = I at n=0" begin
        # [J_1, J_{-1}] = I holds exactly only on states where J_{-1}
        # doesn't push outside the truncated space. At h_max=4, n=0,
        # states up to level 3 are safe (J_{-1} adds level 1 → level 4 exists).
        basis = build_fock_basis(1.0, 4.0)
        J, _ = build_J_matrices(basis, 4)
        J_minus1 = build_creation_matrix(basis, 0, 1)
        J1 = J[0][2]
        comm = J1 * J_minus1 - J_minus1 * J1
        # Check on safe subspace (levels 0-3)
        safe = findall(l -> l <= 3, basis.levels[0])
        for i in safe
            @test comm[i, i] ≈ 1.0 atol=1e-12
        end
    end

    @testset "6.5 Commutation at n ≠ 0" begin
        basis = build_fock_basis(1.0, 4.0)
        J, _ = build_J_matrices(basis, 4)
        J_minus1 = build_creation_matrix(basis, 1, 1)
        J1 = J[1][2]
        comm = J1 * J_minus1 - J_minus1 * J1
        # At n=1, h_n=0.5, max_level=3. J_{-1} adds 1, so safe up to level 2.
        h_n = (1 / basis.R)^2 / 2
        max_safe_level = floor(Int, basis.h_max - h_n - 1 + 1e-10)
        safe = findall(l -> l <= max_safe_level, basis.levels[1])
        for i in safe
            @test comm[i, i] ≈ 1.0 atol=1e-12
        end
    end

    # Generalises 6.4–6.5 to k = 1..5 across all charge sectors and
    # adds the off-diagonal vanishing [J_k, J_{-l}] = 0 (k ≠ l) plus the
    # adjoint property J_k = transpose(J_{-k}). All are exact algebraic
    # identities in the unit-normalised basis (worst-case error 0.0).
    @testset "6.6 Heisenberg algebra across sectors and k ≥ 1" begin
        basis = build_fock_basis(1.0, 5.0)
        k_max = 5
        J, _ = build_J_matrices(basis, k_max)
        for n in keys(basis.states)
            for k in 1:k_max
                Jk = J[n][k + 1]
                Jmk = build_creation_matrix(basis, n, k)
                @test Jk == transpose(Jmk)

                cap = basis.cutoffs[n]
                safe_kk = findall(l -> l + k ≤ cap, basis.levels[n])
                comm_kk = Jk * Jmk - Jmk * Jk
                for i in safe_kk
                    @test comm_kk[i, i] ≈ k atol=1e-12
                end

                for l in 1:k_max
                    l == k && continue
                    Jml = build_creation_matrix(basis, n, l)
                    comm_kl = Jk * Jml - Jml * Jk
                    safe_kl = findall(lvl -> lvl + l ≤ cap, basis.levels[n])
                    for i in safe_kl
                        @test comm_kl[i, i] ≈ 0.0 atol=1e-12
                    end
                end
            end
        end
    end

end
