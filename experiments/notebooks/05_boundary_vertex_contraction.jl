### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ f0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ f0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ f0000001-0003-0000-0000-000000000001
using TensorKit

# ╔═╡ f0000001-0004-0000-0000-000000000001
using Plots

# ╔═╡ f0000001-0005-0000-0000-000000000001
using Printf

# ╔═╡ f0000001-0010-0000-0000-000000000001
md"""
# 05 — Contract Vertex with $|B^{\text{open}}\rangle$

Contract $\widetilde{V}_\ell$ with $|B^{\text{open}}\rangle$ on the T arm,
raise one bond index via BPZ, and check if the resulting operator is
**proportional to the identity**.

All operations via TensorKit composition:
```
Vm_reshaped = permute(Vm, ((1,),(2,3)))    # V' ← V²
M = ψ_B' * Vm_reshaped                     # ℂ ← V²  (contract T)
M_raised = permute(M, ((1,),(2,)))          # V' ← V  (adjoint one leg)
O = η' * M_raised                          # V ← V   (BPZ index raise)
```
"""

# ╔═╡ f0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    h_bond = 4.0   # bond truncation
    h_phys = 8.0   # higher phys truncation — boundary state has weight near cutoff
    set_cache_dir(joinpath(@__DIR__, "..", "..", "experiments", "results", "cache"))
end

# ╔═╡ f0000001-0012-0000-0000-000000000001
cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(h_bond=h_bond, h_phys=h_phys))

# ╔═╡ f0000001-0013-0000-0000-000000000001
md"### Build $|B^{\text{open}}\rangle$ as TensorKit ket ($V_{\text{phys}} \leftarrow \mathbb{C}$)"

# ╔═╡ f0000001-0014-0000-0000-000000000001
"""Single-mode squeezed vacuum coefficient: b(even)=∏√((2j-1)/(2j)), b(odd)=0."""
_bc(m) = m < 0 ? 0.0 : isodd(m) ? 0.0 : (b = 1.0; for j in 2:2:m; b *= sqrt((j-1)/j); end; b)

# ╔═╡ f0000001-0015-0000-0000-000000000001
function build_boundary_ket(basis)
    ψ = zeros(Float64, basis.V, one(basis.V))
    for (f1, f2) in fusiontrees(ψ)
        Int(f1.uncoupled[1].charge) == 0 || continue
        blk = ψ[f1, f2]
        for i in axes(blk, 1)
            lambda = basis.states[0][i]
            b = isempty(lambda) ? 1.0 : begin
                k = lambda[1]; mk = 1; bv = 1.0
                for j in 2:length(lambda)
                    lambda[j] == k ? (mk += 1) : (bv *= _bc(mk); bv == 0 && break; k = lambda[j]; mk = 1)
                end
                bv * _bc(mk)
            end
            blk[i, 1] = b
        end
        ψ[f1, f2] = blk
    end
    ψ
end

# ╔═╡ f0000001-0016-0000-0000-000000000001
ψ_B = build_boundary_ket(cft.basis_phys)

# ╔═╡ f0000001-0020-0000-0000-000000000001
η_bond = CFTTruncation.build_bpz_map(cft.basis_bond)

# ╔═╡ f0000001-0021-0000-0000-000000000001
md"""
### Compute $O(\ell) : V_{\text{bond}} \to V_{\text{bond}}$
"""

# ╔═╡ f0000001-0022-0000-0000-000000000001
function compute_O(cft, ψ_B, η_bond, ell)
    Vm = modified_vertex(compute_vertex(cft, ell))
    bra_B = permute(ψ_B, ((), (1,)))               # ℂ ← V_phys' (bra via permute)
    Vm_reshaped = permute(Vm, ((1,), (2, 3)))       # V_phys' ← V_bond²
    M = bra_B * Vm_reshaped                         # ℂ ← V_bond²
    M_raised = permute(M, ((1,), (2,)))             # V_bond' ← V_bond
    O = η_bond' * M_raised                          # V_bond ← V_bond
    O
end

# ╔═╡ f0000001-0030-0000-0000-000000000001
md"""
### Result: $O \propto (-1)^{\text{level}} \cdot e^{-d(\ell) \cdot L_0}$ (propagator)

$O$ is NOT $\propto I$. It's the **BPZ-signed strip propagator**: diagonal with
eigenvalues $\propto (-1)^N e^{-d(\ell)(h+N)}$. States at the same level have
the same eigenvalue (confirmed numerically).

Extract $d(\ell)$ from $|O_{00}/O_{11}| = e^{d(\ell)}$ (ratio of level-0 to level-1).
"""

# ╔═╡ f0000001-0031-0000-0000-000000000001
function extract_d(O)
    for (f1, f2) in fusiontrees(O)
        Int(f2.uncoupled[1].charge) == 0 || continue
        blk = O[f1, f2]
        return log(abs(blk[1,1]) / abs(blk[2,2]))
    end
    NaN
end

# ╔═╡ f0000001-0040-0000-0000-000000000001
md"### $d(\ell)$ vs $\ell$ and diagonality check"

# ╔═╡ f0000001-0050-0000-0000-000000000001
md"""
## Summary

- $O$ is the BPZ-raised contraction of $\widetilde{V}_\ell$ with $|B^{\text{open}}\rangle$.
- $O$ is **diagonal** (propagator-like) with eigenvalues $\propto (-1)^N e^{-d(\ell)(h+N)}$.
- $d(\ell)$ is the effective propagation distance after the $e^{H\ell/2}$ modification.
- Off-diagonal fraction grows with $\ell$ (truncation effects).
"""

# ╔═╡ f0000001-0032-0000-0000-000000000001
function offdiag_fraction(O)
    for (f1, f2) in fusiontrees(O)
        Int(f2.uncoupled[1].charge) == 0 || continue
        blk = O[f1, f2]; d = size(blk,1)
        diag_sq = sum(blk[i,i]^2 for i in 1:d)
        return sqrt(max(0, 1 - diag_sq / sum(blk .^ 2)))
    end
    NaN
end

# ╔═╡ f0000001-0041-0000-0000-000000000001
let
    ells = collect(0.02:0.02:0.5)
    ds = Float64[]
    offdiags = Float64[]
    for ell in ells
        O = compute_O(cft, ψ_B, η_bond, ell)
        push!(ds, extract_d(O))
        push!(offdiags, offdiag_fraction(O))
    end

    p1 = plot(ells, ds; xlabel="ℓ", ylabel="d(ℓ)",
              title="Effective propagation distance d(ℓ)",
              marker=:circle, markersize=3, legend=false, size=(650, 300))

    p2 = plot(ells, offdiags; xlabel="ℓ", ylabel="off-diag fraction",
              title="Diagonality (0 = perfect propagator)",
              marker=:circle, markersize=3, legend=false,
              yscale=:log10, size=(650, 300))

    plot(p1, p2; layout=(2, 1), size=(650, 550))
end

# ╔═╡ Cell order:
# ╟─f0000001-0010-0000-0000-000000000001
# ╠═f0000001-0001-0000-0000-000000000001
# ╠═f0000001-0002-0000-0000-000000000001
# ╠═f0000001-0003-0000-0000-000000000001
# ╠═f0000001-0004-0000-0000-000000000001
# ╠═f0000001-0005-0000-0000-000000000001
# ╠═f0000001-0011-0000-0000-000000000001
# ╠═f0000001-0012-0000-0000-000000000001
# ╟─f0000001-0013-0000-0000-000000000001
# ╠═f0000001-0014-0000-0000-000000000001
# ╠═f0000001-0015-0000-0000-000000000001
# ╠═f0000001-0016-0000-0000-000000000001
# ╠═f0000001-0020-0000-0000-000000000001
# ╟─f0000001-0021-0000-0000-000000000001
# ╠═f0000001-0022-0000-0000-000000000001
# ╟─f0000001-0030-0000-0000-000000000001
# ╠═f0000001-0031-0000-0000-000000000001
# ╟─f0000001-0040-0000-0000-000000000001
# ╠═f0000001-0041-0000-0000-000000000001
# ╟─f0000001-0050-0000-0000-000000000001
# ╠═f0000001-0032-0000-0000-000000000001
