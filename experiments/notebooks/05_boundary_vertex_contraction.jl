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
md"### Check $O \propto I$"

# ╔═╡ f0000001-0031-0000-0000-000000000001
let
    ells = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    V = cft.basis_bond.V
    I_V = id(V)

    lines = ["  ℓ       ‖O‖           ‖O/‖O‖ - c·I/‖I‖‖"]
    push!(lines, repeat("-", 55))
    for ell in ells
        O = compute_O(cft, ψ_B, η_bond, ell)
        # Best scalar multiple: c = tr(O†·I) / tr(I†·I) = tr(O) / dim
        # For TensorKit: tr via @tensor or sum of diagonal blocks
        nO = norm(O)
        nI = norm(I_V)
        # Cosine similarity: |⟨O, I⟩| / (‖O‖·‖I‖)
        # ⟨O, I⟩ = tr(O†·I) — for real TensorMaps this is just the Frobenius inner product
        # In TensorKit: dot(O, I_V)... or just compute O_normalized vs I_normalized
        O_hat = O / nO
        I_hat = I_V / nI
        dist = norm(O_hat - dot(O_hat, I_hat) * I_hat)  # rejection from I direction
        cosine = abs(real(dot(O_hat, I_hat)))
        push!(lines, @sprintf("  %.2f    %12.4e    cos(O,I) = %.8f   dist = %.2e", ell, nO, cosine, dist))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0040-0000-0000-000000000001
md"### Plot: cosine similarity $|\cos(O, I)|$ vs $\ell$"

# ╔═╡ f0000001-0041-0000-0000-000000000001
let
    ells = collect(0.05:0.05:1.0)
    V = cft.basis_bond.V
    I_V = id(V)
    nI = norm(I_V)
    cosines = Float64[]
    for ell in ells
        O = compute_O(cft, ψ_B, η_bond, ell)
        push!(cosines, abs(real(dot(O / norm(O), I_V / nI))))
    end
    plot(ells, cosines; xlabel="ℓ", ylabel="|cos(O, I)|",
         title="Is O ∝ I?  (1.0 = perfect proportionality)",
         marker=:circle, markersize=4, legend=false,
         ylims=(0, 1.05), size=(650, 400))
end

# ╔═╡ f0000001-0050-0000-0000-000000000001
md"""
## Summary

- $|B^{\text{open}}\rangle$ contracted with $\widetilde{V}_\ell$ via TensorKit composition.
- Raised one bond index via BPZ → operator $O : V_{\text{bond}} \to V_{\text{bond}}$.
- Measured cosine similarity $|\cos(O, I)| = |\langle O/\|O\|, I/\|I\| \rangle|$.
- If $\approx 1.0$ for all $\ell$, the boundary state correctly caps the arm and $O \propto I$.
"""

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
