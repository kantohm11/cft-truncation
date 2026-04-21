### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ f0000009-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ f0000009-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ f0000009-0003-0000-0000-000000000001
using TensorKit

# ╔═╡ f0000009-0004-0000-0000-000000000001
using LinearAlgebra

# ╔═╡ f0000009-0005-0000-0000-000000000001
using Plots

# ╔═╡ f0000009-0006-0000-0000-000000000001
using Printf

# ╔═╡ f0000009-0010-0000-0000-000000000001
md"""
# 09 — Phase 1: Finite-Entanglement Scaling from $V_\ell$

Treat $V_\ell$ as an MPS tensor with $V_T$ = physical leg and
$V_L, V_R$ = bond legs. Build the MPS transfer matrix
$E = V_\ell \otimes V_\ell^\perp$ contracted over $V_T$
(the "I-shape"). Extract:

- **Correlation length** $\xi_D = -1 / \log |\lambda_1|$ from the
  subleading eigenvalue of $E$ (after normalising $\lambda_0 = 1$).
- **Entanglement entropy** $S_D$ via the mixed-canonical $C$ matrix:
  left/right fixed points $l, r$, factor $l = L^\dagger L$,
  $r = R R^\dagger$, $C = LR$, SVD $C$ to get Schmidt values $C_i$, and
  $S_D = -\sum_i C_i^2 \log C_i^2$.

Finite-entanglement scaling: $S_D \sim (c/6) \log \xi_D$, target $c = 1$.

See [`docs/design/finite_entanglement_scaling.md`](../../docs/design/finite_entanglement_scaling.md)
for the full method writeup.

## Note on the bra layer: $V_\ell^\dagger$, not $V_\ell^\perp$

In MPS language, the transfer matrix uses the Hermitian conjugate of
the ket: $E = \sum_s A^s \otimes (A^s)^\dagger$. For $A = V_\ell$ with
real entries, $V_\ell^\dagger$ has the same numerical values as
$V_\ell$ (only the index interpretation — which space is domain vs
codomain — changes), so the contraction is just
$\sum_T V_{T,L,R}\,V_{T,L',R'}$. That's what this notebook
computes.

The $V_\ell^\perp$ of `truncation_strategies.md` §5 (geometric
180°-flipped T-vertex) is a **different** object used for Phase 2
(cross / MPO composition). Don't confuse the two.
"""

# ╔═╡ f0000009-0020-0000-0000-000000000001
md"""
## Setup

Fixed radius $R = 1$. Fixed $\ell$ in the good regime (propagator test
convergent). Sweep $h_{\text{bond}}$ (= effective bond dimension $D$)
with $h_{\text{phys}}$ held large enough to not be the binding
constraint.
"""

# ╔═╡ f0000009-0021-0000-0000-000000000001
begin
    R_val = 1.0
    ell = 0.1
    h_phys = 8.0
    # Small sweep by default for a fast first run; extend after the
    # pipeline is confirmed working.
    h_bond_sweep = [4.0, 6.0]
end

# ╔═╡ f0000009-0030-0000-0000-000000000001
md"""
## Building the I-shape transfer matrix

Given a `VertexData` with `vertex::TensorMap`, extract the charge-0
block of $V_\ell$ (the only one relevant for the Neumann-0-Wilson
sector used here) as a dense array `V[T, L, R]`, and form the
transfer matrix

$$E_{(L, L'), (R, R')} \;=\; \sum_T V_{T, L, R}\, \bar V_{T, L', R'}.$$

Shape: $(D \cdot D) \times (D \cdot D)$ with $D = \dim V_L$.
"""

# ╔═╡ f0000009-0031-0000-0000-000000000001
function extract_V_charge0(vd::CFTTruncation.VertexData)
    V = vd.vertex
    bond = vd.cft.basis_bond
    phys = vd.cft.basis_phys
    d_bond = length(bond.states[0])
    d_phys = length(phys.states[0])
    W = zeros(Float64, d_phys, d_bond, d_bond)
    for (f1, f2) in fusiontrees(V)
        all(Int(f2.uncoupled[i].charge) == 0 for i in 1:3) || continue
        blk = V[f1, f2]
        @assert size(blk) == (d_phys, d_bond, d_bond)
        W .+= blk
    end
    W, d_bond, d_phys
end

# ╔═╡ f0000009-0032-0000-0000-000000000001
"""
Build the I-shape transfer matrix E as a D²×D² matrix. The row index
groups (L, L') and the column index groups (R, R'), each as
(a, b) -> a + (b-1)*D. The physical (V_T) leg is summed.
"""
function build_transfer_matrix(W::Array{Float64,3})
    d_phys, D, _ = size(W)
    E = zeros(Float64, D*D, D*D)
    @inbounds for aL in 1:D, aLp in 1:D, aR in 1:D, aRp in 1:D
        s = 0.0
        for aT in 1:d_phys
            s += W[aT, aL, aR] * W[aT, aLp, aRp]
        end
        row = aL + (aLp-1)*D
        col = aR + (aRp-1)*D
        E[row, col] = s
    end
    E
end

# ╔═╡ f0000009-0040-0000-0000-000000000001
md"""
## Shortcut sanity check (SVD only)

Before the full canonical-form EE, do the cheap thing first: SVD the
I-shape directly, use $p_i = \sigma_i^2 / \sum \sigma_j^2$ and compute
$-\sum p_i \log p_i$. Not the proper EE, but should grow logarithmically
with $h_{\text{bond}}$ if the vertex is CFT-like.
"""

# ╔═╡ f0000009-0041-0000-0000-000000000001
function shortcut_entropy(E::Matrix{Float64}; keep=nothing)
    σ = svdvals(E)
    keep === nothing || (σ = σ[1:min(keep, length(σ))])
    p = σ.^2
    p ./= sum(p)
    p = p[p .> 1e-16]
    -sum(p .* log.(p)), σ
end

# ╔═╡ f0000009-0050-0000-0000-000000000001
md"""
## Correlation length

Top eigenvalues of $E$ (non-Hermitian in general). After normalising
$\lambda_0 = 1$, the correlation length is $\xi = -1 / \log |\lambda_1|$.
"""

# ╔═╡ f0000009-0051-0000-0000-000000000001
function correlation_length(E::Matrix{Float64}; n_eigs=6)
    λ = eigvals(E)
    # Sort by magnitude descending.
    order = sortperm(abs.(λ); rev=true)
    λs = λ[order]
    λ0 = λs[1]
    λ1 = λs[2]
    λs_normalised = λs ./ abs(λ0)
    ξ = -1.0 / log(abs(λ1) / abs(λ0))
    ξ, λs_normalised[1:min(n_eigs, length(λs_normalised))]
end

# ╔═╡ f0000009-0060-0000-0000-000000000001
md"""
## Entanglement entropy via canonical form

Full recipe (Vanderstraeten §2.1):

1. Left dominant eigenvector $l$ of $E$ (as a $D \times D$ matrix on
   the left-bond double space).
2. Right dominant eigenvector $r$.
3. Factor $l = L^\dagger L$ (symmetric square root of the positive
   Hermitian matrix); analogously $r = R R^\dagger$.
4. $C = L R$; SVD $C$ to get Schmidt values $C_i$.
5. $S = -\sum C_i^2 \log C_i^2$.

The dominant eigenvectors, when reshaped as $D \times D$ matrices, are
the "environment" matrices $l, r$ of the MPS.
"""

# ╔═╡ f0000009-0061-0000-0000-000000000001
function dominant_eigenvector(M::Matrix{Float64}; right=true)
    # Find eigenvector corresponding to the largest-magnitude eigenvalue.
    F = eigen(right ? M : transpose(M))
    i = argmax(abs.(F.values))
    λ = F.values[i]
    v = F.vectors[:, i]
    # Return real and Hermitised (for the positive-fixed-point use case).
    # Caller is responsible for reshaping v into D×D and Hermitising.
    λ, v
end

# ╔═╡ f0000009-0062-0000-0000-000000000001
function hermitise_and_positive(v::Vector, D::Int)
    M = reshape(v, D, D)
    M = (M + M') / 2
    # Fix overall sign/phase: make trace positive.
    if real(tr(M)) < 0
        M = -M
    end
    # Project onto positive part (clip tiny negative eigs).
    F = eigen(Hermitian(real.(M)))
    vals = max.(F.values, 0.0)
    F.vectors * Diagonal(vals) * F.vectors'
end

# ╔═╡ f0000009-0063-0000-0000-000000000001
function canonical_C(E::Matrix{Float64}, D::Int)
    @assert size(E, 1) == D*D
    λR, vR = dominant_eigenvector(E; right=true)
    λL, vL = dominant_eigenvector(E; right=false)
    @assert isapprox(λR, λL; atol=1e-8) "left and right dominant eigenvalues should match"
    r = hermitise_and_positive(vR, D)
    l = hermitise_and_positive(vL, D)
    # Normalise so tr(l r) = 1 (standard MPS normalisation).
    norm_factor = real(tr(l * r))
    r ./= norm_factor
    # Square roots. l = L^†L with L upper triangular (Cholesky),
    # r = R R^† with R lower triangular; use the symmetric square-root
    # form here for simplicity.
    L = sqrt(Hermitian(l))
    R = sqrt(Hermitian(r))
    C = L * R
    C, λR
end

# ╔═╡ f0000009-0064-0000-0000-000000000001
function ee_from_C(C::Matrix{Float64})
    σ = svdvals(C)
    p = σ.^2
    p ./= sum(p)
    p = p[p .> 1e-16]
    -sum(p .* log.(p)), σ
end

# ╔═╡ f0000009-0070-0000-0000-000000000001
md"""
## Single-point computation

Build the pipeline for one $(h_{\text{bond}}, h_{\text{phys}})$ pair.
Useful for sanity-checking before the sweep.
"""

# ╔═╡ f0000009-0071-0000-0000-000000000001
function phase1_at(ell::Float64, h_bond::Float64, h_phys::Float64)
    cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(h_bond=h_bond, h_phys=h_phys))
    vd = compute_vertex(cft, ell; cache=:auto)
    W, D, d_T = extract_V_charge0(vd)
    E = build_transfer_matrix(W)
    ξ, λ_top = correlation_length(E)
    S_short, σ_E = shortcut_entropy(E)
    C, λ0 = canonical_C(E, D)
    S, σ_C = ee_from_C(C)
    (; h_bond, h_phys, D, d_T, ξ, λ_top, S, S_short, σ_E, σ_C, λ0)
end

# ╔═╡ f0000009-0072-0000-0000-000000000001
# Single-point sanity run.
single = phase1_at(ell, 6.0, h_phys)

# ╔═╡ f0000009-0080-0000-0000-000000000001
md"""
## Sweep over $h_{\text{bond}}$

For each $h_{\text{bond}}$ in the sweep, compute $\xi_D$ and $S_D$.
"""

# ╔═╡ f0000009-0081-0000-0000-000000000001
sweep = [phase1_at(ell, h, h_phys) for h in h_bond_sweep]

# ╔═╡ f0000009-0082-0000-0000-000000000001
let
    lines = ["h_bond    D       ξ_D          S_D       S_shortcut    top-|λ|s"]
    push!(lines, repeat("-", 80))
    for r in sweep
        top = join([@sprintf("%.4f", abs(λ)) for λ in r.λ_top[1:min(4, end)]], ", ")
        push!(lines, @sprintf("  %4.1f    %3d   %10.4f   %7.4f   %7.4f     [%s]",
            r.h_bond, r.D, r.ξ, r.S, r.S_short, top))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000009-0090-0000-0000-000000000001
md"""
## Fit: $S_D$ vs $\log \xi_D$

Linear fit slope → $c/6$. Target: slope $\approx 1/6$ (i.e. $c = 1$).
"""

# ╔═╡ f0000009-0091-0000-0000-000000000001
let
    logξ = [log(r.ξ) for r in sweep]
    Ss = [r.S for r in sweep]
    # Simple least-squares slope.
    n = length(logξ)
    x̄, ȳ = sum(logξ)/n, sum(Ss)/n
    slope = sum((logξ .- x̄) .* (Ss .- ȳ)) / sum((logξ .- x̄).^2)
    intercept = ȳ - slope * x̄
    c_est = 6 * slope
    @sprintf("slope = %.4f  →  c_est = 6·slope = %.4f  (target: c = 1)\nintercept = %.4f",
             slope, c_est, intercept)
end

# ╔═╡ f0000009-0092-0000-0000-000000000001
let
    logξ = [log(r.ξ) for r in sweep]
    Ss = [r.S for r in sweep]
    S_shorts = [r.S_short for r in sweep]
    plt = plot(logξ, Ss;
        marker=:o, label="S (canonical C)",
        xlabel="log ξ_D", ylabel="S_D",
        title="Finite-entanglement scaling — ℓ = $ell")
    plot!(plt, logξ, S_shorts; marker=:x, linestyle=:dash, label="S (shortcut SVD)")
    # Reference line with slope 1/6.
    slope_ref = 1/6
    x0, y0 = logξ[1], Ss[1]
    plot!(plt, logξ, y0 .+ slope_ref .* (logξ .- x0); linestyle=:dot,
          label="slope 1/6 (c=1)")
    plt
end

# ╔═╡ f0000009-0100-0000-0000-000000000001
md"""
## Diagnostics

- Top eigenvalues of $E$ on a log scale (should show power-law-like
  decay for a CFT MPS).
- Schmidt spectrum $C_i^2$ on a log scale.
"""

# ╔═╡ f0000009-0101-0000-0000-000000000001
let
    plts = []
    for r in sweep
        λs = abs.(r.λ_top)
        push!(plts, plot(1:length(λs), λs;
            marker=:o, yscale=:log10,
            xlabel="i", ylabel="|λ_i|",
            title="h_bond = $(r.h_bond)", legend=false))
    end
    plot(plts...; layout=(1, length(plts)), size=(900, 250))
end

# ╔═╡ f0000009-0102-0000-0000-000000000001
let
    plts = []
    for r in sweep
        p = r.σ_C.^2 ./ sum(r.σ_C.^2)
        p = p[p .> 1e-16]
        push!(plts, plot(1:length(p), p;
            marker=:o, yscale=:log10,
            xlabel="i", ylabel="C_i²",
            title="h_bond = $(r.h_bond)", legend=false))
    end
    plot(plts...; layout=(1, length(plts)), size=(900, 250))
end

# ╔═╡ Cell order:
# ╠═f0000009-0001-0000-0000-000000000001
# ╠═f0000009-0002-0000-0000-000000000001
# ╠═f0000009-0003-0000-0000-000000000001
# ╠═f0000009-0004-0000-0000-000000000001
# ╠═f0000009-0005-0000-0000-000000000001
# ╠═f0000009-0006-0000-0000-000000000001
# ╟─f0000009-0010-0000-0000-000000000001
# ╟─f0000009-0020-0000-0000-000000000001
# ╠═f0000009-0021-0000-0000-000000000001
# ╟─f0000009-0030-0000-0000-000000000001
# ╠═f0000009-0031-0000-0000-000000000001
# ╠═f0000009-0032-0000-0000-000000000001
# ╟─f0000009-0040-0000-0000-000000000001
# ╠═f0000009-0041-0000-0000-000000000001
# ╟─f0000009-0050-0000-0000-000000000001
# ╠═f0000009-0051-0000-0000-000000000001
# ╟─f0000009-0060-0000-0000-000000000001
# ╠═f0000009-0061-0000-0000-000000000001
# ╠═f0000009-0062-0000-0000-000000000001
# ╠═f0000009-0063-0000-0000-000000000001
# ╠═f0000009-0064-0000-0000-000000000001
# ╟─f0000009-0070-0000-0000-000000000001
# ╠═f0000009-0071-0000-0000-000000000001
# ╠═f0000009-0072-0000-0000-000000000001
# ╟─f0000009-0080-0000-0000-000000000001
# ╠═f0000009-0081-0000-0000-000000000001
# ╠═f0000009-0082-0000-0000-000000000001
# ╟─f0000009-0090-0000-0000-000000000001
# ╠═f0000009-0091-0000-0000-000000000001
# ╠═f0000009-0092-0000-0000-000000000001
# ╟─f0000009-0100-0000-0000-000000000001
# ╠═f0000009-0101-0000-0000-000000000001
# ╠═f0000009-0102-0000-0000-000000000001
