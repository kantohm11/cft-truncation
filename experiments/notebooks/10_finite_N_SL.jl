### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ f0000010-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ f0000010-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ f0000010-0003-0000-0000-000000000001
using TensorKit

# ╔═╡ f0000010-0004-0000-0000-000000000001
using LinearAlgebra

# ╔═╡ f0000010-0005-0000-0000-000000000001
using Plots

# ╔═╡ f0000010-0006-0000-0000-000000000001
using Printf

# ╔═╡ f0000010-0010-0000-0000-000000000001
md"""
# 10 — Finite-N $S(L)$ from $V_\ell$ tiled as an MPS

Phase 1 (notebook 09) found that the half-infinite canonical EE of
$V_\ell$ tiled as a uniform MPS is tiny (~10⁻³). That's expected:
in the $N \to \infty$ limit with fixed ℓ, the tiled vertex is the open
boundary state $|B\rangle$ on a half-plane — a product state — zero
entanglement on any bond.

At **finite $N$** (cylinder of circumference $N\ell$), a subsystem of
$L$ out of $N$ sites carries non-trivial geometric EE. CFT prediction
for PBC:

$$S(L; N) \;=\; \frac{c}{3} \log\!\left[\frac{N}{\pi}\sin\!\frac{\pi L}{N}\right] + \text{const}.$$

With $c = 1$ (compact boson) this is our benchmark.

The prototype at ℓ=0.1, no stab (notebook 09) already showed non-zero
finite-N EE (~10⁻³) decreasing with $N$. Here we push $L$ up enough
to see the log-scaling, for two ℓ values.

## Method

In the uniform-MPS picture, $A^s_{i,j} = V_{s, i, j}$ (physical leg =
$V_T$, bonds = $V_L, V_R$). For PBC on $N$ sites, the reduced density
matrix on $L$ contiguous sites is

$$\rho(L)_{s, s'} \;=\; \frac{1}{Z_N}\,\operatorname{tr}\!\bigl[E^{N-L} \cdot (A^{s_1} \cdots A^{s_L}) \otimes (A^{s_1'} \cdots A^{s_L'})^*\bigr]$$

with $Z_N = \operatorname{tr}(E^N)$ and $E = \sum_s A^s \otimes (A^s)^*$
the MPS transfer matrix.

## Cost note

The naive implementation dense-diagonalises a $d_T^L \times d_T^L$
matrix. At $h_\text{phys} = 3$ this gives $d_T = 7$: L ≤ 4 takes
seconds; L = 5 ~ an hour; L ≥ 6 infeasible. An $O(D^6)$ rank-$D^2$
algorithm exists (the interval's Schmidt rank is bounded by $D^2$
across the PBC cut), not implemented here.
"""

# ╔═╡ f0000010-0020-0000-0000-000000000001
md"""
## Setup
"""

# ╔═╡ f0000010-0021-0000-0000-000000000001
begin
    R_val = 1.0
    h_phys_val = 3.0  # d_T = 7: keeps d_T^L tractable up to L = 4
    h_bond_val = 6.0
    ells = [0.01, 0.1]
end

# ╔═╡ f0000010-0030-0000-0000-000000000001
md"""
## Extracting the charge-0 block of $V_\ell$
"""

# ╔═╡ f0000010-0031-0000-0000-000000000001
function extract_W(vd::CFTTruncation.VertexData)
    V = vd.vertex
    bond = vd.cft.basis_bond
    phys = vd.cft.basis_phys
    D = length(bond.states[0])
    d_T = length(phys.states[0])
    W = zeros(Float64, d_T, D, D)
    for (f1, f2) in fusiontrees(V)
        all(Int(f2.uncoupled[i].charge) == 0 for i in 1:3) || continue
        W .+= V[f1, f2]
    end
    W, D, d_T
end

# ╔═╡ f0000010-0040-0000-0000-000000000001
md"""
## MPS transfer matrix $E$
"""

# ╔═╡ f0000010-0041-0000-0000-000000000001
function build_E(A::Array{Float64,3})
    d, D, _ = size(A)
    E = zeros(Float64, D*D, D*D)
    @inbounds for aL in 1:D, aLp in 1:D, aR in 1:D, aRp in 1:D
        s = 0.0
        for aT in 1:d
            s += A[aT, aL, aR] * A[aT, aLp, aRp]
        end
        E[aL + (aLp-1)*D, aR + (aRp-1)*D] = s
    end
    E
end

# ╔═╡ f0000010-0050-0000-0000-000000000001
md"""
## Reduced density matrix on L sites for PBC chain of length N

Vectorised version — two reshape+matmul passes — fast for (L, N)
ranges of interest.
"""

# ╔═╡ f0000010-0051-0000-0000-000000000001
function rho_L_PBC(A::Array{Float64,3}, L::Int, N::Int)
    d, D, _ = size(A)
    @assert L ≥ 1 && L < N "need 1 ≤ L < N"
    E = build_E(A)
    E_NmL = E^(N - L)
    Z_N = real(tr(E^N))
    # Build all L-site products M[s_flat, aL, aR] = (A^{s_1} ... A^{s_L})_{aL, aR}
    Ndim = d^L
    M = zeros(Float64, Ndim, D, D)
    idx = zeros(Int, L)
    for s_flat in 1:Ndim
        k = s_flat - 1
        for l in 1:L
            idx[l] = (k % d) + 1
            k ÷= d
        end
        mat = Matrix{Float64}(I, D, D)
        for l in 1:L
            mat = mat * A[idx[l], :, :]
        end
        M[s_flat, :, :] = mat
    end
    # ρ[s, s'] = Σ_{aL, aLp, aR, aRp} E_NmL[(aR, aRp), (aL, aLp)]
    #                                   · M[s, aL, aR] · M[s', aLp, aRp]  / Z_N
    # Reshape E_NmL to (aR, aRp, aL, aLp), then permute to (aL, aR, aRp, aLp).
    Et = reshape(E_NmL, D, D, D, D)          # (aR, aRp, aL, aLp)
    Etp = permutedims(Et, (3, 1, 2, 4))      # (aL, aR, aRp, aLp)
    Mflat = reshape(M, Ndim, D * D)          # M[s, (aL, aR)]
    Etpflat = reshape(Etp, D * D, D * D)     # (aL, aR) × (aRp, aLp)
    X = Mflat * Etpflat                       # X[s, (aRp, aLp)]
    X3 = reshape(X, Ndim, D, D)              # X[s, aRp, aLp]
    Xperm = permutedims(X3, (1, 3, 2))       # X[s, aLp, aRp]
    ρ = reshape(Xperm, Ndim, D * D) * reshape(M, Ndim, D * D)'  # (s, sp)
    ρ ./= Z_N
    ρ
end

# ╔═╡ f0000010-0060-0000-0000-000000000001
md"""
## Entanglement entropy from $\rho(L)$
"""

# ╔═╡ f0000010-0061-0000-0000-000000000001
function ee_from_rho(ρ::Matrix{Float64})
    vals = eigvals(Hermitian((ρ + ρ') / 2))
    vals = sort(real.(vals); rev=true)
    trace = sum(vals)
    if trace ≤ 0
        return NaN, vals[1:min(6, end)]
    end
    p = vals ./ trace
    p = p[p .> 1e-14]
    S = -sum(p .* log.(p))
    S, vals[1:min(6, end)]
end

# ╔═╡ f0000010-0070-0000-0000-000000000001
md"""
## Single-point harness

Compute $S(L; N)$ at one $(L, N)$ for a given ℓ. Useful for sanity
checks and for the full sweep.
"""

# ╔═╡ f0000010-0071-0000-0000-000000000001
function SL_at(ell::Float64, h_bond::Float64, h_phys::Float64, L::Int, N::Int)
    cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(h_bond=h_bond, h_phys=h_phys))
    vd = compute_vertex(cft, ell; cache=:auto)
    A, D, d_T = extract_W(vd)
    ρ = rho_L_PBC(A, L, N)
    S, top = ee_from_rho(ρ)
    (; ell, h_bond, h_phys, L, N, D, d_T, S, top_eigs=top)
end

# ╔═╡ f0000010-0072-0000-0000-000000000001
single = SL_at(0.1, h_bond_val, h_phys_val, 2, 8)

# ╔═╡ f0000010-0080-0000-0000-000000000001
md"""
## (L, N) sweep at both ℓ values
"""

# ╔═╡ f0000010-0081-0000-0000-000000000001
begin
    Ns = [6, 8, 10, 12, 16]
    Lmax = 4
end

# ╔═╡ f0000010-0082-0000-0000-000000000001
function run_sweep(ell::Float64, h_bond::Float64, h_phys::Float64, Ns::Vector{Int}, Lmax::Int)
    cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(h_bond=h_bond, h_phys=h_phys))
    vd = compute_vertex(cft, ell; cache=:auto)
    A, D, d_T = extract_W(vd)
    results = Dict{Tuple{Int,Int}, Float64}()
    for N in Ns, L in 1:min(Lmax, N - 1)
        ρ = rho_L_PBC(A, L, N)
        S, _ = ee_from_rho(ρ)
        results[(L, N)] = S
    end
    (; ell, D, d_T, results)
end

# ╔═╡ f0000010-0083-0000-0000-000000000001
sweep_01 = run_sweep(0.01, h_bond_val, h_phys_val, Ns, Lmax)

# ╔═╡ f0000010-0084-0000-0000-000000000001
sweep_1 = run_sweep(0.1, h_bond_val, h_phys_val, Ns, Lmax)

# ╔═╡ f0000010-0085-0000-0000-000000000001
let
    lines = String["ℓ=0.01:"]
    push!(lines, "  L\\N" * join([@sprintf("%10d", N) for N in Ns]))
    for L in 1:Lmax
        row = @sprintf("   %d  ", L)
        for N in Ns
            haskey(sweep_01.results, (L, N)) ?
                (row *= @sprintf("%10.5f", sweep_01.results[(L, N)])) :
                (row *= "         -")
        end
        push!(lines, row)
    end
    push!(lines, "")
    push!(lines, "ℓ=0.1:")
    push!(lines, "  L\\N" * join([@sprintf("%10d", N) for N in Ns]))
    for L in 1:Lmax
        row = @sprintf("   %d  ", L)
        for N in Ns
            haskey(sweep_1.results, (L, N)) ?
                (row *= @sprintf("%10.5f", sweep_1.results[(L, N)])) :
                (row *= "         -")
        end
        push!(lines, row)
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000010-0090-0000-0000-000000000001
md"""
## CFT slope fit

For each N separately, fit $S(L)$ vs $\log[(N/\pi)\sin(\pi L/N)]$
linearly. Slope → estimate of $c/3$. Target: slope $\approx 1/3$
for $c = 1$.
"""

# ╔═╡ f0000010-0091-0000-0000-000000000001
"""CFT x-coordinate log[(N/π) sin(πL/N)] for PBC interval."""
cft_x(L::Int, N::Int) = log((N / π) * sin(π * L / N))

# ╔═╡ f0000010-0092-0000-0000-000000000001
function fit_slope_per_N(sweep, Ns, Lmax)
    rows = Tuple{Int, Float64, Float64}[]  # (N, slope, intercept)
    for N in Ns
        xs = Float64[]; ys = Float64[]
        for L in 1:min(Lmax, N - 1)
            S = get(sweep.results, (L, N), NaN)
            isnan(S) && continue
            push!(xs, cft_x(L, N))
            push!(ys, S)
        end
        length(xs) < 2 && continue
        x̄, ȳ = sum(xs)/length(xs), sum(ys)/length(ys)
        slope = sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄).^2)
        intercept = ȳ - slope * x̄
        push!(rows, (N, slope, intercept))
    end
    rows
end

# ╔═╡ f0000010-0093-0000-0000-000000000001
let
    lines = ["ℓ=0.01 slopes (target 1/3 for c=1):"]
    push!(lines, @sprintf("  %-4s %-12s %-12s %-12s", "N", "slope", "6·slope (=c?)", "intercept"))
    for (N, slope, inter) in fit_slope_per_N(sweep_01, Ns, Lmax)
        push!(lines, @sprintf("  %-4d %-12.4f %-12.4f %-12.4f", N, slope, 3*slope, inter))
    end
    push!(lines, "")
    push!(lines, "ℓ=0.1 slopes:")
    push!(lines, @sprintf("  %-4s %-12s %-12s %-12s", "N", "slope", "3·slope (=c?)", "intercept"))
    for (N, slope, inter) in fit_slope_per_N(sweep_1, Ns, Lmax)
        push!(lines, @sprintf("  %-4d %-12.4f %-12.4f %-12.4f", N, slope, 3*slope, inter))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000010-0100-0000-0000-000000000001
md"""
## Plots: $S(L; N)$ vs CFT curve

Left: raw $S(L; N)$ grouped by $N$, $L$ on x-axis.
Right: $S$ vs $\log[(N/\pi)\sin(\pi L/N)]$ — should be linear with
slope $c/3$ if CFT scaling holds.
"""

# ╔═╡ f0000010-0101-0000-0000-000000000001
let
    plts = []
    for (sweep, name) in ((sweep_01, "ℓ=0.01"), (sweep_1, "ℓ=0.1"))
        p1 = plot(;
            xlabel="L", ylabel="S(L; N)", title="$name : S vs L",
            legend=:topleft, size=(400, 320))
        for N in Ns
            xs = Int[]; ys = Float64[]
            for L in 1:min(Lmax, N - 1)
                haskey(sweep.results, (L, N)) || continue
                push!(xs, L); push!(ys, sweep.results[(L, N)])
            end
            length(xs) ≥ 1 && plot!(p1, xs, ys; marker=:o, label="N=$N")
        end
        push!(plts, p1)
    end
    plot(plts...; layout=(1, 2), size=(900, 360))
end

# ╔═╡ f0000010-0102-0000-0000-000000000001
let
    plts = []
    for (sweep, name) in ((sweep_01, "ℓ=0.01"), (sweep_1, "ℓ=0.1"))
        p = plot(;
            xlabel="log[(N/π) sin(πL/N)]",
            ylabel="S(L; N)",
            title="$name : CFT x-axis",
            legend=:topleft, size=(400, 320))
        for N in Ns
            xs = Float64[]; ys = Float64[]
            for L in 1:min(Lmax, N - 1)
                haskey(sweep.results, (L, N)) || continue
                push!(xs, cft_x(L, N)); push!(ys, sweep.results[(L, N)])
            end
            length(xs) ≥ 1 && plot!(p, xs, ys; marker=:o, label="N=$N")
        end
        # Reference line with slope 1/3
        x_range = [-2.0, 1.5]
        # Use ℓ=0.1, N=12 as an anchor for intercept if available.
        anchor = get(sweep.results, (2, 12), nothing)
        if anchor !== nothing
            x0 = cft_x(2, 12); y0 = anchor
            plot!(p, x_range, y0 .+ (1/3) .* (x_range .- x0);
                  linestyle=:dash, color=:black, label="slope 1/3 (c=1)")
        end
        push!(plts, p)
    end
    plot(plts...; layout=(1, 2), size=(900, 360))
end

# ╔═╡ Cell order:
# ╠═f0000010-0001-0000-0000-000000000001
# ╠═f0000010-0002-0000-0000-000000000001
# ╠═f0000010-0003-0000-0000-000000000001
# ╠═f0000010-0004-0000-0000-000000000001
# ╠═f0000010-0005-0000-0000-000000000001
# ╠═f0000010-0006-0000-0000-000000000001
# ╟─f0000010-0010-0000-0000-000000000001
# ╟─f0000010-0020-0000-0000-000000000001
# ╠═f0000010-0021-0000-0000-000000000001
# ╟─f0000010-0030-0000-0000-000000000001
# ╠═f0000010-0031-0000-0000-000000000001
# ╟─f0000010-0040-0000-0000-000000000001
# ╠═f0000010-0041-0000-0000-000000000001
# ╟─f0000010-0050-0000-0000-000000000001
# ╠═f0000010-0051-0000-0000-000000000001
# ╟─f0000010-0060-0000-0000-000000000001
# ╠═f0000010-0061-0000-0000-000000000001
# ╟─f0000010-0070-0000-0000-000000000001
# ╠═f0000010-0071-0000-0000-000000000001
# ╠═f0000010-0072-0000-0000-000000000001
# ╟─f0000010-0080-0000-0000-000000000001
# ╠═f0000010-0081-0000-0000-000000000001
# ╠═f0000010-0082-0000-0000-000000000001
# ╠═f0000010-0083-0000-0000-000000000001
# ╠═f0000010-0084-0000-0000-000000000001
# ╠═f0000010-0085-0000-0000-000000000001
# ╟─f0000010-0090-0000-0000-000000000001
# ╠═f0000010-0091-0000-0000-000000000001
# ╠═f0000010-0092-0000-0000-000000000001
# ╠═f0000010-0093-0000-0000-000000000001
# ╟─f0000010-0100-0000-0000-000000000001
# ╠═f0000010-0101-0000-0000-000000000001
# ╠═f0000010-0102-0000-0000-000000000001
