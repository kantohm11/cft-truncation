#!/usr/bin/env julia
# |α_i|(ℓ) for the T-shape arms, with primary suppression factors overlaid.
#
# Convention check (against PrimaryVertex.jl line 36):
#   primary vertex Jacobian = ∏ |α_i|^{2 h_i}, with h_i = (n_i/R)²/2.
#   So the ℓ-dependent suppression factor for an n-charge primary on
#   arm i is |α_i(ℓ)|^{n²/R²}.
# α_i is the leading coefficient of f_i(ζ) = α_i·ζ + O(ζ²), where
#   ζ = z − x_i. NOT the inverse: g_i'(0) = 1/α_i.

using CFTTruncation
using Plots
using Printf

ells = exp10.(range(-1.5, 1.0; length=50))   # 0.03 → 10
αs = Dict{Symbol, Vector{Float64}}(:L => Float64[], :R => Float64[], :T => Float64[])
for ℓ in ells
    geom = CFTTruncation.compute_geometry(ℓ, 25)
    push!(αs[:L], abs(geom.arms.L.α))
    push!(αs[:R], abs(geom.arms.R.α))
    push!(αs[:T], abs(geom.arms.T.α))
end

# Print at a few key points.
println(@sprintf("%6s | %12s %12s %12s", "ℓ", "|α_L|", "|α_R|", "|α_T|"))
for ℓ_target in (0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
    i = argmin(abs.(ells .- ℓ_target))
    println(@sprintf("%6.3f | %12.4e %12.4e %12.4e",
        ells[i], αs[:L][i], αs[:R][i], αs[:T][i]))
end

# Plot raw |α_i|(ℓ).
p1 = plot(ells, αs[:L]; xscale=:log10, yscale=:log10, label="|α_L|",
    xlabel="ℓ", ylabel="|α_i|", lw=2, color=:blue,
    title="T-shape Jacobians vs ℓ", legend=:topleft)
plot!(p1, ells, αs[:R]; label="|α_R|", lw=2, color=:red, linestyle=:dash)
plot!(p1, ells, αs[:T]; label="|α_T|", lw=2, color=:green)

# Overlay suppression factors |α_i|^{2h_n} = |α_i|^{n²/R²} at R=1.
# n=1: exponent 1; n=2: exponent 4.
function suppression(αs_arm::Vector{Float64}, n::Int, R::Real=1.0)
    [α^(n^2 / R^2) for α in αs_arm]
end

p2 = plot(; xscale=:log10, yscale=:log10,
    xlabel="ℓ", ylabel="|α_i|^{n²/R²} = primary suppression",
    title="Primary suppression on each arm (R=1)",
    legend=:topleft, ylims=(1e-3, 10))
for (i, arm) in enumerate((:L, :R, :T))
    color = (:blue, :red, :green)[i]
    plot!(p2, ells, suppression(αs[arm], 1); label="$(arm), n=1", lw=2, color=color, linestyle=:solid)
    plot!(p2, ells, suppression(αs[arm], 2); label="$(arm), n=2", lw=1.5, color=color, linestyle=:dash)
end

plt = plot(p1, p2, layout=(1, 2), size=(1100, 400))
savefig(plt, "/Users/kantaro/repositories/cft-truncation/docs/design/figures/alpha_vs_ell.png")
println("\nSaved to docs/design/figures/alpha_vs_ell.png")
