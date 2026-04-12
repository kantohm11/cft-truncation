### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 7c4b2a00-0002-0000-0000-000000000002
begin
    import Pkg
    # Activate the library's project so `using CFTTruncation` resolves to the
    # version checked out in this repo.
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 7c4b2a00-0003-0000-0000-000000000003
using CFTTruncation

# ╔═╡ 7c4b2a00-0004-0000-0000-000000000004
using TensorKit: dim, domain

# ╔═╡ 7c4b2a00-0005-0000-0000-000000000005
using LinearAlgebra: norm

# ╔═╡ 7c4b2a00-0001-0000-0000-000000000001
md"""
# 01 — Smoke Test

This notebook is a **mock** — its only job is to confirm the experiment plumbing works.
No sweep, no plot, no science. It makes one basic API call against the layered
`compute_vertex(cft, ell)` form and prints a few sanity numbers.

If this notebook re-runs cleanly in a fresh Julia session, the rest of the experiment
machinery (project activation, package import, layered API call, notebook-as-text
storage) is good to go.
"""

# ╔═╡ 7c4b2a00-0006-0000-0000-000000000006
md"""
## Build the CFT data once

`CompactBosonCFT` bundles the ℓ-independent state spaces and mode action.
Build it once; reuse across as many ℓ values as you like.
"""

# ╔═╡ 7c4b2a00-0007-0000-0000-000000000007
cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))

# ╔═╡ 7c4b2a00-0008-0000-0000-000000000008
(R=cft.R, h_bond=cft.trunc.h_bond, h_phys=cft.trunc.h_phys, m_max=cft.m_max,
 dim_bond=dim(cft.basis_bond.V), dim_phys=dim(cft.basis_phys.V))

# ╔═╡ 7c4b2a00-0009-0000-0000-000000000009
md"""
## Compute one vertex

Single call to the layered form `compute_vertex(cft, ell)`.
"""

# ╔═╡ 7c4b2a00-000a-0000-0000-00000000000a
vd = compute_vertex(cft, 1.0)

# ╔═╡ 7c4b2a00-000b-0000-0000-00000000000b
(ell=vd.ell, vertex_codomain=string(codomain(vd.vertex)),
 vertex_domain=string(domain(vd.vertex)),
 vertex_norm=norm(vd.vertex),
 raw_entries=length(vd.raw))

# ╔═╡ 7c4b2a00-000c-0000-0000-00000000000c
md"""
## Smoke check passed if you can see numbers above

If `vertex_norm` is a finite positive number and `raw_entries` is non-zero,
the layered API call is working from inside Pluto. You can clone this notebook
to start a real experiment as `02_<topic>.jl`.
"""

# ╔═╡ Cell order:
# ╟─7c4b2a00-0001-0000-0000-000000000001
# ╠═7c4b2a00-0002-0000-0000-000000000002
# ╠═7c4b2a00-0003-0000-0000-000000000003
# ╠═7c4b2a00-0004-0000-0000-000000000004
# ╠═7c4b2a00-0005-0000-0000-000000000005
# ╟─7c4b2a00-0006-0000-0000-000000000006
# ╠═7c4b2a00-0007-0000-0000-000000000007
# ╠═7c4b2a00-0008-0000-0000-000000000008
# ╟─7c4b2a00-0009-0000-0000-000000000009
# ╠═7c4b2a00-000a-0000-0000-00000000000a
# ╠═7c4b2a00-000b-0000-0000-00000000000b
# ╟─7c4b2a00-000c-0000-0000-00000000000c
