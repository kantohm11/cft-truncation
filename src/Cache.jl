"""
On-disk caching for vertex TensorMaps.

Caching is controlled by the `cache` kwarg on `compute_vertex`:
- `:auto` (default): load from disk if cached, else compute + save.
- `:off`: no disk IO, always compute fresh.
- `:regenerate`: always compute fresh, overwrite cache file.

The cache directory must be set once per session:
    CFTTruncation.set_cache_dir("experiments/results/cache")
"""

using JLD2
using JSON
using SHA: sha256

# ----------------------------------------------------------------
# Version tag — embedded in cache filenames for invalidation.
#
# BUMP CACHE_VERSION when changing any of:
#   - _compute_vertex_raw, _recurse_entry, _apply_Jk_on_arm_sparse (VertexRecursion.jl)
#   - _compute_vertex_raw_cross, _recurse_entry_cross, _apply_Jk_on_arm_sparse_cross (VertexRecursionCross.jl)
#   - compute_geometry (LocalCoordinates.jl)
#   - compute_geometry_cross (LocalCoordinatesCross.jl)
#   - compute_neumann (NeumannCoefficients.jl)
#   - primary_vertex (PrimaryVertex.jl)
#   - BPZ conventions (BPZ.jl)
#   - FockBasis state ordering or normalization (FockSpace.jl)
#   - modified_vertex propagator factor formula (VertexProjections.jl)
#   - cache filename / sidecar layout itself
# ----------------------------------------------------------------
const CACHE_VERSION = "v11_rho0_T_simpson"
# History:
#   v11_rho0_T_simpson — T-arm ρ₀ now computed by Simpson integration with
#                         t² substitution at the √-branch endpoint + pole
#                         subtraction at z = 0, replacing the slowly-
#                         converging N=41 series truncation evaluated at
#                         |z| = R_conv = p. Accuracy ~1e−12 at npts=20000,
#                         vs ~1e−4 (small ℓ) to ~1e−5 (moderate ℓ) before.
#                         All T-shape vertex values shift at the level of
#                         the old truncation error; mostly affects high-h
#                         descendants on the T arm.
#   v10_per_charge_cutoffs_sidecar — TruncationSpec now per-charge level
#                                    cutoffs (Dict{Int,Int} for bond and
#                                    phys). Cache filename embeds an
#                                    8-char hash of the cutoffs spec, with
#                                    a JSON sidecar (one per (R, so, trunc,
#                                    shape) tuple) decoding the hash to
#                                    the full spec. Backward-compat path:
#                                    `TruncationSpec(h_bond, h_phys; R)`
#                                    constructs the per-charge cutoffs
#                                    via `uniform_cutoffs`. Old v9 files
#                                    are silently ignored.
#   v9_primary_vertex_jacobian_fix — primary_vertex parameterised by boundary
#                                     scaling dimension Δ (= 2 h_bulk for charged
#                                     primaries; = h_chiral for chiral primaries).
#                                     Jacobian rule fixed to (1/α)^Δ per leg
#                                     (was |α|^{+2 h_bulk}, opposite sign and
#                                     half-magnitude in the distance exponent).
#                                     All non-zero-charge vertex values change;
#                                     charge-0 sector unaffected. See
#                                     experiments/scripts/jacobian_jjvac_check.jl
#                                     and test/test_primary_jacobian.jl for
#                                     the convention-derivation check.
#   v8_shape_tag — cache filename now embeds a shape tag (T or cross) so
#                   the two vertex types coexist on disk without collision.
#                   This is a BREAKING change: prior on-disk cache files
#                   (v7 and below) will not be loaded — they are silently
#                   ignored, and `compute_vertex(...; cache=:auto)` will
#                   recompute and write fresh files at v8 paths.
#                   T-shape filename: vertex_v8_shape_tag_T_R..._so..jld2
#                   Cross filename:   vertex_v8_shape_tag_cross_R..._so..jld2
#   v7_rho0_cross_simpson — cross ρ₀ now computed by Simpson integration with
#                            t² substitution at the √-branch endpoint + pole
#                            subtraction at the arm-preimage endpoint, replacing
#                            the slowly-converging N=41 series truncation at
#                            |ζ|=R_conv. Accuracy ≈ 1e-7 at npts=20000,
#                            vs the previous 2e-4 truncation error. The D₄
#                            equality |N^{II}_{m,k}| is now visible at ~2e-7
#                            precision for k ≥ 1 entries at ℓ=1 (k=0 shows
#                            genuine zero-mode anomaly).
#   v6_cross_neumann — added 4-arm Neumann coefficients for cross geometry
#                       (compute_neumann(::GeometryCross, m_max) returning
#                       NeumannDataCross with 16 matrices). Refactored
#                       _compute_F_polys to take ArmData; extended
#                       _compose_Fm_with_g with two B-arm branches
#                       (B source and B target, both handling x_B = ∞ via
#                       u = 1/z). No behaviour change to the T-shape path.
#   v5_rho0_uhp_upper_semidisc — ρ₀ conventions unified so f_i maps UHP of z
#                                 to the upper semidisc of ξ on every arm:
#                                 removed +i shift from T-shape R and L;
#                                 cross R uses target_reg = −log(1−q1)/π;
#                                 cross B uses target_reg = −(iℓ/π) log(q1) − ℓ.
#                                 Reintroduces (-1)^N sign in the propagator
#                                 obtained by |B^open⟩ contraction (vs v4).
#   v4_rho0_R_corner_plus1 — ρ₀^R shifted by +i so ξ_R(p) = +1, α_R = α_L.
#                             Removes (-1)^N sign from propagator.
#   v3_rho0_unit_circle — ρ₀^T targets corners at |ξ_T| = ±1 (R_conv = 1),
#                          instead of corner-matching f_T(p) = i.
#   v2_rho0    — ρ₀ computed from SC map geometry (corner matching),
#                BPZ sign fixed to U(1) convention ∏(-1)^{k+1}.
#   v1_initial — first version, after σ convention fix, sparse J_k,
#                VertexArray optimization, @tensor modified_vertex.

# ----------------------------------------------------------------
# Cache directory (module-level global)
# ----------------------------------------------------------------
const _CACHE_DIR = Ref{Union{String,Nothing}}(nothing)

"""
    set_cache_dir(dir::String)

Set the directory for on-disk vertex caching. Called once per session.
Creates the directory if it doesn't exist.
"""
function set_cache_dir(dir::String)
    _CACHE_DIR[] = dir
    mkpath(dir)
    dir
end

# ----------------------------------------------------------------
# Truncation hashing.
#
# A vertex cache file's full identity = (R, series_order, truncation
# spec, shape, ell). We fingerprint the truncation spec into 8 hex
# chars. A JSON sidecar (named the same as the .jld2 minus _ell..., +
# .json) decodes the hash back to the full spec so files can be audited.
# ----------------------------------------------------------------

"""
    truncation_hash(trunc::TruncationSpec) -> String

8-char hex fingerprint of the (bond_cutoffs, phys_cutoffs) pair.
Stable across sessions: charge keys are sorted before serialisation.
"""
function truncation_hash(trunc::TruncationSpec)
    sorted_pairs(c) = [(string(n), c[n]) for n in sort(collect(keys(c)))]
    canonical = (; bond_cutoffs = sorted_pairs(trunc.bond_cutoffs),
                   phys_cutoffs = sorted_pairs(trunc.phys_cutoffs))
    str = JSON.json(canonical)
    bytes2hex(sha256(str))[1:8]
end

"""
    truncation_spec_dict(trunc::TruncationSpec) -> Dict

Plain-data representation of the truncation spec, suitable for
JSON serialisation. Charge keys become strings (JSON requirement).
"""
function truncation_spec_dict(trunc::TruncationSpec)
    Dict(
        "bond_cutoffs" => Dict(string(n) => trunc.bond_cutoffs[n]
                                for n in sort(collect(keys(trunc.bond_cutoffs)))),
        "phys_cutoffs" => Dict(string(n) => trunc.phys_cutoffs[n]
                                for n in sort(collect(keys(trunc.phys_cutoffs)))),
        "h_bond_eff"   => trunc.h_bond,
        "h_phys_eff"   => trunc.h_phys,
    )
end

function _cache_dir()
    dir = _CACHE_DIR[]
    dir === nothing && error(
        "Cache directory not set. Call CFTTruncation.set_cache_dir(\"path\") first.")
    dir
end

function _cache_stem(cft::CompactBosonCFT, series_order::Int, shape::Symbol)
    th = truncation_hash(cft.trunc)
    "vertex_$(CACHE_VERSION)_$(String(shape))_R$(cft.R)_so$(series_order)_t$(th)"
end

function _cache_path(cft::CompactBosonCFT, ell::Float64, series_order::Int;
                     shape::Symbol = :T)
    joinpath(_cache_dir(), "$(_cache_stem(cft, series_order, shape))_ell$(ell).jld2")
end

function _sidecar_path(cft::CompactBosonCFT, series_order::Int; shape::Symbol = :T)
    joinpath(_cache_dir(), "$(_cache_stem(cft, series_order, shape)).json")
end

"""
    write_sidecar(cft, series_order; shape) -> String

Write (or re-write) the JSON sidecar describing the truncation spec
behind a cache hash. Returns the sidecar path. Idempotent.
"""
function write_sidecar(cft::CompactBosonCFT, series_order::Int; shape::Symbol = :T)
    path = _sidecar_path(cft, series_order; shape = shape)
    payload = Dict(
        "cache_version" => CACHE_VERSION,
        "R"             => cft.R,
        "c"             => cft.c,
        "series_order"  => series_order,
        "shape"         => String(shape),
        "trunc_hash"    => truncation_hash(cft.trunc),
        "trunc"         => truncation_spec_dict(cft.trunc),
    )
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    path
end

function _load_vertex(path::String)
    @info "[cache] vertex ← $(basename(path))"
    JLD2.load(path, "vertex")
end

function _save_vertex(path::String, vertex)
    mkpath(dirname(path))
    JLD2.save(path, "vertex", vertex)
    @info "[cache] vertex → $(basename(path))"
end
