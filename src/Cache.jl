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

# ----------------------------------------------------------------
# Version tag — embedded in cache filenames for invalidation.
#
# BUMP CACHE_VERSION when changing any of:
#   - _compute_vertex_raw, _recurse_entry, _apply_Jk_on_arm_sparse (Recursion.jl)
#   - compute_geometry (LocalCoordinates.jl)
#   - compute_neumann (NeumannCoefficients.jl)
#   - primary_vertex (PrimaryVertex.jl)
#   - BPZ conventions (BPZ.jl)
#   - FockBasis state ordering or normalization (FockSpace.jl)
#   - modified_vertex propagator factor formula
# ----------------------------------------------------------------
const CACHE_VERSION = "v9_primary_vertex_jacobian_fix"
# History:
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

function _cache_path(cft::CompactBosonCFT, ell::Float64, series_order::Int;
                     shape::Symbol = :T)
    dir = _CACHE_DIR[]
    dir === nothing && error(
        "Cache directory not set. Call CFTTruncation.set_cache_dir(\"path\") first.")
    R = cft.R; hb = cft.trunc.h_bond; hp = cft.trunc.h_phys
    shape_tag = String(shape)
    joinpath(dir,
        "vertex_$(CACHE_VERSION)_$(shape_tag)_R$(R)_hb$(hb)_hp$(hp)_ell$(ell)_so$(series_order).jld2")
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
