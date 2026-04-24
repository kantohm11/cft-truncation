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
const CACHE_VERSION = "v5_rho0_uhp_upper_semidisc"
# History:
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

function _cache_path(cft::CompactBosonCFT, ell::Float64, series_order::Int)
    dir = _CACHE_DIR[]
    dir === nothing && error(
        "Cache directory not set. Call CFTTruncation.set_cache_dir(\"path\") first.")
    R = cft.R; hb = cft.trunc.h_bond; hp = cft.trunc.h_phys
    joinpath(dir,
        "vertex_$(CACHE_VERSION)_R$(R)_hb$(hb)_hp$(hp)_ell$(ell)_so$(series_order).jld2")
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
