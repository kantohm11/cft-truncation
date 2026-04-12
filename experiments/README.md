# Experiments

Exploratory notebooks and scripts for the CFT truncation project. The library
itself lives in `../src/` and is tested in `../test/`; this directory is for
running computations and investigating results.

## Layout

```
experiments/
├── README.md          (this file)
├── notebooks/         Pluto notebooks (.jl, plain text, versioned)
├── scripts/           headless Julia scripts for long runs
└── results/           output data and figures — GITIGNORED
```

- `notebooks/` — interactive exploration. Pluto stores notebooks as plain `.jl`
  files, which makes them git-friendly out of the box. Numbered prefixes
  (`01_`, `02_`, ...) keep them sorted by chronology of the investigation.
- `scripts/` — for headless / batch runs that don't need a UI (e.g., long
  sweeps over many ℓ values). Scripts write output to `results/`.
- `results/` — plots, data dumps, exported notebook HTML. Gitignored except
  for `.gitkeep`. If a figure becomes canonical (used in a paper/slides),
  move it to `../docs/` and commit it explicitly.

## Launching a notebook

From the repository root:

```bash
julia --project=. -e 'using Pluto; Pluto.run(notebook="experiments/notebooks/01_smoke_test.jl")'
```

This opens Pluto in your browser, activates the library's `Project.toml`, and
runs the notebook. All `using CFTTruncation` calls resolve to the checked-out
library code.

## Exporting a notebook to static HTML

For sharing a completed run without requiring the reader to have Julia
installed:

```bash
julia -e '
using Pkg; Pkg.activate(temp=true); Pkg.add("PlutoSliderServer")
using PlutoSliderServer
PlutoSliderServer.export_notebook(
    "experiments/notebooks/01_smoke_test.jl";
    Export_output_dir = "experiments/results",
)'
```

This produces `experiments/results/01_smoke_test.html` with cell outputs
baked into the HTML (no live Julia process needed to view it). The HTML is
gitignored; if you want to share it, email/upload it directly or commit it
under `../docs/` if it's a canonical result.

## Current notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_smoke_test.jl` | Mock — single `compute_vertex` call to confirm the experiment plumbing (project activation, package import, layered API) works end-to-end. No science. |

## Conventions

- One notebook per investigation; don't pack multiple experiments into one file.
- Start each notebook with project activation:
  ```julia
  import Pkg
  Pkg.activate(joinpath(@__DIR__, "..", ".."))
  ```
- Write output data and figures to `experiments/results/<notebook_name>/` (create the
  subdir if needed). Never commit these files.
- If an experiment needs a new dep not in the library's `Project.toml`
  (e.g., a plotting library), we may eventually split experiments into their
  own nested environment (`experiments/Project.toml`). Until then, add the
  dep to the shared `../Project.toml`.
