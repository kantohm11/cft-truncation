#!/usr/bin/env julia
# Export a Pluto notebook to static HTML without PlutoSliderServer.
# Much faster for non-interactive notebooks (~20s vs ~80s).
#
# Usage:
#   julia --project=. experiments/scripts/export_notebook.jl experiments/notebooks/02_truncation_convergence.jl

using Pluto

nb_path = abspath(ARGS[1])
out_dir = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(dirname(dirname(nb_path)), "results")
out_name = replace(basename(nb_path), ".jl" => ".html")
out_path = joinpath(out_dir, out_name)

println("Opening notebook: $nb_path")
ss = Pluto.ServerSession()
ss.options.evaluation.workspace_use_distributed = false  # in-process, no worker spawn
nb = Pluto.SessionActions.open(ss, nb_path; run_async=false)

# Check for errors
nerr = count(c -> c.errored, nb.cells)
if nerr > 0
    println("WARNING: $nerr cell(s) errored!")
    for (i, c) in enumerate(nb.cells)
        c.errored || continue
        println("  [ERROR cell $i] $(first(split(c.code, '\n')))")
    end
end

println("Generating HTML...")
html = Pluto.generate_html(nb)
mkpath(out_dir)
write(out_path, html)
println("Written: $out_path ($(filesize(out_path)) bytes)")

Pluto.SessionActions.shutdown(ss, nb)
