#!/usr/bin/env julia
# check_notebook.jl — run a Pluto notebook headless and fail if any cell errors.
#
# Usage:
#   julia --project=. experiments/scripts/check_notebook.jl experiments/notebooks/01_smoke_test.jl
#
# Exits 0 if all cells run without error, 1 otherwise. Useful as a CI / pre-export gate.

using Pluto

function check_notebook(nb_path::AbstractString)
    println("Opening notebook: $nb_path")
    ss = Pluto.ServerSession()
    ss.options.evaluation.workspace_use_distributed = false
    nb = Pluto.SessionActions.open(ss, nb_path; run_async=false)

    nerr = 0
    for (i, cell) in enumerate(nb.cells)
        first_line = strip(first(split(cell.code, "\n")))
        preview = first_line[1:min(70, lastindex(first_line))]
        if cell.errored
            nerr += 1
            body = cell.output.body
            println("  [ERROR cell $i] $preview")
            println("    output: ", body)
        else
            println("  [ok    cell $i] $preview")
        end
    end

    Pluto.SessionActions.shutdown(ss, nb)
    println("Result: $nerr error(s) / $(length(nb.cells)) cell(s)")
    return nerr
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        println(stderr, "usage: check_notebook.jl <path-to-pluto-notebook.jl>")
        exit(2)
    end
    nerrs = check_notebook(ARGS[1])
    exit(nerrs == 0 ? 0 : 1)
end
