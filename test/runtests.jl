#!/usr/bin/env julia

using Base.Test
using Distributions
using ForwardDiff
using StaticArrays

using Celeste: Model, DeterministicVI

import Celeste: DeterministicVI, ParallelRun, Log
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import Celeste.SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))

using SampleData

# Set logging level and timing reporting (TODO: ability to set on command line)
Log.LEVEL[] = Log.WARN
verbose_timing = false

# Check whether to run time-consuming tests.
long_running_flag = "--long-running"
test_long_running = long_running_flag in ARGS


test_files = setdiff(ARGS, [ long_running_flag ])
if length(test_files) > 0
    testfiles = ["test_$(arg).jl" for arg in test_files]
else
    testdir = joinpath(Pkg.dir("Celeste"), "test")
    testfiles = filter(r"^test_.*\.jl$", readdir(testdir))
end

timing_info = Any[]
for testfile in testfiles
        _, t, bytes, gctime, memallocs = @timed include(testfile)
        push!(timing_info, (t, bytes, gctime, memallocs))
end

println("\nTiming info:")
totaltime = 0.0
for i in eachindex(timing_info)
    t, bytes, gctime, memallocs = timing_info[i]
    totaltime += t
    @printf "%30s: " testfiles[i]
    if verbose_timing
        Base.time_print(1e9 * t, memallocs.allocd, memallocs.total_time,
                        Base.gc_alloc_count(memallocs))
    else
        @printf "%7.2f seconds\n" t
    end
end
@printf "%30s: %7.2f seconds\n" "Total time" totaltime
