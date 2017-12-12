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

Log.LEVEL[] = Log.WARN  # do not show info during tests.
anyerrors = false

wd = pwd()
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
cd(datadir)
run(`make`)
run(`make RUN=4263 CAMCOL=5 FIELD=119`)
# Ensure GalSim test images are available.
const galsim_benchmark_dir = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
cd(galsim_benchmark_dir)
run(`make fetch`)
cd(wd)

# Check whether to run time-consuming tests.
long_running_flag = "--long-running"
test_long_running = long_running_flag in ARGS
test_files = setdiff(ARGS, [ long_running_flag ])

if length(test_files) > 0
    testfiles = ["test_$(arg).jl" for arg in test_files]
else
    testdir = joinpath(Pkg.dir("Celeste"), "test")
    testfiles = filter(r"^test_.*\.jl$", readdir(testdir))
    if !test_long_running
        info("Skipping stripe82 tests without --long-running flag.")
        testfiles = setdiff(testfiles, ["test_stripe82.jl"])
    end
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
    Base.time_print(1e9 * t, memallocs.allocd, memallocs.total_time,
                    Base.gc_alloc_count(memallocs))
end
@printf "Total time: %7.2f seconds\n" totaltime
