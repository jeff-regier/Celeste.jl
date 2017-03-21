#!/usr/bin/env julia

# Lists all of the quarters of square degrees around the area
# imaged by SDSS. Output in this format can be piped to either infer-box.jl
# or estimate-box-runtime.jl, e.g.
# 
#   ./list_quarters.jl | sort -R | xargs -n 4 estimate-box-runtime.jl

for ra in 0.01:.5:380, dec in -30.01:.5:90
    println("$(ra - 0.5) $ra $(dec - 0.5) $dec")
end


