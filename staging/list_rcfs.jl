using Celeste

function main()
    f = FITSIO.FITS(ENV["FIELD_EXTENTS"])

    hdu = f[2]::FITSIO.TableHDU

    # read in the entire table.
    all_run = read(hdu, "run")::Vector{Int16}
    all_camcol = read(hdu, "camcol")::Vector{UInt8}
    all_field = read(hdu, "field")::Vector{Int16}

    close(f)

    for i in eachindex(all_run)
        println("RUN=$(all_run[i]) CAMCOL=$(all_camcol[i]) FIELD=$(all_field[i])")
    end
end

main()
