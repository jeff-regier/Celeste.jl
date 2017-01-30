"""
Tools for exploratory data analysis and debugging.
"""
module CelesteEDA

using DataFrames
using StaticArrays

import ..DeterministicVI:
    load_source_brightnesses, accumulate_source_pixel_brightness!, ElboArgs,
    populate_fsm!
import ..Model:
    linear_world_to_pix, load_bvn_mixtures, ids_names, ids, CatalogEntry
import ..SensitiveFloats.clear!
import ..DeterministicVIImagePSF:
    FSMSensitiveFloatMatrices
using ..DeterministicVIImagePSF:
    clear_brightness!, populate_star_fsm_image!, populate_gal_fsm_image!,
    accumulate_source_image_brightness!, load_gal_bvn_mixtures

import Base.print
function print(ce::CatalogEntry)
    for field in fieldnames(ce)
      println(field, ": ", getfield(ce, field))
    end
end


function print_vp(vp::Vector{Float64})
    df = DataFrame(ids=ids_names)
    s = 1
    df[Symbol(string("v", s))] = vp
    println(df)
end


function source_pixel_location(ea::ElboArgs, s::Int, n::Int)
    p = ea.patches[s, n]
    pix_loc = linear_world_to_pix(
        p.wcs_jacobian,
        p.center,
        p.pixel_center,
        ea.vp[s][ids.u])
    return pix_loc - p.bitmap_offset
end


function render_source(ea::ElboArgs, s::Int, n::Int;
                       include_epsilon=true, field=:E_G,
                       include_iota=true)
    local p = ea.patches[s, n]
    local image = fill(NaN, size(p.active_pixel_bitmap))
    local sbs = load_source_brightnesses(
        ea, calculate_gradient=false, calculate_hessian=false)
    star_mcs, gal_mcs = load_bvn_mixtures(ea.S, ea.patches,
                                ea.vp, ea.active_sources,
                                ea.psf_K, n,
                                calculate_gradient=false,
                                calculate_hessian=false)

    p = ea.patches[s,n]

    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_offset[1] + h2
        w = p.bitmap_offset[2] + w2

        if !p.active_pixel_bitmap[h2, w2]
            continue
        end

        clear!(ea.elbo_vars.E_G)
        clear!(ea.elbo_vars.var_G)

        hw = SVector{2,Float64}(h, w)
        is_active_source = s in ea.active_sources
        populate_fsm!(ea.elbo_vars.bvn_derivs,
                      ea.elbo_vars.fs0m, ea.elbo_vars.fs1m,
                      s, hw, is_active_source,
                      p.wcs_jacobian,
                      gal_mcs, star_mcs)

        accumulate_source_pixel_brightness!(
            ea.elbo_vars, ea, ea.elbo_vars.E_G, ea.elbo_vars.var_G,
            ea.elbo_vars.fs0m, ea.elbo_vars.fs1m,
            sbs[s], ea.images[n].b, s, false)

        if field == :E_G
            image[h2, w2] = ea.elbo_vars.E_G.v[]
        elseif field == :fs0m
            image[h2, w2] = ea.elbo_vars.fs0m.v[]
        elseif field == :fs1m
            image[h2, w2] = ea.elbo_vars.fs1m.v[]
        else
            error("Unknown field ", field)
        end
        if include_epsilon
            image[h2, w2] += ea.images[n].epsilon_mat[h, w]
        end
        if include_iota
            image[h2, w2] *= ea.images[n].iota_vec[h]
        end
    end

    return image
end


function render_source_fft(
    ea::ElboArgs,
    fsm_mat::Matrix{FSMSensitiveFloatMatrices},
    s::Int, n::Int;
    include_epsilon=true,
    field=:E_G, include_iota=true)

    local p = ea.patches[s, n]
    local image = fill(NaN, size(p.active_pixel_bitmap))
    local sbs = load_source_brightnesses(
        ea, calculate_gradient=false, calculate_hessian=false)
    local fsms = fsm_mat[s, n]

    local gal_mcs = load_gal_bvn_mixtures(
            ea.S, ea.patches, ea.vp, ea.active_sources, n,
            calculate_gradient=false,
            calculate_hessian=false);

    clear_brightness!(fsms)
    populate_star_fsm_image!(
        ea, s, n, fsms.psf, fsms.fs0m_conv,
        fsms.h_lower, fsms.w_lower, fsms.kernel_fun, fsms.kernel_width)
    populate_gal_fsm_image!(ea, s, n, gal_mcs, fsms)
    accumulate_source_image_brightness!(ea, s, n, fsms, sbs[s])

    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_offset[1] + h2
        w = p.bitmap_offset[2] + w2

        if !p.active_pixel_bitmap[h2, w2]
            continue
        end

        h_fsm = h - fsms.h_lower + 1
        w_fsm = w - fsms.w_lower + 1

        image[h2, w2] = getfield(fsms, field)[h_fsm, w_fsm].v[]
        if include_epsilon
            image[h2, w2] += ea.images[n].epsilon_mat[h, w]
        end
        if include_iota
            image[h2, w2] *= ea.images[n].iota_vec[h]
        end
    end

    return deepcopy(image)
end


function show_source_image(ea::ElboArgs, s::Int, n::Int)
    p = ea.patches[s, n]
    H2, W2 = size(p.active_pixel_bitmap)
    image = fill(NaN, H2, W2);
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_offset[1] + h2
        w = p.bitmap_offset[2] + w2

        if !p.active_pixel_bitmap[h2, w2]
            continue
        end

        image[h2, w2] = ea.images[n].pixels[h, w]
    end
    return image
end

end
