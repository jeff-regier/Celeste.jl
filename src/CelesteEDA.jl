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
    linear_world_to_pix, load_bvn_mixtures, ids_names, ids, CatalogEntry, populate_fsm!
using ..SensitiveFloats: clear!
import ..DeterministicVIImagePSF:
    FSMSensitiveFloatMatrices
using ..DeterministicVIImagePSF:
    clear_brightness!, populate_star_fsm_image!, populate_gal_fsm_image!,
    accumulate_source_image_brightness!, load_gal_bvn_mixtures
using ..Infer: get_active_pixel_range, is_pixel_in_patch

using StaticArrays: SVector

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
    df
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


function render_valid_pixels(ea::ElboArgs, sources::Vector{Int}, n::Int)
    H_min, W_min, H_max, W_max = get_active_pixel_range(ea.patches, sources, n)
    image = falses(H_max - H_min + 1, W_max - W_min + 1)

    for h in H_min:H_max, w in W_min:W_max
        # The pixel in the image
        h_img = h - H_min + 1
        w_img = w - W_min + 1

        if any([ is_pixel_in_patch(h, w, p) for p in ea.patches[sources, n] ])
            image[h_img, w_img] = true
        end
    end
    image
end


function render_sources(ea::ElboArgs, sources::Vector{Int}, n::Int;
                        include_epsilon=true, field=:E_G,
                        include_iota=true)
    local sbs = load_source_brightnesses(
        ea, calculate_gradient=true, calculate_hessian=true)
    star_mcs, gal_mcs = load_bvn_mixtures(ea.S, ea.patches,
                                ea.vp, ea.active_sources,
                                ea.psf_K, n,
                                calculate_gradient=true,
                                calculate_hessian=true)

    H_min, W_min, H_max, W_max = get_active_pixel_range(ea.patches, sources, n)
    image = fill(NaN, H_max - H_min + 1, W_max - W_min + 1)

    for h in H_min:H_max, w in W_min:W_max
        # The pixel in the image
        h_img = h - H_min + 1
        w_img = w - W_min + 1

        if any([ is_pixel_in_patch(h, w, p) for p in ea.patches[sources, n] ])
            image[h_img, w_img] = 0.0
        end
    end

    for s in sources, h in H_min:H_max, w in W_min:W_max
        p = ea.patches[s, n]
        if is_pixel_in_patch(h, w, p)
            # The pixel in the image
            h_img = h - H_min + 1
            w_img = w - W_min + 1

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
                image[h_img, w_img] += ea.elbo_vars.E_G.v[]
            elseif field == :fs0m
                image[h_img, w_img] += ea.elbo_vars.fs0m.v[]
            elseif field == :fs1m
                image[h_img, w_img] += ea.elbo_vars.fs1m.v[]
            else
                error("Unknown field ", field)
            end
        end
    end

    for h in H_min:H_max, w in W_min:W_max
        if !any([ is_pixel_in_patch(h, w, p) for p in ea.patches[sources, n] ])
            continue
        end
        
        # The pixel in the image
        h_img = h - H_min + 1
        w_img = w - W_min + 1
        if include_epsilon
            image[h_img, w_img] += ea.images[n].epsilon_mat[h, w]
        end
        if include_iota
            image[h_img, w_img] *= ea.images[n].iota_vec[h]
        end        
    end

    return image
end


# Render a single source from a single image
# in an matrix that is the size of the active_pixel_bitmap.
function render_sources_fft(
    ea::ElboArgs,
    fsm_mat::Matrix{FSMSensitiveFloatMatrices},
    sources::Vector{Int}, n::Int;
    include_epsilon=true,
    field=:E_G, include_iota=true)

    
    sbs = load_source_brightnesses(
        ea, calculate_gradient=true, calculate_hessian=true)
        
    gal_mcs = load_gal_bvn_mixtures(
            ea.S, ea.patches, ea.vp, ea.active_sources, n,
            calculate_gradient=true, calculate_hessian=true);

    for s in sources
        fsms = fsm_mat[s, n]
        clear_brightness!(fsms)
        populate_star_fsm_image!(
            ea, s, n, fsms.psf, fsms.fs0m_conv,
            fsms.h_lower, fsms.w_lower, fsms.kernel_fun, fsms.kernel_width)
        populate_gal_fsm_image!(ea, s, n, gal_mcs, fsms)
        accumulate_source_image_brightness!(ea, s, n, fsms, sbs[s])
    end

    H_min, W_min, H_max, W_max = get_active_pixel_range(ea.patches, sources, n)
    image = fill(NaN, H_max - H_min + 1, W_max - W_min + 1)

    for h in H_min:H_max, w in W_min:W_max
        # The pixel in the image
        h_img = h - H_min + 1
        w_img = w - W_min + 1

        if any([ is_pixel_in_patch(h, w, p) for p in ea.patches[sources, n] ])
            image[h_img, w_img] = 0.0
        end
    end

    for s in sources, h in H_min:H_max, w in W_min:W_max
        fsms = fsm_mat[s, n]
        p = ea.patches[s, n]
        if is_pixel_in_patch(h, w, p)
            # The pixel in the fsm matrix
            h_fsm = h - fsms.h_lower + 1
            w_fsm = w - fsms.w_lower + 1

            # The pixel in the image
            h_img = h - H_min + 1
            w_img = w - W_min + 1
            
            image[h_img, w_img] += getfield(fsms, field)[h_fsm, w_fsm].v[]
        end
    end

    for h in H_min:H_max, w in W_min:W_max
        if !any([ is_pixel_in_patch(h, w, p) for p in ea.patches[sources, n] ])
            continue
        end

        # The pixel in the image
        h_img = h - H_min + 1
        w_img = w - W_min + 1
        if include_epsilon
            image[h_img, w_img] += ea.images[n].epsilon_mat[h, w]
        end
        if include_iota
            image[h_img, w_img] *= ea.images[n].iota_vec[h]
        end        
    end

    return image
end


function show_sources_image(ea::ElboArgs, sources::Vector{Int}, n::Int)
    H_min, W_min, H_max, W_max = get_active_pixel_range(ea.patches, sources, n)
    image = fill(NaN, H_max - H_min + 1, W_max - W_min + 1)
    for h in H_min:H_max, w in W_min:W_max, p in ea.patches[sources, n]
        if is_pixel_in_patch(h, w, p)
            # The pixel in the image
            h_img = h - H_min + 1
            w_img = w - W_min + 1
            
            image[h_img, w_img] = ea.images[n].pixels[h, w]
        end
    end
    return image
end

end
