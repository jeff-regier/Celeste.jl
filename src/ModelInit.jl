# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module ModelInit

export sample_prior, cat_init, peak_init

using FITSIO
using Distributions
using WCSLIB
using Util
using CelesteTypes


function sample_prior()
    const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

    stars_file = open("$dat_dir/priors/stars.dat")
    r_fit1, k1, cmean1, ccov1 = deserialize(stars_file)
    close(stars_file)

    gals_file = open("$dat_dir/priors/gals.dat")
    r_fit2, k2, cmean2, ccov2 = deserialize(gals_file)
    close(gals_file)

    # TODO: use r_fit1 and r_fit2 instead of magic numbers ?

    # magic numbers below determined from the output of primary
    # on the test set of stamps
    PriorParams(
        [0.28, 0.72],                       # a
        [(0.47, 0.012), (1.28, 0.11)],      # r
        Vector{Float64}[k1, k2],            # k
        [(cmean1, ccov1), (cmean2, ccov2)]) # c
end


#TODO: use blob (and perhaps priors) to initialize these sensibly
function init_source(init_pos::Vector{Float64})
    ret = Array(Float64, length(CanonicalParams))
    ret[ids.a[2]] = 0.5
    ret[ids.a[1]] = 1.0 - ret[ids.a[2]]
    ret[ids.u[1]] = init_pos[1]
    ret[ids.u[2]] = init_pos[2]
    ret[ids.r1] = 1e3
    ret[ids.r2] = 2e-3
    ret[ids.e_dev] = 0.5
    ret[ids.e_axis] = 0.5
    ret[ids.e_angle] = 0.
    ret[ids.e_scale] = 1.
    ret[ids.k] = 1. / size(ids.k, 1)
    ret[ids.c1] = 0.
    ret[ids.c2] =  1e-2
    ret
end


function init_source(ce::CatalogEntry)
    ret = init_source(ce.pos)

    ret[ids.r1[1]] = max(0.0001, ce.star_fluxes[3]) ./ ret[ids.r2[1]]
    ret[ids.r1[2]] = max(0.0001, ce.gal_fluxes[3]) ./ ret[ids.r2[2]]

    get_color(c2, c1) = begin
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end
    get_colors(raw_fluxes) = begin
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.c1[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.c1[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.e_dev] = min(max(ce.gal_frac_dev, 0.01), 0.99)

    ret[ids.e_axis] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.0001), 0.9999)
    ret[ids.e_angle] = ce.gal_angle
    ret[ids.e_scale] = ce.is_star ? 0.2 : max(ce.gal_scale, 0.2)

    ret
end


function matched_filter(img::Image)
    H, W = 5, 5
    kernel = zeros(Float64, H, W)
    for k in 1:3
        mvn = MvNormal(img.psf[k].xiBar, img.psf[k].tauBar)
        for h in 1:H
            for w in 1:W
                x = [h - (H + 1) / 2., w - (W + 1) / 2.]
                kernel[h, w] += img.psf[k].alphaBar * pdf(mvn, x)
            end
        end
    end
    kernel /= sum(kernel)
end


function convolve_image(img::Image)
    # Not totally sure why this is helpful,
    # but it may help find
    # peaks in an image that has already been gaussian-blurred.
    # (Ref in ICML & NIPS papers).
    kernel = matched_filter(img)
    H, W = size(img.pixels)
    padded_pixels = Array(Float64, H + 8, W + 8)
    fill!(padded_pixels, median(img.pixels))
    padded_pixels[5:H+4,5:W+4] = img.pixels
    conv2(padded_pixels, kernel)[7:H+6, 7:W+6]
end


function peak_starts(blob::Blob)
    # Heuristically find the peaks in the blob.  (Blob == field)
    H, W = size(blob[1].pixels)
    added_pixels = zeros(Float64, H, W)
    for b in 1:5
        added_pixels += convolve_image(blob[b])
    end
    spread = quantile(added_pixels[:], .7) - quantile(added_pixels[:], .2)
    threshold = median(added_pixels) + 3spread

    peaks = Array(Vector{Float64}, 0)
    i = 0
    for h=3:(H-3), w=3:(W-3)
        if added_pixels[h, w] > threshold &&
                added_pixels[h, w] > maximum(added_pixels[h-2:h+2, w-2:w+2]) - .1
            i += 1
#            println("found peak $i: ", h, " ", w)
#            println(added_pixels[h-3:min(h+3,99), w-3:min(w+3,99)])
            push!(peaks, [h, w])
        end
    end

    R = length(peaks)
    peaks_mat = Array(Float64, 2, R)
    for i in 1:R
        peaks_mat[:, i] = peaks[i]
    end

    peaks_mat
#    wcsp2s(img.wcs, peaks_mat)
end


function peak_init(blob::Blob; patch_radius::Float64=Inf,
        tile_width::Int64=typemax(Int64))
    v1 = peak_starts(blob)
    S = size(v1)[2]
    vp = [init_source(v1[:, s]) for s in 1:S]
    twice_radius = float(max(blob[1].H, blob[1].W))
    # TODO: use non-trival patch radii, based on blob detection routine
    patches = [SkyPatch(v1[:, s], patch_radius) for s in 1:S]
    ModelParams(vp, sample_prior(), patches, tile_width)
end

#=
function min_patch_radius(ce::CatalogEntry, blob::Blob)
    max_var = maximum([maximum([maximum(pc.tauBar) for pc in img.psf])
                    for img in blob])
    if !ce.is_star
        XiXi = Util.get_bvn_cov(ce.gal_ab, ce.gal_angle, ce.gal_scale)
        XiXi_max = maximum(XiXi)
        max_var += maximum([maximum([gc.nuBar * XiXi_max
            for gc in galaxy_prototypes[i]]) for i in 1:2])
    end

    sky = [img.epsilon for img in blob]
    fluxes = ce.is_star ? : ce.star_fluxes : ce.gal_fluxes
    thresholds = fluxes ./ (20 * sky)  # 5% of sky
    pdf_target = threshold / intensity
    # ignore xiBar for now
    max_var = maximum(diag(pc.sigmaBar))
    rhs = log(pdf_target) + log(2pi) + 0.5logdet(pc.sigmaBar)
    x = sqrt(-2 * max_var * rhs)
    println("patch radius requirement: $x")
end
=#

function cat_init(cat::Vector{CatalogEntry}; patch_radius::Float64=Inf,
        tile_width::Int64=typemax(Int64))
    vp = [init_source(ce) for ce in cat]
    # TODO: use non-trivial patch radii, based on the catalog
    patches = [SkyPatch(ce.pos, patch_radius) for ce in cat]
    ModelParams(vp, sample_prior(), patches, tile_width)
end


end
