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
    Phi = 0.5

    const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

    Upsilon = Array(Float64, 2)
    Psi = Array(Float64, 2)
    r_file = open("$dat_dir/r_prior.dat")
    ((Upsilon[1], Psi[1]), (Upsilon[2], Psi[2])) = deserialize(r_file)
    close(r_file)

    Xi = Array(Vector{Float64}, 2)
    Omega = Array(Array{Float64, 2}, 2)
    Lambda = Array(Array{Array{Float64, 2}}, 2)
    ck_file = open("$dat_dir/ck_prior.dat")
    ((Xi[1], Omega[1], Lambda[1]), (Xi[2], Omega[2], Lambda[2])) = deserialize(ck_file)
    close(r_file)

    muReg = -0.1989290211077033  # from mle for a log normal fit to primary
    sigmaReg = 0.6157385349577496

    alphaReg = 1.32  # from mle for a beta dist fit to primary
    betaReg = 1.21

    PriorParams(Phi, Upsilon, Psi, Xi, Omega, Lambda,
        muReg, sigmaReg, alphaReg, betaReg)
end


#TODO: use blob (and perhaps priors) to initialize these sensibly
function init_source(init_pos::Vector{Float64})
    ret = Array(Float64, length(all_params))
    ret[ids.chi] = 0.5
    ret[ids.mu[1]] = init_pos[1]
    ret[ids.mu[2]] = init_pos[2]
    ret[ids.gamma] = 1e3
    ret[ids.zeta] = 2e-3
    ret[ids.theta] = 0.5
    ret[ids.rho] = 0.5
    ret[ids.phi] = 0.
    ret[ids.sigma] = 1.
    ret[ids.kappa] = 1. / size(ids.kappa, 1)
    ret[ids.beta] = 0.
    ret[ids.lambda] =  1e-2
    ret
end


function init_source(ce::CatalogEntry)
    ret = init_source(ce.pos)

    ret[ids.gamma[1]] = max(0.0001, ce.star_fluxes[3]) ./ ret[ids.zeta[1]]
    ret[ids.gamma[2]] = max(0.0001, ce.gal_fluxes[3]) ./ ret[ids.zeta[2]]

    get_color(c2, c1) = begin
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end
    get_colors(raw_fluxes) = begin
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.beta[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.beta[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.theta] = min(max(ce.gal_frac_dev, 0.01), 0.99)

    ret[ids.rho] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.0001), 0.9999)
    ret[ids.phi] = ce.gal_angle
    ret[ids.sigma] = ce.is_star ? 0.2 : max(ce.gal_scale, 0.2)

    ret
end


function matched_filter(img::Image)
    H, W = 5, 5
    kernel = zeros(Float64, H, W)
    for k in 1:3
        mvn = MvNormal(img.psf[k].xiBar, img.psf[k].SigmaBar)
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
    max_var = maximum([maximum([maximum(pc.SigmaBar) for pc in img.psf]) 
                    for img in blob])
    if !ce.is_star
        XiXi = Util.get_bvn_cov(ce.gal_ab, ce.gal_angle, ce.gal_scale)
        XiXi_max = maximum(XiXi)
        max_var += maximum([maximum([gc.sigmaTilde * XiXi_max 
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
