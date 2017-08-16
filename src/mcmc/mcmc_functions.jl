"""
Star log likelihood maker.  Creates a star log prob function that is
parameterized by a flat vector

  - th : [log_fluxes; unconstrained_pos]

Args:
  imgs: Array of observed data Images with the .pixel field
  pos0: Initial location of the source in ra/dec
  pos_delta: (optional) determines how much ra/dec we allow the sampler
    to drift away from pos0
"""
function make_star_loglike(imgs::Array{Image},
                           pos0::Array{Float64, 1};
                           pos_delta::Array{Float64, 1}=[1., 1.])

    # create a function that constrains the pixel location to be within
    # pos0 +- [pos_delta]
    constrain_pos, unconstrain_pos = 
        make_position_transformations(pos0, pos_delta)

    function star_loglike(th::Array{Float64, 1})

      # unpack location and log fluxes (smushed)
      lnfluxes, unc_pos = th[1:5], th[6:end]
      pos = constrain_pos(unc_pos)

      ll = 0.
      for img in imgs

          # sky pixel intensity (sky image)
          epsilon    = img.sky[1, 1]
          iota       = img.nelec_per_nmgy[1]
          sky_pixels = [epsilon * iota for h=1:img.H, w=1:img.W]

          # create and cache unit flux src image
          #src_pixels = [0. for h=1:img.H, w=1:img.W]
          src_pixels = zeros(img.H, img.W)
          write_star_unit_flux(img, pos, src_pixels)

          # compute model image
          rates = sky_pixels .+ (exp(lnfluxes[img.b])*src_pixels)

          # sum per-pixel likelihood contribution
          H, W = size(rates)
          for h in 1:H, w in 1:W
              ll += poisson_lnpdf(img.pixels[h,w], rates[h,w])
          end

      end
    return ll
    end

    return star_loglike, constrain_pos, unconstrain_pos
end


"""
Galaxy log likelihood maker.  Creates a galaxy log prob function that is
parameterized by a flat vector

  - th : [log_fluxes, unconstrained_pos, unconstrained_gal_shape]

Args:
  imgs: Array of observed data Images with the .pixel field
  pos0: Initial location of the source in ra/dec
  pos_delta: (optional) determines how much ra/dec we allow the sampler
    to drift away from pos0
"""
function make_gal_loglike(imgs::Array{Image},
                          pos0::Array{Float64, 1};
                          pos_delta::Array{Float64, 1}=[1., 1.])

    # create a function that constrains the pixel location to be within
    # pos0 +- [pos_delta]
    constrain_pos, unconstrain_pos =
        make_position_transformations(pos0, pos_delta)

    function gal_loglike(th::Array{Float64, 1})

        # unpack location and log fluxes (smushed)
        lnfluxes, unc_pos, ushape = th[1:5], th[6:7], th[8:end]
        pos   = constrain_pos(unc_pos)
        shape = Model.constrain_gal_shape(ushape)

        gal_frac_dev, gal_ab, gal_angle, gal_scale = 
            Model.constrain_gal_shape(ushape)

        ll = 0.
        for img in imgs
            # sky pixel intensity (sky image)
            epsilon    = img.sky[1, 1]
            iota       = img.nelec_per_nmgy[1]
            sky_pixels = [epsilon * iota for h=1:img.H, w=1:img.W]

            # create and cache unit flux src image
            src_pixels = [0. for h=1:img.H, w=1:img.W]
            write_galaxy_unit_flux(img, pos, gal_frac_dev,
                               gal_ab, gal_angle, gal_scale, src_pixels)

            # compute model image
            rates = sky_pixels .+ exp(lnfluxes[img.b])*src_pixels

            # sum per-pixel likelihood contribution
            H, W = size(rates)
            for h in 1:H, w in 1:W
                ll += poisson_lnpdf(img.pixels[h,w], rates[h,w])
            end
        end

    return ll
    end

    return gal_loglike, constrain_pos, unconstrain_pos
end




function make_position_transformations(pos0::Array{Float64, 1},
                                       pos_delta::Array{Float64, 1})

    # create a function that constrains the pixel location to be within
    # pos0 +- [pos_delta]
    pos_lo = pos0 .- pos_delta
    pos_hi = pos0 .+ pos_delta
    pos_range = pos_hi .- pos_lo

    function constrain_pos(upos::Array{Float64, 1})
        unit = 1 ./ (1. + exp.(-upos))
        cpos  = (unit.*pos_range) .+ pos_lo
        return cpos
    end

    function unconstrain_pos(cpos::Array{Float64, 1})
        unit = (cpos .- pos_lo) ./ pos_range
        upos = log.(unit) .- log.(1 .- unit)
        return upos
    end

    return constrain_pos, unconstrain_pos
end



#####################################
# simple log prob implementation    #
#####################################

"""
given a single image, create a function that returns log likelihood of image
as a function of the log flux = [lnu, lnr, lng, lni, lnz]
"""
function make_single_image_logflux_loglike(img0::Image,
                                           pos::Array{Float64, 1};
                                           is_star               = true,
                                           gal_frac_dev::Float64 = .5,
                                           gal_ab::Float64       = .7,
                                           gal_angle::Float64    = .79,
                                           gal_scale::Float64    = 4.)
    # sky pixel intensity
    epsilon    = img0.sky[1, 1]
    iota       = img0.nelec_per_nmgy[1]
    sky_pixels = [epsilon * iota for h=1:img0.H, w=1:img0.W]

    # create and cache unit flux src image
    src_pixels = [0. for h=1:img0.H, w=1:img0.W]
    if is_star
        write_star_unit_flux(img0, pos, src_pixels)
    else
        write_galaxy_unit_flux(img0, pos, gal_frac_dev,
                               gal_ab, gal_angle, gal_scale, src_pixels)
    end

    # create log like handle and return
    function lnflux_loglike(lnflux)

        # compute model image
        rates = sky_pixels .+ exp(lnflux)*src_pixels

        # sum per-pixel likelihood contribution
        ll = 0.
        H, W = size(rates)
        for h in 1:H, w in 1:W
            ll += poisson_lnpdf(img0.pixels[h,w], rates[h,w])
        end

        return ll
    end

    return lnflux_loglike
end


"""
create vectorized star log flux log like function with multiple images
"""
function make_logflux_loglike(imgs::Array{Image},
                              pos::Array{Float64, 1};
                              is_star = true,
                              gal_frac_dev::Float64 = .5,
                              gal_ab::Float64       = .7,
                              gal_angle::Float64    = .79,
                              gal_scale::Float64    = 4.)

    # create per image scalar log flux function (caches unit flux pixel image)
    if is_star
        scalar_funs = [make_single_image_logflux_loglike(img, pos;
                          is_star=true)
                       for img in imgs]
    else
        scalar_funs = [make_single_image_logflux_loglike(img, pos;
                          is_star=false,
                          gal_frac_dev=gal_frac_dev,
                          gal_ab=gal_ab,
                          gal_angle=gal_angle,
                          gal_scale=gal_scale)
                       for img in imgs]
    end

    function lnflux_loglike(lnflux::Vector{Float64})
        # for each of those image/function pairs, compute flux-specific ll
        ll = 0.
        for i in 1:length(imgs)
            b   = imgs[i].b
            ll += scalar_funs[i](lnflux[b])
        end
        return ll
    end

    return lnflux_loglike
end


#############################
# Priors and prior samplers #
#############################

const pp = Model.load_prior()

"""
prior over log fluxes (for stars and galaxies)
"""
function logflux_logprior(lnfluxes::Vector{Float64}; is_star::Bool=true)
    # toggle star / galaxy
    type_i = is_star ? 1 : 2

    # convert to color space for prior calc
    lnr, colors = logfluxes_to_colors(lnfluxes)

    # compute brightness distribution
    llr = logpdf(Normal(pp.flux_mean[type_i], 2*pp.flux_var[type_i]), lnr)

    # compute color likelihoods (mixture model --- computes all component lls)
    nc, nk, ns = size(pp.color_mean)
    llk = [logpdf(MvNormal(pp.color_mean[:,k,type_i],
                           pp.color_cov[:,:,k,type_i]), colors)
           for k in 1:nk]
    lnpik = log.(pp.k[:, type_i])
    llc   = Model.logsumexp(llk .+ lnpik)

    # add and return
    return llr + llc
end

"""
More numerically stable converter from log fluxes to colors
"""
function logfluxes_to_colors(lnfluxes::Vector{Float64})
    lnr = lnfluxes[3]
    colors = Vector{Float64}(4)
    colors[1] = lnfluxes[2] - lnfluxes[1]
    colors[2] = lnfluxes[3] - lnfluxes[2]
    colors[3] = lnfluxes[4] - lnfluxes[3]
    colors[4] = lnfluxes[5] - lnfluxes[4]
    return lnr, colors
end


"""
sample fluxes from prior
"""
function sample_logfluxes(; is_star=true, lnr=nothing)
    # toggle star / galaxy
    type_i = is_star ? 1 : 2

    # sample log r
    if lnr == nothing
        lnr = rand(Normal(pp.flux_mean[type_i], 2*pp.flux_var[type_i]))
    end

    # sample colors
    k = rand(Distributions.Categorical(pp.k[:, type_i]))
    c = rand(MvNormal(pp.color_mean[:,k,type_i], pp.color_cov[:,:,k,type_i]))

    # convert to fluxes
    lnfluxes = log.(Model.colors_to_fluxes(lnr, c))
    return lnfluxes
end

"""
sample colors from prior
"""
function sample_colors(; is_star=true)
    # toggle star / galaxy
    type_i = is_star ? 1 : 2
    # sample colors
    k = rand(Distributions.Categorical(pp.k[:, type_i]))
    c = rand(MvNormal(pp.color_mean[:,k,type_i], pp.color_cov[:,:,k,type_i]))
    return c
end

function sample_logr(; is_star=true)
    type_i = is_star ? 1 : 2
    lnr = rand(Normal(pp.flux_mean[type_i], 2*pp.flux_var[type_i]))
    return lnr
end


function sample_fluxes(i::Int, r_s)
    k_s = rand(Distributions.Categorical(pp.k[i]))
    c_s = rand(Distributions.MvNormal(pp.color[i][:, k_s], pp.color[i][:, :, k_s]))

    l_s = Vector{Float64}(5)
    l_s[3] = r_s
    l_s[4] = l_s[3] * exp(c_s[3])
    l_s[5] = l_s[4] * exp(c_s[4])
    l_s[2] = l_s[3] / exp(c_s[2])
    l_s[1] = l_s[2] / exp(c_s[1])
    l_s
end


