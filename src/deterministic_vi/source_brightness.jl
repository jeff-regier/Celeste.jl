"""
SensitiveFloat objects for expectations involving r_s and c_s.

Args:
vs: A vector of variational parameters

Attributes:
Each matrix has one row for each color and a column for
star / galaxy.  Row 3 is the gamma distribute baseline brightness,
and all other rows are lognormal offsets.
- E_l_a: A NUM_BANDS x NUM_SOURCE_TYPES matrix of expectations and derivatives of
  color terms.  The rows are bands, and the columns
  are star / galaxy.
- E_ll_a: A NUM_BANDS x NUM_SOURCE_TYPES matrix of expectations and derivatives of
  squared color terms.  The rows are bands, and the columns
  are star / galaxy.
"""
struct SourceBrightness{T<:Number}
    # [E[l|a=0], E[l]|a=1]]
    E_l_a::Matrix{SensitiveFloat{T}}

    # [E[l^2|a=0], E[l^2]|a=1]]
    E_ll_a::Matrix{SensitiveFloat{T}}
end


function SourceBrightness(vs::Vector{T};
                          calculate_gradient=true,
                          calculate_hessian=true) where {T<:Number}
    flux_loc = vs[ids.flux_loc]
    flux_scale = vs[ids.flux_scale]
    color_mean = vs[ids.color_mean]
    color_var = vs[ids.color_var]

    # E_l_a has a row for each of the five colors and columns
    # for star / galaxy.
    E_l_a  = Matrix{SensitiveFloat{T}}(NUM_BANDS, NUM_SOURCE_TYPES)
    E_ll_a = Matrix{SensitiveFloat{T}}(NUM_BANDS, NUM_SOURCE_TYPES)

    for i = 1:NUM_SOURCE_TYPES
        for b = 1:NUM_BANDS
            E_l_a[b, i] = SensitiveFloat{T}(length(BrightnessParams), 1,
                                       calculate_gradient, calculate_hessian)
        end

        E_l_a[3, i].v[] = exp(flux_loc[i] + 0.5 * flux_scale[i])
        E_l_a[4, i].v[] = exp(color_mean[3, i] + .5 * color_var[3, i])
        E_l_a[5, i].v[] = exp(color_mean[4, i] + .5 * color_var[4, i])
        E_l_a[2, i].v[] = exp(-color_mean[2, i] + .5 * color_var[2, i])
        E_l_a[1, i].v[] = exp(-color_mean[1, i] + .5 * color_var[1, i])

        if calculate_gradient
            # band 3 is the reference band, relative to which the colors are
            # specified.
            # It is denoted r_s and has a lognormal expectation.
            E_l_a[3, i].d[bids.flux_loc] = E_l_a[3, i].v[]
            E_l_a[3, i].d[bids.flux_scale] = E_l_a[3, i].v[] * .5

            if calculate_hessian
                set_hess!(E_l_a[3, i], bids.flux_loc, bids.flux_loc, E_l_a[3, i].v[])
                set_hess!(E_l_a[3, i], bids.flux_loc, bids.flux_scale, E_l_a[3, i].v[] * 0.5)
                set_hess!(E_l_a[3, i], bids.flux_scale, bids.flux_scale, E_l_a[3, i].v[] * 0.25)
            end

            # The remaining indices involve c_s and have lognormal
            # expectations times E_c_3.

            # band 4 = band 3 * color 3.
            E_l_a[4, i].d[bids.color_mean[3]] = E_l_a[4, i].v[]
            E_l_a[4, i].d[bids.color_var[3]] = E_l_a[4, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[4, i], bids.color_mean[3], bids.color_mean[3], E_l_a[4, i].v[])
                set_hess!(E_l_a[4, i], bids.color_mean[3], bids.color_var[3], E_l_a[4, i].v[] * 0.5)
                set_hess!(E_l_a[4, i], bids.color_var[3], bids.color_var[3], E_l_a[4, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[4, i], E_l_a[3, i])

            # Band 5 = band 4 * color 4.
            E_l_a[5, i].d[bids.color_mean[4]] = E_l_a[5, i].v[]
            E_l_a[5, i].d[bids.color_var[4]] = E_l_a[5, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[5, i], bids.color_mean[4], bids.color_mean[4], E_l_a[5, i].v[])
                set_hess!(E_l_a[5, i], bids.color_mean[4], bids.color_var[4], E_l_a[5, i].v[] * 0.5)
                set_hess!(E_l_a[5, i], bids.color_var[4], bids.color_var[4], E_l_a[5, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[5, i], E_l_a[4, i])

            # Band 2 = band 3 * color 2.
            E_l_a[2, i].d[bids.color_mean[2]] = E_l_a[2, i].v[] * -1.
            E_l_a[2, i].d[bids.color_var[2]] = E_l_a[2, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[2, i], bids.color_mean[2], bids.color_mean[2], E_l_a[2, i].v[])
                set_hess!(E_l_a[2, i], bids.color_mean[2], bids.color_var[2], E_l_a[2, i].v[] * -0.5)
                set_hess!(E_l_a[2, i], bids.color_var[2], bids.color_var[2], E_l_a[2, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[2, i], E_l_a[3, i])

            # Band 1 = band 2 * color 1.
            E_l_a[1, i].d[bids.color_mean[1]] = E_l_a[1, i].v[] * -1.
            E_l_a[1, i].d[bids.color_var[1]] = E_l_a[1, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[1, i], bids.color_mean[1], bids.color_mean[1], E_l_a[1, i].v[])
                set_hess!(E_l_a[1, i], bids.color_mean[1], bids.color_var[1], E_l_a[1, i].v[] * -0.5)
                set_hess!(E_l_a[1, i], bids.color_var[1], bids.color_var[1], E_l_a[1, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[1, i], E_l_a[2, i])
        else
            # Simply update the values if not calculating derivatives.
            E_l_a[4, i].v[] *= E_l_a[3, i].v[]
            E_l_a[5, i].v[] *= E_l_a[4, i].v[]
            E_l_a[2, i].v[] *= E_l_a[3, i].v[]
            E_l_a[1, i].v[] *= E_l_a[2, i].v[]
        end # Derivs

        ################################
        # Squared terms.

        for b = 1:NUM_BANDS
            E_ll_a[b, i] = SensitiveFloat{T}(length(BrightnessParams), 1,
                                           calculate_gradient, calculate_hessian)
        end

        E_ll_a[3, i].v[] = exp(2 * flux_loc[i] + 2 * flux_scale[i])
        E_ll_a[4, i].v[] = exp(2 * color_mean[3, i] + 2 * color_var[3, i])
        E_ll_a[5, i].v[] = exp(2 * color_mean[4, i] + 2 * color_var[4, i])
        E_ll_a[2, i].v[] = exp(-2 * color_mean[2, i] + 2 * color_var[2, i])
        E_ll_a[1, i].v[] = exp(-2 * color_mean[1, i] + 2 * color_var[1, i])

        if calculate_gradient
            # Band 3, the reference band.
            E_ll_a[3, i].d[bids.flux_loc] = 2 * E_ll_a[3, i].v[]
            E_ll_a[3, i].d[bids.flux_scale] = 2 * E_ll_a[3, i].v[]
            if calculate_hessian
                for hess_ids in [(bids.flux_loc, bids.flux_loc),
                                 (bids.flux_loc, bids.flux_scale),
                                 (bids.flux_scale, bids.flux_scale)]
                    set_hess!(E_ll_a[3, i], hess_ids..., 4.0 * E_ll_a[3, i].v[])
                end
            end

            # Band 4 = band 3 * color 3.
            E_ll_a[4, i].d[bids.color_mean[3]] = E_ll_a[4, i].v[] * 2.
            E_ll_a[4, i].d[bids.color_var[3]] = E_ll_a[4, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.color_mean[3], bids.color_mean[3]),
                                 (bids.color_mean[3], bids.color_var[3]),
                                 (bids.color_var[3], bids.color_var[3])]
                    set_hess!(E_ll_a[4, i], hess_ids..., E_ll_a[4, i].v[] * 4.0)
                end
            end
            multiply_sfs!(E_ll_a[4, i], E_ll_a[3, i])

            # Band 5 = band 4 * color 4.
            tmp4 = exp(2 * color_mean[4, i] + 2 * color_var[4, i])
            E_ll_a[5, i].d[bids.color_mean[4]] = E_ll_a[5, i].v[] * 2.
            E_ll_a[5, i].d[bids.color_var[4]] = E_ll_a[5, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.color_mean[4], bids.color_mean[4]),
                                 (bids.color_mean[4], bids.color_var[4]),
                                 (bids.color_var[4], bids.color_var[4])]
                    set_hess!(E_ll_a[5, i], hess_ids..., E_ll_a[5, i].v[] * 4.0)
                end
            end
            multiply_sfs!(E_ll_a[5, i], E_ll_a[4, i])

            # Band 2 = band 3 * color 2
            tmp2 = exp(-2 * color_mean[2, i] + 2 * color_var[2, i])
            E_ll_a[2, i].d[bids.color_mean[2]] = E_ll_a[2, i].v[] * -2.
            E_ll_a[2, i].d[bids.color_var[2]] = E_ll_a[2, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.color_mean[2], bids.color_mean[2]),
                                 (bids.color_var[2], bids.color_var[2])]
                    set_hess!(E_ll_a[2, i], hess_ids..., E_ll_a[2, i].v[] * 4.0)
                end
                set_hess!(E_ll_a[2, i], bids.color_mean[2], bids.color_var[2],
                          E_ll_a[2, i].v[] * -4.0)
            end
            multiply_sfs!(E_ll_a[2, i], E_ll_a[3, i])

            # Band 1 = band 2 * color 1
            E_ll_a[1, i].d[bids.color_mean[1]] = E_ll_a[1, i].v[] * -2.
            E_ll_a[1, i].d[bids.color_var[1]] = E_ll_a[1, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.color_mean[1], bids.color_mean[1]),
                                 (bids.color_var[1], bids.color_var[1])]
                    set_hess!(E_ll_a[1, i], hess_ids..., E_ll_a[1, i].v[] * 4.0)
                end
                set_hess!(E_ll_a[1, i], bids.color_mean[1], bids.color_var[1],
                          E_ll_a[1, i].v[] * -4.0)
            end
            multiply_sfs!(E_ll_a[1, i], E_ll_a[2, i])
        else
            # Simply update the values if not calculating derivatives.
            E_ll_a[4, i].v[] *= E_ll_a[3, i].v[]
            E_ll_a[5, i].v[] *= E_ll_a[4, i].v[]
            E_ll_a[2, i].v[] *= E_ll_a[3, i].v[]
            E_ll_a[1, i].v[] *= E_ll_a[2, i].v[]
        end # calculate_gradient
    end

    SourceBrightness{T}(E_l_a, E_ll_a)
end


"""
Load the source brightnesses for these model params.  Each SourceBrightness
object has information for all bands and object types.

Returns:
  - An array of SourceBrightness objects for each object in 1:ea.S.  Only
    sources in ea.active_sources will have derivative information.
"""
function load_source_brightnesses(
        ea::ElboArgs,
        vp::VariationalParams{T};
        calculate_gradient::Bool=true,
        calculate_hessian::Bool=true) where {T<:Number}
    sbs = Vector{SourceBrightness{T}}(ea.S)

    for s in 1:ea.S
        this_deriv = (s in ea.active_sources) && calculate_gradient
        this_hess = (s in ea.active_sources) && calculate_hessian
        sbs[s] = SourceBrightness(vp[s];
                                  calculate_gradient=this_deriv,
                                  calculate_hessian=this_hess)
    end

    sbs
end
