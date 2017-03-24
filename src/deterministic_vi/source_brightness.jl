"""
SensitiveFloat objects for expectations involving r_s and c_s.

Args:
vs: A vector of variational parameters

Attributes:
Each matrix has one row for each color and a column for
star / galaxy.  Row 3 is the gamma distribute baseline brightness,
and all other rows are lognormal offsets.
- E_l_a: A B x Ia matrix of expectations and derivatives of
  color terms.  The rows are bands, and the columns
  are star / galaxy.
- E_ll_a: A B x Ia matrix of expectations and derivatives of
  squared color terms.  The rows are bands, and the columns
  are star / galaxy.
"""
immutable SourceBrightness{NumType <: Number}
    # [E[l|a=0], E[l]|a=1]]
    E_l_a::Matrix{SensitiveFloat{NumType}}

    # [E[l^2|a=0], E[l^2]|a=1]]
    E_ll_a::Matrix{SensitiveFloat{NumType}}
end


function SourceBrightness{NumType <: Number}(vs::Vector{NumType};
                                             calculate_gradient=true,
                                             calculate_hessian=true)
    r1 = vs[ids.r1]
    r2 = vs[ids.r2]
    c1 = vs[ids.c1]
    c2 = vs[ids.c2]

    # E_l_a has a row for each of the five colors and columns
    # for star / galaxy.
    E_l_a  = Matrix{SensitiveFloat{NumType}}(B, Ia)
    E_ll_a = Matrix{SensitiveFloat{NumType}}(B, Ia)

    for i = 1:Ia
        for b = 1:B
            E_l_a[b, i] = SensitiveFloat{NumType}(length(BrightnessParams), 1,
                                       calculate_gradient, calculate_hessian)
        end

        E_l_a[3, i].v[] = exp(r1[i] + 0.5 * r2[i])
        E_l_a[4, i].v[] = exp(c1[3, i] + .5 * c2[3, i])
        E_l_a[5, i].v[] = exp(c1[4, i] + .5 * c2[4, i])
        E_l_a[2, i].v[] = exp(-c1[2, i] + .5 * c2[2, i])
        E_l_a[1, i].v[] = exp(-c1[1, i] + .5 * c2[1, i])

        if calculate_gradient
            # band 3 is the reference band, relative to which the colors are
            # specified.
            # It is denoted r_s and has a lognormal expectation.
            E_l_a[3, i].d[bids.r1] = E_l_a[3, i].v[]
            E_l_a[3, i].d[bids.r2] = E_l_a[3, i].v[] * .5

            if calculate_hessian
                set_hess!(E_l_a[3, i], bids.r1, bids.r1, E_l_a[3, i].v[])
                set_hess!(E_l_a[3, i], bids.r1, bids.r2, E_l_a[3, i].v[] * 0.5)
                set_hess!(E_l_a[3, i], bids.r2, bids.r2, E_l_a[3, i].v[] * 0.25)
            end

            # The remaining indices involve c_s and have lognormal
            # expectations times E_c_3.

            # band 4 = band 3 * color 3.
            E_l_a[4, i].d[bids.c1[3]] = E_l_a[4, i].v[]
            E_l_a[4, i].d[bids.c2[3]] = E_l_a[4, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[4, i], bids.c1[3], bids.c1[3], E_l_a[4, i].v[])
                set_hess!(E_l_a[4, i], bids.c1[3], bids.c2[3], E_l_a[4, i].v[] * 0.5)
                set_hess!(E_l_a[4, i], bids.c2[3], bids.c2[3], E_l_a[4, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[4, i], E_l_a[3, i])

            # Band 5 = band 4 * color 4.
            E_l_a[5, i].d[bids.c1[4]] = E_l_a[5, i].v[]
            E_l_a[5, i].d[bids.c2[4]] = E_l_a[5, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[5, i], bids.c1[4], bids.c1[4], E_l_a[5, i].v[])
                set_hess!(E_l_a[5, i], bids.c1[4], bids.c2[4], E_l_a[5, i].v[] * 0.5)
                set_hess!(E_l_a[5, i], bids.c2[4], bids.c2[4], E_l_a[5, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[5, i], E_l_a[4, i])

            # Band 2 = band 3 * color 2.
            E_l_a[2, i].d[bids.c1[2]] = E_l_a[2, i].v[] * -1.
            E_l_a[2, i].d[bids.c2[2]] = E_l_a[2, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[2, i], bids.c1[2], bids.c1[2], E_l_a[2, i].v[])
                set_hess!(E_l_a[2, i], bids.c1[2], bids.c2[2], E_l_a[2, i].v[] * -0.5)
                set_hess!(E_l_a[2, i], bids.c2[2], bids.c2[2], E_l_a[2, i].v[] * 0.25)
            end
            multiply_sfs!(E_l_a[2, i], E_l_a[3, i])

            # Band 1 = band 2 * color 1.
            E_l_a[1, i].d[bids.c1[1]] = E_l_a[1, i].v[] * -1.
            E_l_a[1, i].d[bids.c2[1]] = E_l_a[1, i].v[] * .5
            if calculate_hessian
                set_hess!(E_l_a[1, i], bids.c1[1], bids.c1[1], E_l_a[1, i].v[])
                set_hess!(E_l_a[1, i], bids.c1[1], bids.c2[1], E_l_a[1, i].v[] * -0.5)
                set_hess!(E_l_a[1, i], bids.c2[1], bids.c2[1], E_l_a[1, i].v[] * 0.25)
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

        for b = 1:B
            E_ll_a[b, i] = SensitiveFloat{NumType}(length(BrightnessParams), 1,
                                           calculate_gradient, calculate_hessian)
        end

        E_ll_a[3, i].v[] = exp(2 * r1[i] + 2 * r2[i])
        E_ll_a[4, i].v[] = exp(2 * c1[3, i] + 2 * c2[3, i])
        E_ll_a[5, i].v[] = exp(2 * c1[4, i] + 2 * c2[4, i])
        E_ll_a[2, i].v[] = exp(-2 * c1[2, i] + 2 * c2[2, i])
        E_ll_a[1, i].v[] = exp(-2 * c1[1, i] + 2 * c2[1, i])

        if calculate_gradient
            # Band 3, the reference band.
            E_ll_a[3, i].d[bids.r1] = 2 * E_ll_a[3, i].v[]
            E_ll_a[3, i].d[bids.r2] = 2 * E_ll_a[3, i].v[]
            if calculate_hessian
                for hess_ids in [(bids.r1, bids.r1),
                                 (bids.r1, bids.r2),
                                 (bids.r2, bids.r2)]
                    set_hess!(E_ll_a[3, i], hess_ids..., 4.0 * E_ll_a[3, i].v[])
                end
            end

            # Band 4 = band 3 * color 3.
            E_ll_a[4, i].d[bids.c1[3]] = E_ll_a[4, i].v[] * 2.
            E_ll_a[4, i].d[bids.c2[3]] = E_ll_a[4, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.c1[3], bids.c1[3]),
                                 (bids.c1[3], bids.c2[3]),
                                 (bids.c2[3], bids.c2[3])]
                    set_hess!(E_ll_a[4, i], hess_ids..., E_ll_a[4, i].v[] * 4.0)
                end
            end
            multiply_sfs!(E_ll_a[4, i], E_ll_a[3, i])

            # Band 5 = band 4 * color 4.
            tmp4 = exp(2 * c1[4, i] + 2 * c2[4, i])
            E_ll_a[5, i].d[bids.c1[4]] = E_ll_a[5, i].v[] * 2.
            E_ll_a[5, i].d[bids.c2[4]] = E_ll_a[5, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.c1[4], bids.c1[4]),
                                 (bids.c1[4], bids.c2[4]),
                                 (bids.c2[4], bids.c2[4])]
                    set_hess!(E_ll_a[5, i], hess_ids..., E_ll_a[5, i].v[] * 4.0)
                end
            end
            multiply_sfs!(E_ll_a[5, i], E_ll_a[4, i])

            # Band 2 = band 3 * color 2
            tmp2 = exp(-2 * c1[2, i] + 2 * c2[2, i])
            E_ll_a[2, i].d[bids.c1[2]] = E_ll_a[2, i].v[] * -2.
            E_ll_a[2, i].d[bids.c2[2]] = E_ll_a[2, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.c1[2], bids.c1[2]),
                                 (bids.c2[2], bids.c2[2])]
                    set_hess!(E_ll_a[2, i], hess_ids..., E_ll_a[2, i].v[] * 4.0)
                end
                set_hess!(E_ll_a[2, i], bids.c1[2], bids.c2[2],
                          E_ll_a[2, i].v[] * -4.0)
            end
            multiply_sfs!(E_ll_a[2, i], E_ll_a[3, i])

            # Band 1 = band 2 * color 1
            E_ll_a[1, i].d[bids.c1[1]] = E_ll_a[1, i].v[] * -2.
            E_ll_a[1, i].d[bids.c2[1]] = E_ll_a[1, i].v[] * 2.
            if calculate_hessian
                for hess_ids in [(bids.c1[1], bids.c1[1]),
                                 (bids.c2[1], bids.c2[1])]
                    set_hess!(E_ll_a[1, i], hess_ids..., E_ll_a[1, i].v[] * 4.0)
                end
                set_hess!(E_ll_a[1, i], bids.c1[1], bids.c2[1],
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

    SourceBrightness(E_l_a, E_ll_a)
end


"""
Load the source brightnesses for these model params.  Each SourceBrightness
object has information for all bands and object types.

Returns:
  - An array of SourceBrightness objects for each object in 1:ea.S.  Only
    sources in ea.active_sources will have derivative information.
"""
function load_source_brightnesses{NumType <: Number}(
                    ea::ElboArgs,
                    vp::VariationalParams{NumType};
                    calculate_gradient::Bool=true,
                    calculate_hessian::Bool=true)
    sbs = Vector{SourceBrightness{NumType}}(ea.S)

    for s in 1:ea.S
        this_deriv = (s in ea.active_sources) && calculate_gradient
        this_hess = (s in ea.active_sources) && calculate_hessian
        sbs[s] = SourceBrightness(vp[s];
                                  calculate_gradient=this_deriv,
                                  calculate_hessian=this_hess)
    end

    sbs
end
