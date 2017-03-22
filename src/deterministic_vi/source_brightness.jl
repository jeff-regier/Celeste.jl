using Celeste.Model: Star, Galaxy, bids
using Celeste: @implicit_transpose, @syntactic_unroll

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
    E_l_a::Tuple{Vector{DenseHessianSSSF(BrightnessParams{Star}, NumType)},
                 Vector{DenseHessianSSSF(BrightnessParams{Galaxy}, NumType)}}

    # [E[l^2|a=0], E[l^2]|a=1]]
    E_ll_a::Tuple{Vector{DenseHessianSSSF(BrightnessParams{Star}, NumType)},
                  Vector{DenseHessianSSSF(BrightnessParams{Galaxy}, NumType)}}
end


@inline function accumulate_band!(sf, idx1, idx2, factor1, factor2, calculate_hessian)
    sf.d[idx1] = sf.v[] * factor1
    sf.d[idx2] = sf.v[] * factor2

    if calculate_hessian
        sf.h[idx1, idx1] = sf.v[] * factor1 * factor1
        sf.h[idx2, idx2] = sf.v[] * factor2 * factor2
        @implicit_transpose begin
            sf.h[idx1, idx2] = sf.v[] * factor1 * factor2
        end
    end
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
    E_l_a  = tuple(Vector{DenseHessianSSSF(BrightnessParams{Star}, NumType)}(B),
                   Vector{DenseHessianSSSF(BrightnessParams{Galaxy}, NumType)}(B))
    E_ll_a = deepcopy(E_l_a)

    @syntactic_unroll for kind in (Star(), Galaxy())
        i = (kind == Star() ? 1 : 2)
        for b = 1:B
            E_l_a[i][b] = DenseHessianSSSF(BrightnessParams{typeof(kind)}, NumType)(
                                       calculate_gradient, calculate_hessian)
        end

        E_l_a[i][3].v[] = exp(r1[i] + 0.5 * r2[i])
        E_l_a[i][4].v[] = exp(c1[3, i] + .5 * c2[3, i])
        E_l_a[i][5].v[] = exp(c1[4, i] + .5 * c2[4, i])
        E_l_a[i][2].v[] = exp(-c1[2, i] + .5 * c2[2, i])
        E_l_a[i][1].v[] = exp(-c1[1, i] + .5 * c2[1, i])

        if calculate_gradient
            # band 3 is the reference band, relative to which the colors are
            # specified.
            # It is denoted r_s and has a lognormal expectation.
            accumulate_band!(E_l_a[i][3], bids(kind).r1, bids(kind).r2, 1.0, 0.5, calculate_hessian)

            # The remaining indices involve c_s and have lognormal
            # expectations times E_c_3.

            # band 4 = band 3 * color 3.
            accumulate_band!(E_l_a[i][4], bids(kind).c1[3], bids(kind).c2[3], 1.0, 0.5, calculate_hessian)
            multiply_sfs!(E_l_a[i][4], E_l_a[i][3])

            # Band 5 = band 4 * color 4.
            accumulate_band!(E_l_a[i][5], bids(kind).c1[4], bids(kind).c2[4], 1.0, 0.5, calculate_hessian)
            multiply_sfs!(E_l_a[i][5], E_l_a[i][4])

            # Band 2 = band 3 * color 2.
            accumulate_band!(E_l_a[i][2], bids(kind).c1[2], bids(kind).c2[2], -1.0, 0.5, calculate_hessian)
            multiply_sfs!(E_l_a[i][2], E_l_a[i][3])

            # Band 1 = band 2 * color 1.
            accumulate_band!(E_l_a[i][1], bids(kind).c1[1], bids(kind).c2[1], -1.0, 0.5, calculate_hessian)
            multiply_sfs!(E_l_a[i][1], E_l_a[i][2])
        else
            # Simply update the values if not calculating derivatives.
            E_l_a[i][4].v[] *= E_l_a[i][3].v[]
            E_l_a[i][5].v[] *= E_l_a[i][4].v[]
            E_l_a[i][2].v[] *= E_l_a[i][3].v[]
            E_l_a[i][1].v[] *= E_l_a[i][2].v[]
        end # Derivs

        ################################
        # Squared terms.

        for b = 1:B
            E_ll_a[i][b] = DenseHessianSSSF(BrightnessParams{typeof(kind)}, NumType)(
                                       calculate_gradient, calculate_hessian)
        end

        E_ll_a[i][3].v[] = exp(2 * r1[i] + 2 * r2[i])
        E_ll_a[i][4].v[] = exp(2 * c1[3, i] + 2 * c2[3, i])
        E_ll_a[i][5].v[] = exp(2 * c1[4, i] + 2 * c2[4, i])
        E_ll_a[i][2].v[] = exp(-2 * c1[2, i] + 2 * c2[2, i])
        E_ll_a[i][1].v[] = exp(-2 * c1[1, i] + 2 * c2[1, i])

        if calculate_gradient
            # Band 3, the reference band.
            accumulate_band!(E_ll_a[i][3], bids(kind).r1, bids(kind).r2, 2.0, 2.0, calculate_hessian)

            # Band 4 = band 3 * color 3.
            accumulate_band!(E_ll_a[i][4], bids(kind).c1[3], bids(kind).c2[3], 2.0, 2.0, calculate_hessian)
            multiply_sfs!(E_ll_a[i][4], E_ll_a[i][3])

            # Band 5 = band 4 * color 4.
            accumulate_band!(E_ll_a[i][5], bids(kind).c1[4], bids(kind).c2[4], 2.0, 2.0, calculate_hessian)
            multiply_sfs!(E_ll_a[i][5], E_ll_a[i][4])

            # Band 2 = band 3 * color 2
            accumulate_band!(E_ll_a[i][2], bids(kind).c1[2], bids(kind).c2[2], -2.0, 2.0, calculate_hessian)
            multiply_sfs!(E_ll_a[i][2], E_ll_a[i][3])

            # Band 1 = band 2 * color 1
            accumulate_band!(E_ll_a[i][1], bids(kind).c1[1], bids(kind).c2[1], -2.0, 2.0, calculate_hessian)
            multiply_sfs!(E_ll_a[i][1], E_ll_a[i][2])
        else
            # Simply update the values if not calculating derivatives.
            E_ll_a[i][4].v[] *= E_ll_a[i][3].v[]
            E_ll_a[i][5].v[] *= E_ll_a[i][4].v[]
            E_ll_a[i][2].v[] *= E_ll_a[i][3].v[]
            E_ll_a[i][1].v[] *= E_ll_a[i][2].v[]
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
