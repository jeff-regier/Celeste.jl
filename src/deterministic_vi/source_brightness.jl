using Celeste.Model: Star, Galaxy, bids
using Celeste: @implicit_transpose, @syntactic_unroll
import Celeste: has_gradient, has_hessian

const ElType{NumType, HasGradient, HasHessian} = Tuple{Vector{DenseHessianSSSF(BrightnessParams{Star}, NumType, HasGradient, HasHessian)},
             Vector{DenseHessianSSSF(BrightnessParams{Galaxy}, NumType, HasGradient, HasHessian)}}

immutable SourceBrightness{NumType <: Number, HasGradient, HasHessian}
    # [E[l|a=0], E[l]|a=1]]
    E_l_a::ElType{NumType, HasGradient, HasHessian}

    # [E[l^2|a=0], E[l^2]|a=1]]
    E_ll_a::ElType{NumType, HasGradient, HasHessian}
    SourceBrightness{NumType, HasGradient, HasHessian}(E_l_a::ElType{NumType}, E_ll_a::ElType{NumType}) where {NumType, HasGradient, HasHessian} = new{NumType, HasGradient, HasHessian}(E_l_a, E_ll_a)
end
has_gradient(sbs::SourceBrightness{<:Any, G}) where {G} = G
has_hessian(sbs::SourceBrightness{<:Any, G, H}) where {G, H} = H

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

function SourceBrightness{NumType, calculate_gradient, calculate_hessian}() where {NumType, calculate_gradient, calculate_hessian}
    # E_l_a has a row for each of the five colors and columns
    # for star / galaxy.
    E_l_a  = tuple(Vector{DenseHessianSSSF(BrightnessParams{Star}, NumType, calculate_gradient, calculate_hessian)}(B),
                   Vector{DenseHessianSSSF(BrightnessParams{Galaxy}, NumType, calculate_gradient, calculate_hessian)}(B))
    E_ll_a = deepcopy(E_l_a)
    @syntactic_unroll for kind in (Star(), Galaxy())
        i = (kind == Star() ? 1 : 2)
        for b = 1:B
            E_l_a[i][b] = DenseHessianSSSF(BrightnessParams{typeof(kind)},
              NumType, calculate_gradient, calculate_hessian)()
        end
        
        for b = 1:B
            E_ll_a[i][b] = DenseHessianSSSF(BrightnessParams{typeof(kind)},
              NumType, calculate_gradient, calculate_hessian)()
        end
    end
  
    SourceBrightness{NumType, calculate_gradient, calculate_hessian}(E_l_a, E_ll_a)
end

function SourceBrightness(vs::Vector{NumType},
                          calculate_gradient=true,
                          calculate_hessian=true) where NumType
    SourceBrightness!(SourceBrightness{NumType}(calculate_gradient, calculate_hessian),
      vs, calculate_gradient, calculate_hessian)
end

function SourceBrightness!{NumType <: Number}(sbs::SourceBrightness{NumType},
                                             vs::Vector{NumType}, skip_gradient=false, skip_hessian=false)
    r1 = vs[ids.r1]
    r2 = vs[ids.r2]
    c1 = vs[ids.c1]
    c2 = vs[ids.c2]
    
    E_l_a = sbs.E_l_a
    E_ll_a = sbs.E_ll_a
    
    calculate_hessian = has_hessian(sbs) && !skip_hessian

    @syntactic_unroll for kind in (Star(), Galaxy())
        i = (kind == Star() ? 1 : 2)
        for b = 1:B
            clear!(E_l_a[i][b])
        end

        E_l_a[i][3].v[] = exp(r1[i] + 0.5 * r2[i])
        E_l_a[i][4].v[] = exp(c1[3, i] + .5 * c2[3, i])
        E_l_a[i][5].v[] = exp(c1[4, i] + .5 * c2[4, i])
        E_l_a[i][2].v[] = exp(-c1[2, i] + .5 * c2[2, i])
        E_l_a[i][1].v[] = exp(-c1[1, i] + .5 * c2[1, i])

        if has_gradient(sbs) && !skip_gradient        
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
            clear!(E_ll_a[i][b])
        end

        E_ll_a[i][3].v[] = exp(2 * r1[i] + 2 * r2[i])
        E_ll_a[i][4].v[] = exp(2 * c1[3, i] + 2 * c2[3, i])
        E_ll_a[i][5].v[] = exp(2 * c1[4, i] + 2 * c2[4, i])
        E_ll_a[i][2].v[] = exp(-2 * c1[2, i] + 2 * c2[2, i])
        E_ll_a[i][1].v[] = exp(-2 * c1[1, i] + 2 * c2[1, i])

        if has_gradient(sbs) && !skip_gradient
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

    sbs
end


"""
Load the source brightnesses for these model params.  Each SourceBrightness
object has information for all bands and object types.

Returns:
  - An array of SourceBrightness objects for each object in 1:ea.S.  Only
    sources in ea.active_sources will have derivative information.
"""
function load_source_brightnesses!{NumType <: Number}(
                    sbs::Vector{<:SourceBrightness{NumType}},
                    ea::ElboArgs,
                    vp::VariationalParams{NumType})
    for s in 1:ea.S
        this_deriv = (s in ea.active_sources)
        this_hess = (s in ea.active_sources)
        sbs[s] = SourceBrightness!(sbs[s], vp[s], !this_deriv, !this_hess)
    end

    sbs
end

function load_source_brightnesses(ea::ElboArgs,
          vp::VariationalParams{NumType},
          calculate_gradient::Bool=true,
          calculate_hessian::Bool=true) where NumType
    sbs = SourceBrightness{NumType}[SourceBrightness{NumType, calculate_gradient, calculate_hessian}() for _ in 1:ea.S]
    load_source_brightnesses!(sbs, ea, vp)
end
