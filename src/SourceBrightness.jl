@doc """
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
""" ->
immutable SourceBrightness{NumType <: Number}
    # [E[l|a=0], E[l]|a=1]]
    E_l_a::Matrix{SensitiveFloat{BrightnessParams, NumType}}

    # [E[l^2|a=0], E[l^2]|a=1]]
    E_ll_a::Matrix{SensitiveFloat{BrightnessParams, NumType}}
end


SourceBrightness{NumType <: Number}(
    vs::Vector{NumType}; calculate_derivs::Bool=true) = begin
  r1 = vs[ids.r1]
  r2 = vs[ids.r2]
  c1 = vs[ids.c1]
  c2 = vs[ids.c2]

  # E_l_a has a row for each of the five colors and columns
  # for star / galaxy.
  E_l_a = Array(SensitiveFloat{BrightnessParams, NumType}, B, Ia)
  E_ll_a = Array(SensitiveFloat{BrightnessParams, NumType}, B, Ia)

  for i = 1:Ia
      ids_band_3 = Int64[bids.r1[i], bids.r2[i]]
      ids_color_1 = Int64[bids.c1[1, i], bids.c2[1, i]]
      ids_color_2 = Int64[bids.c1[2, i], bids.c2[2, i]]
      ids_color_3 = Int64[bids.c1[3, i], bids.c2[3, i]]
      ids_color_4 = Int64[bids.c1[4, i], bids.c2[4, i]]

      for b = 1:B
          E_l_a[b, i] = zero_sensitive_float(BrightnessParams, NumType)
      end

      E_l_a[3, i].v = exp(r1[i] + 0.5 * r2[i])
      E_l_a[4, i].v = exp(c1[3, i] + .5 * c2[3, i])
      E_l_a[5, i].v = exp(c1[4, i] + .5 * c2[4, i])
      E_l_a[2, i].v = exp(-c1[2, i] + .5 * c2[2, i])
      E_l_a[1, i].v = exp(-c1[1, i] + .5 * c2[1, i])

      if calculate_derivs
        # band 3 is the reference band, relative to which the colors are
        # specified.
        # It is denoted r_s and has a lognormal expectation.
        E_l_a[3, i].d[bids.r1[i]] = E_l_a[3, i].v
        E_l_a[3, i].d[bids.r2[i]] = E_l_a[3, i].v * .5
        set_hess!(E_l_a[3, i], bids.r1[i], bids.r1[i], E_l_a[3, i].v)
        set_hess!(E_l_a[3, i], bids.r1[i], bids.r2[i], E_l_a[3, i].v * 0.5)
        set_hess!(E_l_a[3, i], bids.r2[i], bids.r2[i], E_l_a[3, i].v * 0.25)

        # The remaining indices involve c_s and have lognormal
        # expectations times E_c_3.

        # band 4 = band 3 * color 3.
        E_l_a[4, i].d[bids.c1[3, i]] = E_l_a[4, i].v
        E_l_a[4, i].d[bids.c2[3, i]] = E_l_a[4, i].v * .5
        set_hess!(E_l_a[4, i], bids.c1[3, i], bids.c1[3, i], E_l_a[4, i].v)
        set_hess!(E_l_a[4, i], bids.c1[3, i], bids.c2[3, i], E_l_a[4, i].v * 0.5)
        set_hess!(E_l_a[4, i], bids.c2[3, i], bids.c2[3, i], E_l_a[4, i].v * 0.25)
        multiply_sfs!(E_l_a[4, i], E_l_a[3, i], ids1=ids_color_3, ids2=ids_band_3)

        # Band 5 = band 4 * color 4.
        E_l_a[5, i].d[bids.c1[4, i]] = E_l_a[5, i].v
        E_l_a[5, i].d[bids.c2[4, i]] = E_l_a[5, i].v * .5
        set_hess!(E_l_a[5, i], bids.c1[4, i], bids.c1[4, i], E_l_a[5, i].v)
        set_hess!(E_l_a[5, i], bids.c1[4, i], bids.c2[4, i], E_l_a[5, i].v * 0.5)
        set_hess!(E_l_a[5, i], bids.c2[4, i], bids.c2[4, i], E_l_a[5, i].v * 0.25)
        multiply_sfs!(E_l_a[5, i], E_l_a[4, i],
                      ids1=ids_color_4, ids2=union(ids_band_3, ids_color_3))

        # Band 2 = band 3 * color 2.
        E_l_a[2, i].d[bids.c1[2, i]] = E_l_a[2, i].v * -1.
        E_l_a[2, i].d[bids.c2[2, i]] = E_l_a[2, i].v * .5
        set_hess!(E_l_a[2, i], bids.c1[2, i], bids.c1[2, i], E_l_a[2, i].v)
        set_hess!(E_l_a[2, i], bids.c1[2, i], bids.c2[2, i], E_l_a[2, i].v * -0.5)
        set_hess!(E_l_a[2, i], bids.c2[2, i], bids.c2[2, i], E_l_a[2, i].v * 0.25)
        multiply_sfs!(E_l_a[2, i], E_l_a[3, i], ids1=ids_color_2, ids2=ids_band_3)

        # Band 1 = band 2 * color 1.
        E_l_a[1, i].d[bids.c1[1, i]] = E_l_a[1, i].v * -1.
        E_l_a[1, i].d[bids.c2[1, i]] = E_l_a[1, i].v * .5
        set_hess!(E_l_a[1, i], bids.c1[1, i], bids.c1[1, i], E_l_a[1, i].v)
        set_hess!(E_l_a[1, i], bids.c1[1, i], bids.c2[1, i], E_l_a[1, i].v * -0.5)
        set_hess!(E_l_a[1, i], bids.c2[1, i], bids.c2[1, i], E_l_a[1, i].v * 0.25)
        multiply_sfs!(E_l_a[1, i], E_l_a[2, i],
                      ids1=ids_color_1, ids2=union(ids_band_3, ids_color_2))
      else
        # Simply update the values.
        E_l_a[4, i].v *= E_l_a[3, i].v
        E_l_a[5, i].v *= E_l_a[4, i].v
        E_l_a[2, i].v *= E_l_a[3, i].v
        E_l_a[1, i].v *= E_l_a[2, i].v
      end # Derivs

      ################################
      # Squared terms.

      for b = 1:B
          E_ll_a[b, i] = zero_sensitive_float(BrightnessParams, NumType)
      end

      E_ll_a[3, i].v = exp(2 * r1[i] + 2 * r2[i])
      E_ll_a[4, i].v = exp(2 * c1[3, i] + 2 * c2[3, i])
      E_ll_a[5, i].v = exp(2 * c1[4, i] + 2 * c2[4, i])
      E_ll_a[2, i].v = exp(-2 * c1[2, i] + 2 * c2[2, i])
      E_ll_a[1, i].v = exp(-2 * c1[1, i] + 2 * c2[1, i])

      if calculate_derivs
        # Band 3, the reference band.
        E_ll_a[3, i].d[bids.r1[i]] = 2 * E_ll_a[3, i].v
        E_ll_a[3, i].d[bids.r2[i]] = 2 * E_ll_a[3, i].v
        for hess_ids in [(bids.r1[i], bids.r1[i]),
                         (bids.r1[i], bids.r2[i]),
                         (bids.r2[i], bids.r2[i])]
          set_hess!(E_ll_a[3, i], hess_ids..., 4.0 * E_ll_a[3, i].v)
        end

        # Band 4 = band 3 * color 3.
        E_ll_a[4, i].d[bids.c1[3, i]] = E_ll_a[4, i].v * 2.
        E_ll_a[4, i].d[bids.c2[3, i]] = E_ll_a[4, i].v * 2.
        for hess_ids in [(bids.c1[3, i], bids.c1[3, i]),
                         (bids.c1[3, i], bids.c2[3, i]),
                         (bids.c2[3, i], bids.c2[3, i])]
          set_hess!(E_ll_a[4, i], hess_ids..., E_ll_a[4, i].v * 4.0)
        end
        multiply_sfs!(E_ll_a[4, i], E_ll_a[3, i],
                      ids1=ids_color_3, ids2=ids_band_3)

        # Band 5 = band 4 * color 4.
        tmp4 = exp(2 * c1[4, i] + 2 * c2[4, i])
        E_ll_a[5, i].d[bids.c1[4, i]] = E_ll_a[5, i].v * 2.
        E_ll_a[5, i].d[bids.c2[4, i]] = E_ll_a[5, i].v * 2.
        for hess_ids in [(bids.c1[4, i], bids.c1[4, i]),
                         (bids.c1[4, i], bids.c2[4, i]),
                         (bids.c2[4, i], bids.c2[4, i])]
          set_hess!(E_ll_a[5, i], hess_ids..., E_ll_a[5, i].v * 4.0)
        end
        multiply_sfs!(E_ll_a[5, i], E_ll_a[4, i],
                      ids1=ids_color_4, ids2=union(ids_band_3, ids_color_3))

        # Band 2 = band 3 * color 2
        tmp2 = exp(-2 * c1[2, i] + 2 * c2[2, i])
        E_ll_a[2, i].d[bids.c1[2, i]] = E_ll_a[2, i].v * -2.
        E_ll_a[2, i].d[bids.c2[2, i]] = E_ll_a[2, i].v * 2.
        for hess_ids in [(bids.c1[2, i], bids.c1[2, i]),
                         (bids.c2[2, i], bids.c2[2, i])]
          set_hess!(E_ll_a[2, i], hess_ids..., E_ll_a[2, i].v * 4.0)
        end
        set_hess!(E_ll_a[2, i], bids.c1[2, i], bids.c2[2, i],
                  E_ll_a[2, i].v * -4.0)
        multiply_sfs!(E_ll_a[2, i], E_ll_a[3, i],
                      ids1=ids_color_2, ids2=ids_band_3)

        # Band 1 = band 2 * color 1
        E_ll_a[1, i].d[bids.c1[1, i]] = E_ll_a[1, i].v * -2.
        E_ll_a[1, i].d[bids.c2[1, i]] = E_ll_a[1, i].v * 2.
        for hess_ids in [(bids.c1[1, i], bids.c1[1, i]),
                         (bids.c2[1, i], bids.c2[1, i])]
          set_hess!(E_ll_a[1, i], hess_ids..., E_ll_a[1, i].v * 4.0)
        end
        set_hess!(E_ll_a[1, i], bids.c1[1, i], bids.c2[1, i],
                  E_ll_a[1, i].v * -4.0)
        multiply_sfs!(E_ll_a[1, i], E_ll_a[2, i],
                      ids1=ids_color_1, ids2=union(ids_band_3, ids_color_2))
      else
        # Simply update the values.
        E_ll_a[4, i].v *= E_ll_a[3, i].v
        E_ll_a[5, i].v *= E_ll_a[4, i].v
        E_ll_a[2, i].v *= E_ll_a[3, i].v
        E_ll_a[1, i].v *= E_ll_a[2, i].v
      end # calculate_derivs
  end

  SourceBrightness(E_l_a, E_ll_a)
end


@doc """
A convenience function for getting only the brightness parameters
from model parameters.

Args:
  mp: Model parameters

Returns:
  An array of E_l_a and E_ll_a for each source.
""" ->
function get_brightness{NumType <: Number}(mp::ModelParams{NumType})
    brightness = [SourceBrightness(mp.vp[s]) for s in mp.S];
    brightness_vals = [ Float64[b.E_l_a[i, j].v for
        i=1:size(b.E_l_a, 1), j=1:size(b.E_l_a, 2)] for b in brightness]
    brightness_squares = [ Float64[b.E_l_a[i, j].v for
        i=1:size(b.E_ll_a, 1), j=1:size(b.E_ll_a, 2)] for b in brightness]

    brightness_vals, brightness_squares
end


@doc """
Load the source brightnesses for these model params.  Each SourceBrightness
object has information for all bands and object types.
""" ->
function load_source_brightnesses{NumType <: Number}(
    mp::ModelParams{NumType}, calculate_derivs::Bool)

  sbs = Array(SourceBrightness{NumType}, mp.S)
  for s in 1:mp.S
    calculate_this_deriv = (s in mp.active_sources) && calculate_derivs
    sbs[s] = SourceBrightness(mp.vp[s], calculate_derivs=calculate_this_deriv)
  end
  sbs
end
