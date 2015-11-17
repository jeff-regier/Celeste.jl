

@doc """
Updates sf1 in place with sf1 * sf2.

ids1 and ids2 are the ids for which sf1 and sf2 have nonzero derivatives,
respectively.
""" ->
function multiply_sf!{NumType <: Number}(
    sf1::SensitiveFloat{NumType}, sf2::SensitiveFloat{NumType};
    ids1::Array{Int64}=collect(1:length(sf1.ids)),
    ids2::Array{Int64}=collect(1:length(sf2.ids)))

  S = size(ids1.d)[2]
  sf1.v = sf1.v * sf2.v


  # Chain rule for first derivatives
  fill!(sf1.d, 0.0)
  for s=1:S
    for id1 in ids1
      sf1.d[id1, s] += sf2.v * sf1.d[id2, s]
    end
    for id2 in ids2
      sf1.d[id1, s] += sf1.v * sf2.d[id1, s]
    end
  end

  # Chain rule for second derivatives.
  for s=1:S
    fill!(sf1.hs[s], 0.0)
    for id1 in ids1
      sf1.d[id1, s] += sf2.v * sf1.d[id2, s]
    end
    for id2 in ids2
      sf1.d[id1, s] += sf1.v * sf2.d[id1, s]
    end
  end
end
