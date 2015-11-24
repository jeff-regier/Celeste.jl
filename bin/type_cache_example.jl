using Celeste
using CelesteTypes

type SensitiveFloatCacheValue{NumType <: Number}
  sf1::SensitiveFloat{CanonicalParams, NumType}
  sf2::SensitiveFloat{CanonicalParams, NumType}
end

sf_cache_v1 = SensitiveFloatCacheValue(
  zero_sensitive_float(CanonicalParams, Float64),
  zero_sensitive_float(CanonicalParams, Float64));

sf_cache_v2 = SensitiveFloatCacheValue(
  zero_sensitive_float(CanonicalParams, Int64),
  zero_sensitive_float(CanonicalParams, Int64));

SensitiveFloatCacheValue(NumType::DataType) = begin
  @assert NumType <: Number
  SensitiveFloatCacheValue{NumType}(
    zero_sensitive_float(CanonicalParams, NumType),
    zero_sensitive_float(CanonicalParams, NumType));
end

sf_cache = Dict{DataType, SensitiveFloatCacheValue}()
sf_cache[Float64] = sf_cache_v1;
sf_cache[Int64] = sf_cache_v2;

function sf_cache_fetch(T::DataType)
  global sf_cache
  if haskey(sf_cache, T)
    println("Data type $T found in cache.")
    return sf_cache[T]
  else
    # Allocate memory.
    println("Data type $T not found in cache.  Initializing.")
    sf_cache_v = SensitiveFloatCacheValue(T);
    sf_cache[T] = sf_cache_v
    return sf_cache_v
  end
end

sf = sf_cache_fetch(Float64);
sf.sf1.v = 5.0

sf = sf_cache_fetch(Int64);
sf.sf1.v = 7

sf = sf_cache_fetch(Float64);
println(sf.sf1.v)

sf = sf_cache_fetch(Int32);
println(sf.sf1.v)
