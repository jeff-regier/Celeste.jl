#!/usr/bin/env julia

using DataFrames

import Celeste.GalsimBenchmark
import Celeste.DeterministicVI: infer_source
import Celeste.DeterministicVIImagePSF: infer_source_fft

srand(12345)

results = GalsimBenchmark.run_field()
writetable("galsim_field_results.csv", results)

error_summaries = by(results, :field) do df
    field = df[1, :field]
    @assert all(isna(df[:error_sds])) || all(!isna(df[:error_sds]))
    is_map_estimate = all(isna(df[:error_sds]))
    if is_map_estimate
        errors = abs(df[:estimate] .- df[:ground_truth])
    else
        errors = df[:error_sds]
    end
    DataFrame(
        pct25=quantile(errors, 0.25),
        pct50=quantile(errors, 0.5),
        pct75=quantile(errors, 0.75),
        max=maximum(errors),
        mean=mean(errors),
        frac_within_1=is_map_estimate ? NA : mean(errors .<= 1),
        frac_within_2=is_map_estimate ? NA : mean(errors .<= 2),
    )
end
println(repr(error_summaries))
