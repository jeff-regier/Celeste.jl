module Sandbox

import Celeste.AccuracyBenchmark
import Celeste.GalsimBenchmark

function main()

srand(12345)

truth_catalog, single_predictions = GalsimBenchmark.run_benchmarks(
    joint_inference=false,
)
unused, joint_predictions = GalsimBenchmark.run_benchmarks(
    joint_inference=true,
)

score_df = AccuracyBenchmark.score_predictions(
    truth_catalog,
    [single_predictions, joint_predictions],
)
println(repr(score_df))

end
end
