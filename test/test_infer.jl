## test the main entry point in Celeste: the `infer` function
import Celeste

"""
test infer with a single (run, camcol, field)
"""
function test_infer_single()
    result = Celeste.infer([(3900, 6, 269)],
                           ["1237662226208063491"],
                           [datadir])
    @assert !isnull(result)
    println(get(result))
end

test_infer_single()
