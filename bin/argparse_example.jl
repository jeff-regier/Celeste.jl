using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--sources"
        help = "An array of source ids (as a parseable expression)"
        arg_type = ASCIIString
        default = "[]"
    "--image_file"
        help = "The image JLD file"
        default = "initialzed_celeste_003900_6_0269_5px.JLD"
end

parsed_args = parse_args(s)
eval(parse(string("sources = Int64", parsed_args["sources"])))
@assert length(sources) > 0
