module ArgumentParse

const KEYWORD_ARGUMENT_PREFIX = "--"
const HELP_ARGUMENTS = ["--help", "-h"]

struct NoDefault end

struct ArgumentParsingError <: Exception
    message::String
end

struct ShowHelp <: Exception
    message::String
end

struct ArgumentSpecification
    name::String
    argument_string::String
    help::String
    arg_type::Type
    required::Bool
    default
    action::Symbol
end

struct ArgumentParser
    program_name::String
    propagate_errors::Bool
    positional_arguments::Vector{ArgumentSpecification}
    keyword_arguments::Dict{String, ArgumentSpecification}

    function ArgumentParser(; program_name=nothing, propagate_errors=false)
        if program_name == nothing
            program_name = (
                Base.source_path() != nothing ? basename(Base.source_path()) : "<PROGRAM>"
            )
        end
        new(
            program_name,
            propagate_errors,
            ArgumentSpecification[],
            Dict{String, ArgumentSpecification}()
        )
    end
end

strip_keyword_argument_prefix(name::String) = name[(length(KEYWORD_ARGUMENT_PREFIX) + 1):end]

function add_argument(
    parser::ArgumentParser, argument_string::String; help="", arg_type=String, required=nothing,
    default=NoDefault, action=:store
)
    is_keyword = startswith(argument_string, KEYWORD_ARGUMENT_PREFIX)
    if required == nothing
        required = !is_keyword
    end
    if required
        @assert default == NoDefault
    end
    if is_keyword
        name = strip_keyword_argument_prefix(argument_string)
    else
        @assert action == :store
        name = argument_string
    end
    if action == :store_true
        @assert default == NoDefault || default == false
        default = false
    end
    specification = ArgumentSpecification(
        name,
        argument_string,
        help,
        arg_type,
        required,
        default,
        action,
    )
    if is_keyword
        parser.keyword_arguments[name] = specification
    else
        push!(parser.positional_arguments, specification)
    end
end

function parse_args(parser::ArgumentParser, arguments::Vector{String})
    try
        return parse_args_helper(parser, Dict{String, Any}(), arguments)
    catch exc
        if parser.propagate_errors
            rethrow()
        elseif isa(exc, ArgumentParsingError)
            println(STDERR, exc.message)
            exit(1)
        elseif isa(exc, ShowHelp)
            println(STDERR, exc.message)
            exit(0)
        else
            rethrow()
        end
    end
end

function parse_args_helper(
    parser::ArgumentParser, parsed_args::Dict{String, Any}, arguments::Vector{String}
)
    while !isempty(arguments)
        next_argument = shift!(arguments)
        remaining_positional_args = get_unfilled_args(parser.positional_arguments, parsed_args)
        if in(next_argument, HELP_ARGUMENTS)
            show_help(parser)
        elseif startswith(next_argument, KEYWORD_ARGUMENT_PREFIX)
            name = strip_keyword_argument_prefix(next_argument)
            if !haskey(parser.keyword_arguments, name)
                parsing_error(parser, @sprintf("Unknown keyword '%s'", next_argument))
            else
                specification = parser.keyword_arguments[name]
                parse_keyword_argument(parser, parsed_args, specification, arguments)
            end
        elseif isempty(remaining_positional_args)
            parsing_error(parser, "Too many arguments")
        else
            specification = remaining_positional_args[1]
            parse_positional_argument(parser, parsed_args, specification, next_argument)
        end
    end

    remaining_required_args = vcat(
        get_unfilled_args(parser.positional_arguments, parsed_args),
        get_unfilled_args(values(parser.keyword_arguments), parsed_args),
    )
    remaining_required_args = filter(spec -> spec.required, remaining_required_args)
    if !isempty(remaining_required_args)
        remaining_names = join(
            [specification.argument_string for specification in remaining_required_args],
            ", ",
        )
        parsing_error(parser, @sprintf("Missing required arguments: %s", remaining_names))
    else
        set_defaults(parser, parsed_args)
        return parsed_args
    end
end

function get_unfilled_args(specifications, parsed_args::Dict{String, Any})
    if isempty(specifications)
        return []
    end
    mapreduce(vcat, specifications) do specification::ArgumentSpecification
        if !haskey(parsed_args, specification.name)
            [specification]
        else
            []
        end
    end
end

function parse_keyword_argument(
    parser::ArgumentParser, parsed_args::Dict{String, Any}, specification::ArgumentSpecification,
    following_arguments::Vector{String}
)
    if specification.action == :store_true
        parsed_args[specification.name] = true
    else
        @assert specification.action == :store
        if isempty(following_arguments)
            parsing_error(
                parser,
                @sprintf("Missing value for argument %s", specification.argument_string),
            )
        else
            raw_value = shift!(following_arguments)
            parsed_args[specification.name] = parse_value(specification, raw_value)
        end
    end
end

function parse_positional_argument(
    parser::ArgumentParser, parsed_args::Dict{String, Any}, specification::ArgumentSpecification,
    raw_value::String
)
    parsed_args[specification.name] = parse_value(specification, raw_value)
end

function parse_value(specification::ArgumentSpecification, raw_value::String)
    if specification.arg_type == String
        return raw_value
    else
        return parse(specification.arg_type, raw_value)
    end
end

function set_defaults(parser::ArgumentParser, parsed_args::Dict{String, Any})
    all_specifications = vcat(
        parser.positional_arguments,
        collect(values(parser.keyword_arguments)),
    )
    for specification in all_specifications
        if !haskey(parsed_args, specification.name) && specification.default != NoDefault
            parsed_args[specification.name] = specification.default
        end
    end
end

function parsing_error(parser::ArgumentParser, message::String)
    throw(ArgumentParsingError(@sprintf("ERROR: %s\n%s", message, format_help(parser))))
end

function show_help(parser::ArgumentParser)
    throw(ShowHelp(format_help(parser)))
end

function format_help(parser::ArgumentParser)
    keyword_args = map(values(parser.keyword_arguments)) do specification
        display = @sprintf(
            "%s%s",
            specification.argument_string,
            specification.action == :store ? @sprintf(" %s", specification.name) : "",
        )
        if !specification.required
            display = "[$display]"
        end
        display
    end
    positional_args = map(parser.positional_arguments) do specification
        if !specification.required
            "[$(specification.argument_string)]"
        else
            specification.argument_string
        end
    end
    display_parts = []
    if length(keyword_args) > 0
        append!(display_parts, keyword_args)
    end
    if length(positional_args) > 0
        append!(display_parts, positional_args)
    end
    lines = [
        @sprintf("Usage: %s %s", parser.program_name, join(display_parts, " "))
    ]
    join(lines, "\n")
end

export ArgumentParser, add_argument, parse_args

end # module ArgumentParse
