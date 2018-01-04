import Celeste.ArgumentParse

@testset "argument parse" begin

@testset "positional argument" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "my_arg")

    @test_throws ArgumentParse.ArgumentParsingError ArgumentParse.parse_args(parser, String[])

    parsed_args = ArgumentParse.parse_args(parser, ["hello"])
    @test parsed_args["my_arg"] == "hello"

    @test_throws ArgumentParse.ArgumentParsingError ArgumentParse.parse_args(parser, ["foo", "bar"])
end

@testset "positional argument type conversion" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "my_arg", arg_type=Int)
    parsed_args = ArgumentParse.parse_args(parser, ["123"])
    @test parsed_args["my_arg"] == 123
end

@testset "keyword argument" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "--my-arg")

    parsed_args = ArgumentParse.parse_args(parser, String[])
    @test !haskey(parsed_args, "my-arg")

    parsed_args = ArgumentParse.parse_args(parser, ["--my-arg", "hello"])
    @test parsed_args["my-arg"] == "hello"
end

@testset "keyword argument type conversion" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "--my-arg", arg_type=Int)

    parsed_args = ArgumentParse.parse_args(parser, ["--my-arg", "123"])
    @test parsed_args["my-arg"] == 123
end

@testset "required keyword argument fails when missing" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "--my-arg", required=true)
    @test_throws ArgumentParse.ArgumentParsingError ArgumentParse.parse_args(parser, String[])
end

@testset "keyword argument default value when missing" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "--my-arg", default="hi there")
    parsed_args = ArgumentParse.parse_args(parser, String[])
    @test parsed_args["my-arg"] == "hi there"
end

@testset "keyword argument :store_true" begin
    parser = ArgumentParse.ArgumentParser(propagate_errors=true)
    ArgumentParse.add_argument(parser, "--my-arg", action=:store_true)

    parsed_args = ArgumentParse.parse_args(parser, String[])
    @test parsed_args["my-arg"] == false

    parsed_args = ArgumentParse.parse_args(parser, String["--my-arg"])
    @test parsed_args["my-arg"] == true
end

@testset "help message" begin
    parser = ArgumentParse.ArgumentParser(program_name="test.jl", propagate_errors=true)
    ArgumentParse.add_argument(parser, "--my-keyword", default="hello")
    ArgumentParse.add_argument(parser, "my_arg")
    help_message = nothing
    try
        ArgumentParse.parse_args(parser, ["--help"])
        @test false
    catch exc
        if !isa(exc, ArgumentParse.ShowHelp)
            rethrow()
        end
        help_message = exc.message
    end
    @test help_message == "Usage: test.jl [--my-keyword my-keyword] my_arg"
end

end
