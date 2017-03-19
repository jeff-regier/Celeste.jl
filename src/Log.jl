"""
Thread-safe logging
"""
module Log

import Base.Threads.threadid

const is_production_run = haskey(ENV, "CELESTE_PROD") && ENV["CELESTE_PROD"] != ""
const distributed = haskey(ENV, "USE_DTREE") && ENV["USE_DTREE"] != ""

if distributed
import Gasp.grank
else
grank() = 1
end

# thread-safe print function
@inline puts(s...) = ccall(:puts, Cint, (Cstring,), string(s...))
@inline rtputs(s...) = puts("[$(grank())]<$(threadid())> ", s...)


# logging functions
@inline message(msg...) = rtputs(msg...)
@inline one_message(msg...) = grank() == 1 && puts(msg...) 
@inline error(msg...) = rtputs("ERROR: ", msg...)
@inline warn(msg...) = rtputs("WARN: ", msg...)

# In production mode, rather the development mode, don't log debug or info statements
@inline info(msg...) = is_production_run || rtputs("INFO: ", msg...)
@inline debug(msg...) = is_production_run || rtputs("DEBUG: ", msg...)

# Like `error()`, but include exception info and stack trace. Should only be called from a `catch`
# block, e.g.,
# try
#   ...
# catch ex
#   Log.exception(ex, catch_stacktrace(), "Something happened %s", some_var)
# end
function exception(exception::Exception, msg...)
    if length(msg) > 0
        error(msg...)
    end
    error(exception)
    if !is_production_run
        error("Stack trace:")
        stack_trace = catch_stacktrace()
        if length(stack_trace) > 100
            stack_trace = vcat(
                [string(line) for line in stack_trace[1:50]],
                @sprintf("...(removed %d frames)...", length(stack_trace) - 100),
                [string(line) for line in stack_trace[(length(stack_trace) - 50):length(stack_trace)]],
            )
        end
        for stack_line in stack_trace
            error(@sprintf("  %s", stack_line))
        end
    end
end

end

