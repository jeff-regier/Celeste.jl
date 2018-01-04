"""
Thread-safe logging.
"""
module Log

import Base.Threads.threadid

# logging levels in increasing verbosity
@enum LogLevel ERROR WARN INFO DEBUG

const LEVEL = Ref{LogLevel}(INFO)
const VERBOSE = Ref{Bool}(true)  # print stack traces from errors

# Rank for multinode functionality (can be set on startup)
const rank = Ref{Int}(1)
grank() = rank[]

# thread-safe print function
@inline function puts(s...)
    data = string(s..., '\n')
    ccall(:write, Cint, (Cint, Cstring, Csize_t), 1, data, sizeof(data))
end
@inline rtputs(s...) = puts("[$(grank())]<$(threadid())> ", s...)

# logging functions
@inline error(msg...) = LEVEL[] >= ERROR && rtputs("ERROR: ", msg...)
@inline warn(msg...) = LEVEL[] >= WARN && rtputs("WARN: ", msg...)
@inline info(msg...) = LEVEL[] >= INFO && rtputs("INFO: ", msg...)
@inline debug(msg...) = LEVEL[] >= DEBUG && rtputs("DEBUG: ", msg...)


"""
    exception(ex::Exception, msg...)

If the logging level is ERROR or greater, log the message `msg` and
show the exception `ex`. If `Log.VERBOSE[]` is true, also log the
stack trace. Should only be called from a `catch` block, e.g.,

```
try
    ...
catch ex
    Log.exception(ex, "Something", " happened.")
end
```
"""
function exception(exception::Exception, msg...)
    LEVEL[] >= ERROR || return
    if length(msg) > 0
        error(msg...)
    end
    if VERBOSE[]
        stack_trace = catch_stacktrace()
    end
    buf = IOBuffer()
    Base.showerror(buf, exception)
    error(String(take!(buf)))
    if VERBOSE[]
        error("Stack trace:")
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
