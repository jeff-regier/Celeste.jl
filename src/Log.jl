"""
Thread-safe logging
"""
module Log

@inline puts(s) = ccall(:puts, Cint, (Ptr{Int8},), string(s))


@inline function error(msg::String)
    #puts("ERROR: $msg")
end

@inline function warn(msg::String)
    #puts("WARN: $msg")
end

@inline function info(msg::String)
    #puts("INFO: $msg")
end

@inline function debug(msg::String)
    #puts("DEBUG: $msg")
end

end

