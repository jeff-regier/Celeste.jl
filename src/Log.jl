"""
Thread-safe logging
"""
module Log

@inline puts(s) = ccall(:puts, Cint, (Ptr{Int8},), string(s))


function error(msg::String)
    puts("ERROR: $msg")
end

function warn(msg::String)
    puts("WARN: $msg")
end

function info(msg::String)
    puts("INFO: $msg")
end

function debug(msg::String)
    puts("DEBUG: $msg")
end

end

