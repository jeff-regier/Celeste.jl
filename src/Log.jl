"""
Thread-safe logging
"""
module Log

import Base.Threads.threadid


@inline puts(s) = ccall(:puts, Cint, (Ptr{Int8},), string(s))


function error(msg::String)
    puts("[$(threadid())] ERROR: $msg")
end

function warn(msg::String)
    puts("[$(threadid())] WARN: $msg")
end

function info(msg::String)
    puts("[$(threadid())] INFO: $msg")
end

function debug(msg::String)
    # In production mode, rather the development mode, don't log debug statements
    const is_production_run = haskey(ENV, "CELESTE_PROD") &&
                                     ENV["CELESTE_PROD"] != ""
    is_production_run || puts("[$(threadid())] DEBUG: $msg")
end

end

