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
    puts("[$(threadid())] DEBUG: $msg")
end

end

