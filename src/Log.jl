"""
Thread-safe logging
"""
module Log

import Base.Threads.threadid

const is_production_run = haskey(ENV, "CELESTE_PROD") && ENV["CELESTE_PROD"] != ""
const distributed = haskey(ENV, "USE_DTREE") && ENV["USE_DTREE"] != ""

if distributed
nodeid = Gasp.nodeid
else
nodeid = 1
end

# thread-safe print functions
@inline nputs(nid, s...) = ccall(:puts, Cint, (Cstring,), string("[$nid] ", s...))
@inline ntputs(nid, tid, s...) = ccall(:puts, Cint, (Cstring,), string("[$nid]<$tid> ", s...))

# logging functions
@inline error(msg...) = ntputs(nodeid, threadid(), "ERROR: ", msg...)
@inline warn(msg...) = ntputs(nodeid, threadid(), "WARN: ", msg...)
@inline info(msg...) = ntputs(nodeid, threadid(), "INFO: ", msg...)

# In production mode, rather the development mode, don't log debug statements
@inline debug(msg...) = is_production_run || ntputs(nodeid, threadid(), "DEBUG: ", msg...)

end

