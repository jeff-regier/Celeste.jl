"""
Custom exception types used throughout Celeste.
"""
module Exceptions

"""
Some parameter to a function has invalid values. The message should explain what parameter is
invalid and why.
"""
type InvalidInputError <: Exception
    message::String
end

end
