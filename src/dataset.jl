struct BoundingBox
    ramin::Float64
    ramax::Float64
    decmin::Float64
    decmax::Float64

    function BoundingBox(ramin::Float64, ramax::Float64,
                         decmin::Float64, decmax::Float64)
        @assert ramax > ramin "ramax must be greater than ramin"
        @assert decmax > decmin "decmax must be greater than decmin"
        new(ramin, ramax, decmin, decmax)
    end
end


function BoundingBox(ramin::String, ramax::String,
                     decmin::String, decmax::String)
    BoundingBox(parse(Float64, ramin),
                parse(Float64, ramax),
                parse(Float64, decmin),
                parse(Float64, decmax))
end


"""
    SurveyDataSet

Abstract type representing a collection of imaging data and associated
metadata from a survey. Concrete subtypes are for specific
surveys. They provide methods for querying which images are available
and loading Celeste Image objects from disk. In short, they provide an
interface between survey data organized on disk and in-memory Celeste
objects.
"""
abstract type SurveyDataSet end


load_images(::SurveyDataSet, box::BoundingBox) =
    error("load_images() not defined for this SurveyDataSet type")
