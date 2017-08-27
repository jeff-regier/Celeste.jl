module CelesteHTTPIO
    using Celeste
    using FITSIO
    using TranscodingStreams
    import Celeste.SDSSIO: SDSSIO, SDSSImageDesc, NetworkIoStrategy, readFITS

    using HTTP
    struct HTTPStrategy <: NetworkIoStrategy
        host::String
        remote_strategy::SDSSIO.IOStrategy
    end

    function readFITS(strategy::HTTPStrategy, img::SDSSImageDesc)
        @assert Base.Threads.threadid() == 1
        fname, compression = SDSSIO.compute_fname(strategy.remote_strategy, img)
        resp = HTTP.get("http://$(strategy.host)$fname", connecttimeout = 100.0, readtimeout = 100.0)
        @assert HTTP.status(resp) == 200
        data = read(resp.body)
        (compression != nothing) && (data = transcode(compression, data))
        return FITSIO.FITS(data)
    end

end
