import Celeste.Stripe82Score


function test_infer_field_and_score_object()
    rcf = RunCamcolField(4263, 5, 119)
    objid = "1237663784734490644"
    ParallelRun.infer_field(rcf, datadir, datadir; objid=objid)
    truthfile = joinpath(datadir, "coadd_for_4263_5_119.fit")
    Stripe82Score.score_object_disk(rcf, objid, datadir, truthfile, datadir)
end


function test_score_field()
    results_filename = "celeste-004263-5-0119.jld"

    if !isfile(joinpath(datadir, results_filename))
        results_url = "https://www.dropbox.com/s/8jcdmluahsf38tm/celeste-004263-5-0119.jld"
        run(`wget --quiet -O $datadir/$results_filename $results_url`)
    end

    rcf = RunCamcolField(4263, 5, 119)
    truthfile = joinpath(datadir, "coadd_for_4263_5_119.fit")
    Stripe82Score.score_field_disk(rcf, datadir, truthfile, datadir)
end


test_score_field()
test_infer_field_and_score_object()
