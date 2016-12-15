import Celeste.Stripe82Score


function test_score_field()
    results_filename = "celeste-004263-5-0119.jld"

    if !isfile(joinpath(datadir, results_filename))
        results_url = "http://portal.nersc.gov/project/dasrepo/celeste/$results_filename"
        run(`wget --quiet -O $datadir/$results_filename $results_url`)
    end

    rcf = RunCamcolField(4263, 5, 119)
    truthfile = joinpath(datadir, "coadd_for_4263_5_119.fit")
    Stripe82Score.score_field_disk(rcf, datadir, truthfile, datadir)
end


test_score_field()
