"""
Annealed Importance Sampling. Estimate the partition function of lnpdf

Args:
    - lnpdf     : target distribution unnomralized log prob.  Potentially
                  vectorized --- takes in z = D-dimensional tensor, outputs
                  joint log prob
    - lnpdf0    : initial distribution log likelihood --- same input and
                  output as lnpdf
    - z0        : D-dimensional initial sample
    - step      : function that takes in a current state and an
                  unnormalized log probability function, performs
                  an MCMC transition (in place)
    - init_logZ : (optional) initial log partition (unused for now)
    - schedule  : temperature schedule, monotonically increasing from 0 to 1
"""
function ais(lnpdf, lnpdf0, step, z0;
             init_logZ=nothing, schedule=nothing, verbose=false)
    # set up initial log partition function for p(z) --- should be ln(1) = 0
    D = length(z0)

    # set up and check schedule
    if schedule == nothing
        schedule = collect(linspace(0, 1, 10))   # default to 10 steps ...
    end
    @assert(isapprox(schedule[1], 0.), "schedule must start with temp = 0.")
    @assert(isapprox(schedule[end], 1.), "schedule must end with temp = 1.")

    # set up function that returns intermediate distributions
    function lnpdf_t(z, t)
        # do this to catch the 0 * -Inf = NaN cases
        if t==0.
            return lnpdf0(z)
        elseif t==1.
            return lnpdf(z)
        end
        return t*lnpdf(z) + (1.-t)*lnpdf0(z)
    end

    # run AIS: track the ratio at each step, and current z
    z = deepcopy(z0)
    llratios = zeros(length(schedule) - 1)
    accepted = 0.
    for ti in 2:length(schedule)
        # unpack previous and current temperatures
        tprev, tcurr = schedule[ti-1], schedule[ti]
        if ti % 25 == 0
            Log.info(@sprintf "  temp %2.3f (%d/%d): logpost = %2.4f  logprior = %2.4f" tcurr ti length(schedule) lnpdf(z) lnpdf0(z))
        end

        # generate zt | zt-1 using tcurr --- this leaves the distribution
        #   lnp_t = tcurr*lnpdf(z) + (1-tcurr)*lnpdf0(0) invariant
        z, _ = step(z, z -> lnpdf_t(z, tcurr))  # returns z, and aux info

        # record log ratio of distributions
        llprev = lnpdf_t(z, tprev)
        llcurr = lnpdf_t(z, tcurr)
        llratios[ti-1] = llcurr - llprev
    end

    # return z (final posterior sample) and weight
    return z, sum(llratios), llratios, float(accepted)/(length(schedule))
end


"""broadcasting logsumexp"""
function logsumexp(mat; dim=1)
    max_score = maximum(mat, dim)
    lse = log.(sum(exp.(broadcast(-, mat, max_score)), dim))
    return broadcast(+, lse, max_score)
end


""" compute a bootstrap sample of the normalizing constant """
function bootstrap_lnZ(lnZ_samps; num_bootstrap=100)
    lnZ    = zeros(num_bootstrap)
    nsamps = length(lnZ_samps)
    for n in 1:num_bootstrap
        boot_sample = [lnZ_samps[rand(1:nsamps)] for n in 1:nsamps]
        lnZ[n]      = logsumexp(boot_sample)[1] - log(nsamps)
    end
    return lnZ
end


""" copy of sigmoid schedule from https://github.com/tonywu95/eval_gen/blob/master/algorithms/ais.py
This is defined as:
      gamma_t = sigma(rad * (2t/T - 1))
      beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),
where sigma is the logistic sigmoid. This schedule allocates more distributions near
the inverse temperature 0 and 1, since these are often the places where the distributon
changes the fastest.
"""
function sigmoid_schedule(num_steps; rad=4)
    if num_steps == 1
        return collect(linspace(0, 1, 2))
    end
    t    = collect(linspace(-rad, rad, num_steps))
    sigm = 1. ./ (1. .+ exp.(-t))
    return (sigm - minimum(sigm)) / (maximum(sigm) - minimum(sigm))
end


""" Run AIS multiple times to get multiple posterior samples, and a tighter
marginal likelihood estimate """
function ais_slicesample(logposterior::Function,
                         logprior::Function,
                         prior_sample::Function;
                         schedule::Array{Float64,1}=nothing,
                         num_temps::Int=50,
                         num_samps::Int=10,
                         num_bootstrap::Int=5000,
                         num_samples_per_step::Int=1)
    if schedule == nothing
        schedule = sigmoid_schedule(num_temps; rad=1)
    end

    function step(z, lnpdf)
        for i in 1:num_samples_per_step
            z, llh = MCMC.slicesample(z, lnpdf)
        end
        return z, 0.
    end

    D = length(prior_sample())
    zsamps = zeros(D, num_samps)
    wsamps = zeros(num_samps)
    for n in 1:num_samps
        Log.info(@sprintf "ais samp %d / %d" n num_samps)
        z0   = prior_sample()
        #step = (z, lnpdf) -> MCMC.slicesample(z, lnpdf)
        z, w, llrats, _ = ais(logposterior, logprior, step, z0; schedule=schedule)
        zsamps[:,n] = z
        wsamps[n]   = w
    end

    # estimate the partition function and get a bootstrap confidence interval
    lnZ  = logsumexp(wsamps, dim=1)[1] - log(num_samps)
    lnZs = bootstrap_lnZ(wsamps; num_bootstrap=num_bootstrap)
    res = Dict(:lnZ => lnZ, :lnZ_bootstrap => lnZs,
               :zsamps => zsamps, :lnZsamps => wsamps)
    return res
end
