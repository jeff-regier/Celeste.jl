"""generate a new sample from a probability density using slice sampling

Parameters
----------
init_x : array
logprob : callable, `lprob = logprob(x, *logprob_args)`
    A functions which returns the log probability at a given
    location

Returns
-------
new_x : float
    the sampled position
new_llh : float 
    the log likelihood at the new position
Notes
-----
http://en.wikipedia.org/wiki/Slice_sampling
"""
function slicesample(init_x::Vector{Float64},
                     logprob::Function;
                     sigma = 1.,
                     step_out=true,
                     max_steps_out=10,
                     compwise=true,
                     numdir=2,
                     doubling_step = true,
                     verbose = false,
                     upper_bound = Inf,
                     lower_bound = -Inf)

    ## define a univariate directional slice sampler
    function direction_slice(direction, init_x)

        function dir_logprob(z)
            return logprob(direction*z + init_x)
        end

        function acceptable(z, llh_s, L, U)
            #println(" ... entered acceptable... ")
            starting_width = U - L
            Lt, Ut = L, U
            iter = 0
            while ((Ut-Lt) > 1.1*sigma) && ((Ut - Lt) < 1.1*starting_width)
                middle = 0.5*(Lt + Ut)
                splits = ((middle > 0) && (z >= middle)) ||
                         ((middle <= 0) && (z < middle))
                if z < middle
                    Ut = middle
                else
                    Lt = middle
                end
                # Probably these could be cached from the stepping out.
                if splits && (llh_s >= dir_logprob(Ut)) && (llh_s >= dir_logprob(Lt))
                    #println(" ... leaving acceptable... ")
                    return false
                end

                # catch infinite loops
                if iter > 1000
                  throw("acceptable caught in a loop --- exiting")
                else
                  iter += 1
                end
                if (iter > 100) && (iter % 100 == 0)
                  @printf "stuck in acceptable: interval = %2.4f; 1.1*sigma = %2.4f\n" (Ut - Lt) 1.1*sigma
                  println("  interval: ", Ut - Lt)
                  println("  starting width: ", starting_width)
                end
            end
            #println(" ... leaving acceptable... ")
            return true
        end

        # if we have upper/lower bounds on parameters, compute them in z space
        #if compwise:
        #    dir_upper_bound = upper_bound[direction==1.]
        #    dir_lower_bound = lower_bound[direction==1.]
        #else:
        #    dir_upper_bound = np.min(np.sign(direction)*(upper_bound - init_x)/direction)
        #    dir_lower_bound = np.max(np.sign(direction)*(lower_bound - init_x)/direction)

        # compute initial interval bounds (of length sigma)
        upper = sigma * rand()
        lower = upper - sigma

        # sample uniformly under the probability at z = 0
        #llh_s = log(rand()) + dir_logprob(0.0)
        llh_s = dir_logprob(0.0) - randexp() # equivalent to + log(rand())

        # perform the stepping out or doubling procedure to compute
        # interval I = [lower, upper]
        l_steps_out, u_steps_out = 0, 0
        if step_out
            if doubling_step
                while ((dir_logprob(lower) > llh_s) | (dir_logprob(upper) > llh_s)) &
                      ((l_steps_out + u_steps_out) < max_steps_out)
                    if rand() < 0.5
                        l_steps_out += 1
                        lower       -= (upper-lower)
                    else
                        u_steps_out += 1
                        upper       += (upper-lower)
                    end
                end
            else
                while (dir_logprob(lower) > llh_s) & (l_steps_out < max_steps_out)
                    l_steps_out += 1
                    lower       -= sigma
                end
                while (dir_logprob(upper) > llh_s) & (u_steps_out < max_steps_out)
                    u_steps_out += 1
                    upper       += sigma
                end
            end
        end

        # uniformly sample - perform shrinkage (with accept check) on
        # interval I = [lower, upper]
        start_lower, start_upper = lower, upper
        if (start_upper - start_lower) > 1e5
            println("  ... pre-shrinkage interval size: ", start_upper-start_lower)
            println("  ... lower ", start_lower)
            println("  ... upper ", start_upper)
            println("  ... num steps out ....", l_steps_out + u_steps_out)
            println("  ... dir ", direction)
        end
        steps_in, new_z, new_llh = 0, 0., -Inf
        while true
            steps_in += 1
            if steps_in % 100 == 0
                println("shrinking, steps", steps_in)
            end

            # sample uniformly in the interval
            new_z   = (upper - lower)*rand() + lower
            new_llh = dir_logprob(new_z)
            if isnan(new_llh)
                println("new_z, new_th, new_llh: ", new_z, ", ",
                        direction*new_z + init_x, ", ", new_llh)
                println("init_x, llh_s, ll_init_x", init_x, ", ",
                        llh_s, ", ", logprob(init_x))
                throw("Slice sampler got a NaN")
            end

            # accept (break) otherwise shrink the interval
            if (llh_s < new_llh) &&
                  acceptable(new_z, llh_s, start_lower, start_upper)
                break
            elseif new_z < 0
                lower = new_z
            elseif new_z > 0
                upper = new_z
            else
                throw("Slice sampler shrank to zero!")
            end
        end
        #println(" new z, llh post: ", new_z, new_llh)

        #if verbose:
        #    print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in
        return new_z*direction + init_x, new_llh
    end

    ## expand upper and lower bound to be same dimension as samples
    #if type(upper_bound) == float or isinstance(upper_bound, np.number):
    #    upper_bound = np.tile(upper_bound, init_x.shape)
    #if type(lower_bound) == float or isinstance(lower_bound, np.number):
    #    lower_bound = np.tile(lower_bound, init_x.shape)
    ## make sure we're starting in a legal spot
    #assert np.all(init_x < upper_bound), "init_x (%s) >= ub (%s)" % \
    #    (np.array_str(init_x), np.array_str(upper_bound))
    #assert np.all(init_x > lower_bound), "init_x (%s) <= lb (%s)" % \
    #    (np.array_str(init_x), np.array_str(lower_bound))

    # do either component-wise slice sampling, or random direction
    dims = length(init_x)
    new_x, new_llh = nothing, nothing
    if compwise
        ordering = shuffle(1:dims)
        new_x = copy(init_x)
        if verbose
            println("compwise call: ")
        end
        for d in ordering
            if verbose
                println("  ... slice sampling component ", d)
            end
            direction    = zeros(dims)
            direction[d] = 1.0
            new_x, new_llh = direction_slice(direction, new_x)
        end
    else
        new_x = copy(init_x)
        for d in 1:numdir
            direction    = randn(dims)
            direction    = direction / sqrt(sum(direction.^2))
            new_x, new_llh = direction_slice(direction, new_x)
        end
    end

    return new_x, new_llh
end


"""
Helper that draws N samples using slice sampling
"""
function slicesample_chain(lnpdf, th, N;
                           print_skip=50,
                           verbose=false)
    D = length(th)
    samps = zeros(N, D)
    lls   = zeros(N)
    t0 = time_ns()
    Log.info(@sprintf "  iter : \t loglike")
    for n in 1:N
        th, ll = slicesample(th, lnpdf;
                             doubling_step=true,
                             compwise=true,
                             verbose=verbose)
        samps[n,:] = th
        lls[n]     = ll
        if mod(n, print_skip) == 0
            Log.info(@sprintf "   %d   : \t %2.4f" n ll)
        end
    end
    elapsed = 1e-9 * (time_ns() - t0)
    Log.info(@sprintf "%2.3f ms per sample (%d samples in %2.3f seconds) \n" 1000*(elapsed/N) N elapsed)
    return samps, lls
end
