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
                     sigma = 1.0,
                     step_out=true,
                     max_steps_out=1000,
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
            while (U-L) > 1.1*sigma
                middle = 0.5*(L+U)
                splits = ((middle > 0) & (z >= middle)) | 
                         ((middle <= 0) & (z < middle))
                if z < middle
                    U = middle
                else
                    L = middle
                end
                # Probably these could be cached from the stepping out.
                if splits & (llh_s >= dir_logprob(U)) & (llh_s >= dir_logprob(L))
                    return false
                end
            end
            return true
        end

        # if we have upper/lower bounds on parameters, compute them in z space
        #if compwise:
        #    dir_upper_bound = upper_bound[direction==1.]
        #    dir_lower_bound = lower_bound[direction==1.]
        #else:
        #    dir_upper_bound = np.min(np.sign(direction)*(upper_bound - init_x)/direction)
        #    dir_lower_bound = np.max(np.sign(direction)*(lower_bound - init_x)/direction)

        # compute initial interval bounds
        upper = sigma * rand()
        lower = upper - sigma

        # sample uniformly under the probability at z = 0
        llh_s = log(rand()) + dir_logprob(0.0)

        # perform the stepping out or doubling procedure to compute interval I
        l_steps_out = 0
        u_steps_out = 0
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

        # uniformly sample - perform shrinkage (with accept check)
        start_upper = upper
        start_lower = lower
        steps_in = 0
        new_z    = 0.
        new_llh  = -Inf
        while true
            steps_in += 1
            if steps_in % 100 == 0
                println("shrinking, steps", steps_in)
            end

            new_z     = (upper - lower)*rand() + lower
            new_llh   = dir_logprob(new_z)
            if isnan(new_llh)
                println("new_z: ", new_z)
                println("new_th: ", direction*new_z + init_x)
                println("new_llh: ", new_llh)
                println("llh_s: ", llh_s)
                println("init_x", init_x)
                println("ll_init_x", logprob(init_x))
                throw("Slice sampler got a NaN")
            end

            # accept/or shrink
            if (new_llh > llh_s) & acceptable(new_z, llh_s, start_lower, start_upper)
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
    tic()
    @printf "  iter : \t loglike \n"
    for n in 1:N
        th, ll = slicesample(th, lnpdf;
                             doubling_step=true,
                             compwise=true,
                             verbose=verbose)
        samps[n,:] = th
        lls[n]     = ll
        if mod(n, print_skip) == 0
            @printf "   %d   : \t %2.4f \n" n ll
        end
    end
    elapsed = toc()
    @printf "%2.3f ms per sample (%d samples in %2.3f seconds) \n" 1000*(elapsed/N) N elapsed
    return samps, lls
end
