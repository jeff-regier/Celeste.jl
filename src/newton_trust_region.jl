# TODO: This is intended to be used in Optim.jl.  Until that is submitted
# and merged, I'll include the working file in Celeste so the build passes.
using Optim.update!
using Optim.OptimizationTrace
using Optim.assess_convergence
using Optim.MultivariateOptimizationResults
using Optim.TwiceDifferentiableFunction

import Logging


macro newton_tr_trace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(gr)
                dt["h(x)"] = copy(H)
                dt["delta"] = copy(delta)
            end
            grnorm = norm(gr, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end


# Check whether we are in the "hard case".
#
# Args:
#  H_eigv: The eigenvalues of H, low to high
#  qg: The inner product of the eigenvalues and the gradient in the same order
#
# Returns:
#  hard_case: Whether it is a candidate for the hard case
#  lambda_1_multiplicity: The number of times the lowest eigenvalue is repeated,
#                         which is only correct if hard_case is true.
function check_hard_case_candidate(H_eigv, qg)
  @assert length(H_eigv) == length(qg)
  if H_eigv[1] >= 0
    # The hard case is only when the smallest eigenvalue is negative.
    return false, 1
  end
  hard_case = true
  lambda_index = 1
  hard_case_check_done = false
  while !hard_case_check_done
    if lambda_index > length(H_eigv)
      hard_case_check_done = true
    elseif abs(H_eigv[1] - H_eigv[lambda_index]) > 1e-10
      # The eigenvalues are reported in order.
      hard_case_check_done = true
    else
      if abs(qg[lambda_index]) > 1e-10
        hard_case_check_done = true
        hard_case = false
      end
      lambda_index += 1
    end
  end

  hard_case, lambda_index - 1
end

# Choose a point in the trust region for the next step using
# the interative (nearly exact) method of section 4.3 of Nocedal and Wright.
# This is appropriate for Hessians that you factorize quickly.
#
# TODO: Allow the user to specify their own function for the subproblem.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size
#  tolerance: The convergence tolerance for root finding
#  max_iters: The maximum number of root finding iterations
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  lambda - The chosen regularizing quantity
#  s, the step to take from the current x, is updated in place.
function solve_tr_subproblem!{T}(gr::Vector{T},
                                 H::Matrix{T},
                                 delta::T,
                                 s::Vector{T};
                                 tolerance=1e-10, max_iters=5)
    n = length(gr)
    @assert n == length(s)
    @assert (n, n) == size(H)
    delta2 = delta ^ 2

    H_eig = eigfact(H)
    lambda_1 = H_eig[:values][1]

    # TODO: don't call this max_lambda to avoid confusion with min_lambda
    max_lambda = H_eig[:values][n]
    @assert(max_lambda > 0,
            string("Last eigenvalue is <= 0 and Hessian is not positive ",
                   "semidefinite.  (Check that the Hessian is symmetric.)  ",
                   "Eigenvalues: $(H_eig[:values])"))

    # Cache the inner products between the eigenvectors and the gradient.
    qg = Array(T, n)
    for i=1:n
      qg[i] = vecdot(H_eig[:vectors][:, i], gr)
    end

    # Function 4.39 in N&W
    function p_mag2(lambda, min_i)
      p_sum = 0.
      for i = min_i:n
        p_sum = p_sum + (qg[i] ^ 2) / ((lambda + H_eig[:values][i]) ^ 2)
      end
      p_sum
    end

    if p_mag2(0.0, 1) <= delta2
      # No shrinkage is necessary, and -(H \ gr) is the solution.
      s[:] = -(H_eig[:vectors] ./ H_eig[:values]') * H_eig[:vectors]' * gr
      lambda = 0.0
      interior = true
      Logging.debug("Interior")
    else
      interior = false
      Logging.debug("Boundary")

      # The hard case is when the gradient is orthogonal to all
      # eigenvectors associated with the lowest eigenvalue.
      hard_case_candidate, lambda_1_multiplicity =
        check_hard_case_candidate(H_eig[:values], qg)

      # Solutions smaller than this are not allowed.
      # Rather than >= 0, constrain to be at least
      # within 1e-12 of the largest eigenvalue to guarantee that the
      # matrix can be inverted.
      # TODO: What is the right way to do this?
      min_lambda = max(-lambda_1, 0.0) + max_lambda * 1e-12
      lambda = min_lambda

      hard_case = false
      if hard_case_candidate
        # The "hard case".  lambda is taken to be -lambda_1 and we only need
        # to find a multiple of an orthogonal eigenvector that lands the
        # iterate on the boundary.

        # Formula 4.45 in N&W
        p_lambda2 = p_mag2(lambda, lambda_1_multiplicity + 1)
        Logging.debug("lambda_1 = $(lambda_1), p_lambda2 = $(p_lambda2), ",
                "$delta2, $lambda_1_multiplicity")
        if p_lambda2 > delta2
          # Then we can simply solve using root finding.  Set a starting point
          # between the minimum and largest eigenvalues.
          # TODO: is there a better starting point?
          hard_case = false
          lambda = min_lambda + 0.01 * (max_lambda - min_lambda)
        else
          Logging.debug("Hard case!")
          hard_case = true
          tau = sqrt(delta2 - p_lambda2)
          Logging.debug("Tau = $tau delta2 = $delta2 p_lambda2 = $(p_lambda2)")

          # I don't think it matters which eigenvector we pick so take the first..
          for i=1:n
            s[i] = tau * H_eig[:vectors][i, 1]
            for k=(lambda_1_multiplicity + 1):n
              s[i] = s[i] +
                     qg[k] * H_eig[:vectors][i, k] / (H_eig[:values][k] + lambda)
            end
          end
        end
      end

      if !hard_case
        Logging.debug("Easy case")
        # The "easy case".
        # Algorithim 4.3 of N&W, with s insted of p_l to be consistent with
        # the rest of the library.

        root_finding_diff = Inf
        iter = 1
        B = copy(H)

        lambda_previous = copy(lambda)
        for i=1:n
          B[i, i] = H[i, i] + lambda
        end
        while (root_finding_diff > tolerance) && (iter <= max_iters)
          Logging.debug("---")
          Logging.debug("lambda=$lambda min_lambda=$(min_lambda)")
          b_eigv = eigfact(B)[:values]
          Logging.debug("lambda_1=$(lambda_1) $(b_eigv)")
          R = chol(B)
          s[:] = -R \ (R' \ gr)
          q_l = R' \ s
          norm2_s = vecdot(s, s)
          lambda_previous = lambda
          lambda = (lambda_previous +
                    norm2_s * (sqrt(norm2_s) - delta) / (delta * vecdot(q_l, q_l)))

          # Check that lambda is not less than min_lambda, and if so, go half the
          # distance to min_lambda.
          if lambda < min_lambda
            # TODO: add a unit test for this
            lambda = 0.5 * (lambda_previous - min_lambda) + min_lambda
            Logging.debug("Step too low.  Using $(lambda) from $(lambda_previous).")
          end
          root_finding_diff = abs(lambda - lambda_previous)
          iter = iter + 1
          for i=1:n
            B[i, i] = H[i, i] + lambda
          end
        end
        @assert(iter > 1, "Bad tolerance -- no iterations were computed")
        if iter > (max_iters + 1)
          warn(string("In the trust region subproblem max_iters ($max_iters) ",
                      "was exceeded.  Diff vs tolerance: ",
                      "$(root_finding_diff) > $(tolerance)"))
        end # end easy case root finding
      end # end easy case
    end # Getting s
    m = zero(T)
    if interior || hard_case
      m = vecdot(gr, s) + 0.5 * vecdot(s, H * s)
    else
      m = vecdot(gr, s) + 0.5 * vecdot(s, B * s)
    end

    # if !interior
    #     if abs(delta2 - vecdot(s, s)) > 1e-6
    #       warn("The norm of s is not close to delta: s2=$(vecdot(s, s)) delta2=$delta2. ",
    #            "This may occur when the Hessian is badly conditioned.  ",
    #            "max_ev=$(max_lambda), min_ev=$(lambda_1)")
    #     end
    # end
    Logging.debug("Root finding got m=$m, interior=$interior with ",
            "delta^2=$delta2 and ||s||^2=$(vecdot(s, s))")
    return m, interior, lambda
end


function newton_tr{T}(d::TwiceDifferentiableFunction,
                       initial_x::Vector{T};
                       initial_delta::T=1.0,
                       delta_hat::T = 100.0,
                       eta::T = 0.1,
                       xtol::Real = 1e-32,
                       ftol::Real = 1e-8,
                       grtol::Real = 1e-8,
                       rho_lower::Real = 0.25,
                       rho_upper::Real = 0.75,
                       iterations::Integer = 1_000,
                       store_trace::Bool = false,
                       show_trace::Bool = false,
                       extended_trace::Bool = false)

    @assert(delta_hat > 0, "delta_hat must be strictly positive")
    @assert(0 < initial_delta < delta_hat, "delta must be in (0, delta_hat)")
    @assert(0 <= eta < 0.25, "eta must be in [0, 0.25)")
    @assert(rho_lower < rho_upper, "must have rho_lower < rho_upper")
    @assert(rho_lower >= 0.)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 1

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in gr
    gr = Array(T, n)

    # The current search direction
    # TODO: Try to avoid re-allocating s
    s = Array(T, n)

    # Store f(x), the function value, in f_x
    f_x_previous, f_x = NaN, d.fg!(x, gr)

    # We need to store the previous gradient in case we reject a step.
    gr_previous = copy(gr)

    f_calls, g_calls = f_calls + 1, g_calls + 1

    # Store the hessian in H
    H = Array(T, n, n)
    d.h!(x, H)

    # Keep track of trust region sizes
    delta = copy(initial_delta)

    # Trace the history of states visited
    tr = OptimizationTrace()
    tracing = store_trace || show_trace || extended_trace
    @newton_tr_trace

    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iterate until convergence
    converged = false
    while !converged && iteration <= iterations
        Logging.debug("\n-----------------Iter $iteration")

        # Find the next step direction.
        m, interior = solve_tr_subproblem!(gr, H, delta, s)

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position
        for i in 1:n
            @inbounds x[i] = x[i] + s[i]
        end

        # Update the function value and gradient
        copy!(gr_previous, gr)
        f_x_previous, f_x = f_x, d.fg!(x, gr)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        # Update the trust region size based on the discrepancy between
        # the predicted and actual function values.  (Algorithm 4.1 in N&W)
        f_x_diff = f_x_previous - f_x
        if m == 0
          # This should only happen if the step is zero, in which case
          # we should accept the step and assess_convergence().
          @assert(f_x_diff == 0,
                  "m == 0 but the actual function change ($f_x_diff) is nonzero")
          rho = 1.0
        else
          rho = f_x_diff / (0 - m)
        end

        Logging.debug("Got rho = $rho from $(f_x) - $(f_x_previous) ",
                "(diff = $(f_x - f_x_previous)), and m = $m")
        Logging.debug("Interior = $interior, delta = $delta.")

        if rho < rho_lower
            Logging.debug("Shrinking trust region.")
            delta *= 0.25
        elseif (rho > rho_upper) && (!interior)
            Logging.debug("Growing trust region.")
            delta = min(2 * delta, delta_hat)
        else
          # else leave delta unchanged.
          Logging.debug("Keeping trust region the same.")
        end

        if rho > eta
            # Accept the point and check convergence
            Logging.debug("Accepting improvement from f_prev=$(f_x_previous) f=$(f_x).")

            x_converged,
            f_converged,
            gr_converged,
            converged = assess_convergence(x,
                                           x_previous,
                                           f_x,
                                           f_x_previous,
                                           gr,
                                           xtol,
                                           ftol,
                                           grtol)
            if !converged
              # Only compute the next Hessian if we haven't converged
              d.h!(x, H)
            else
              Logging.info("Converged.")
            end
        else
            # The improvement is too small and we won't take it.
            Logging.debug(
              "Rejecting improvement from f_prev = $(f_x_previous) to f=$f_x")

            # If you reject an interior solution, make sure that the next
            # delta is smaller than the current step.  Otherwise you waste
            # steps reducing delta by constant factors while each solution
            # will be the same.
            x_diff = x - x_previous
            delta = 0.25 * sqrt(vecdot(x_diff, x_diff))

            f_x = f_x_previous
            copy!(x, x_previous)
            copy!(gr, gr_previous)

        end

        # Increment the number of steps we've had to perform
        iteration += 1

        @newton_tr_trace
    end

    return MultivariateOptimizationResults("Newton's Method with Trust Region",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == iterations,
                                           x_converged,
                                           xtol,
                                           f_converged,
                                           ftol,
                                           gr_converged,
                                           grtol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
