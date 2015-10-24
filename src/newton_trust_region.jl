# # TODO: This is intended to be used in Optim.jl.  Until that is submitted
# # and merged, I'll include the working file in Celeste so the build passes.
# using Optim.update!
# using Optim.OptimizationTrace
# using Optim._dot
# using Optim.norm2
# using Optim.assess_convergence
# using Optim.MultivariateOptimizationResults
# using Optim.TwiceDifferentiableFunction

using Compat

function verbose_println(x...)
  #println(x...)
end

function verbose_println(x)
  #println(x)
end


macro newton_tr_trace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(trs.x)
                dt["g(x)"] = copy(trs.gr)
                dt["h(x)"] = copy(trs.H)
                dt["delta"] = copy(trs.delta)
            end
            grnorm = norm(trs.gr, Inf)
            update!(trs.tr,
                    trs.iteration,
                    trs.f_x,
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
      qg[i] = _dot(H_eig[:vectors][:, i], gr)
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
      verbose_println("Interior")
    else
      interior = false
      verbose_println("Boundary")

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
        verbose_println("lambda_1 = $(lambda_1), p_lambda2 = $(p_lambda2), ",
                "$delta2, $lambda_1_multiplicity")
        if p_lambda2 > delta2
          # Then we can simply solve using root finding.  Set a starting point
          # between the minimum and largest eigenvalues.
          # TODO: is there a better starting point?
          hard_case = false
          lambda = min_lambda + 0.01 * (max_lambda - min_lambda)
        else
          verbose_println("Hard case!")
          hard_case = true
          tau = sqrt(delta2 - p_lambda2)
          verbose_println("Tau = $tau delta2 = $delta2 p_lambda2 = $(p_lambda2)")

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
        verbose_println("Easy case")
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
          verbose_println("---")
          verbose_println("lambda=$lambda min_lambda=$(min_lambda)")
          b_eigv = eigfact(B)[:values]
          verbose_println("lambda_1=$(lambda_1) $(b_eigv)")
          R = chol(B)
          s[:] = -R \ (R' \ gr)
          q_l = R' \ s
          norm2_s = norm2(s)
          lambda_previous = lambda
          lambda = (lambda_previous +
                    norm2_s * (sqrt(norm2_s) - delta) / (delta * norm2(q_l)))

          # Check that lambda is not less than min_lambda, and if so, go half the
          # distance to min_lambda.
          if lambda < min_lambda
            # TODO: add a unit test for this
            lambda = 0.5 * (lambda_previous - min_lambda) + min_lambda
            verbose_println("Step too low.  Using $(lambda) from $(lambda_previous).")
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
      m = _dot(gr, s) + 0.5 * _dot(s, H * s)
    else
      m = _dot(gr, s) + 0.5 * _dot(s, B * s)
    end

    if !interior
        if abs(delta2 - norm2(s)) > 1e-6
          warn("The norm of s is not close to delta: s2=$(norm2(s)) delta2=$delta2. ",
               "This may occur when the Hessian is badly conditioned.  ",
               "max_ev=$(max_lambda), min_ev=$(lambda_1)")
        end
    end
    verbose_println("Root finding got m=$m, interior=$interior with ",
            "delta^2=$delta2 and ||s||^2=$(norm2(s))")
    return m, interior, lambda
end


# A type for storing the current state of the algorithm.
#
# Attributes:
#  - x: The current parameter vector
#  - x_previous: The last accepted parameter vector
#  - gr: The current gradient
#  - gr_previous: The last accepted gradient
#  - f_x: The current function value
#  - f_x_previous: The last accepted function value
#  - H: The current Hessian
#  - n: The length of the parameter vector
#  - s: The previous step direction
#  - delta: The current size of the trust region
#  - iteration: The current iteration number
#  - f_calls: The number of function calls
#  - g_calls: The number of gradient calls
type NewtonTRState{T}
  x::Vector{T}
  x_previous::Vector{T}
  gr::Vector{T}
  gr_previous::Vector{T}
  f_x::Real
  f_x_previous::Real
  H::Matrix{T} # The Hesssian.
  n::Integer # The size of the pareameter vector.
  s::Vector{T} # Memory allocated for the current search direction.
  delta::Real # The current trust region size.
  iteration::Integer
  f_calls::Integer
  g_calls::Integer
  tr::OptimizationTrace
end


# Initialize a NewtonTRState object.
NewtonTRState{T}(d::TwiceDifferentiableFunction,
                 initial_x::Vector{T};
                 initial_delta::T=1.0) = begin
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

  NewtonTRState(x, x_previous, gr, gr_previous, f_x, f_x_previous,
                H, n, s, delta, iteration, f_calls, g_calls, tr)
end


function take_newton_tr_step!{T}(d::TwiceDifferentiableFunction,
                                 trs::NewtonTRState{T};
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

  verbose_println("\n-----------------Iter $(trs.iteration)")

  x_converged = f_converged = gr_converged = converged = false

  # Find the next step direction.
  m, interior = solve_tr_subproblem!(trs.gr, trs.H, trs.delta, trs.s)

  # Maintain a record of previous position
  copy!(trs.x_previous, trs.x)

  # Update current position
  for i in 1:trs.n
     @inbounds trs.x[i] = trs.x[i] + trs.s[i]
  end

  # Update the function value and gradient
  copy!(trs.gr_previous, trs.gr)
  trs.f_x_previous, trs.f_x = trs.f_x, d.fg!(trs.x, trs.gr)
  trs.f_calls, trs.g_calls = trs.f_calls + 1, trs.g_calls + 1

  # Update the trust region size based on the discrepancy between
  # the predicted and actual function values.  (Algorithm 4.1 in N&W)
  f_x_diff = trs.f_x_previous - trs.f_x
  if m == 0
   # This should only happen if the step is zero, in which case
   # we should accept the step and assess_convergence().
   @assert(f_x_diff == 0,
           "m == 0 but the actual function change ($f_x_diff) is nonzero")
   rho = 1.0
  else
   rho = f_x_diff / (0 - m)
  end

  verbose_println("Got rho = $rho from $(trs.f_x) - $(trs.f_x_previous) ",
         "(diff = $(trs.f_x - trs.f_x_previous)), and m = $m")
  verbose_println("Interior = $interior, delta = $(trs.delta).")

  if rho < rho_lower
     verbose_println("Shrinking trust region.")
     trs.delta *= 0.25
  elseif (rho > rho_upper) && (!interior)
     verbose_println("Growing trust region.")
     trs.delta = min(2 * trs.delta, delta_hat)
  else
   # else leave delta unchanged.
   verbose_println("Keeping trust region the same.")
  end

  if rho > eta
     # Accept the point and check convergence
     verbose_println("Accepting improvement from ",
                     "f_prev=$(trs.f_x_previous) f=$(trs.f_x).")

     x_converged,
     f_converged,
     gr_converged,
     converged = assess_convergence(trs.x,
                                    trs.x_previous,
                                    trs.f_x,
                                    trs.f_x_previous,
                                    trs.gr,
                                    xtol,
                                    ftol,
                                    grtol)
     if !converged
       # Only compute the next Hessian if we haven't converged
       d.h!(trs.x, trs.H)
     else
       verbose_println("Converged.")
     end
  else
     # The improvement is too small and we won't take it.
     verbose_println("Rejecting improvement from $(trs.x_previous) to ",
             "$(trs.x), f=$(trs.f_x) (f_prev = $(trs.f_x_previous))")

     # If you reject an interior solution, make sure that the next
     # delta is smaller than the current step.  Otherwise you waste
     # steps reducing delta by constant factors while each solution
     # will be the same.
     trs.delta = 0.25 * sqrt(norm2(trs.x - trs.x_previous))

     trs.f_x = trs.f_x_previous
     copy!(trs.x, trs.x_previous)
     copy!(trs.gr, trs.gr_previous)

  end

  # Increment the number of steps we've had to perform
  trs.iteration += 1

  # Record the step
  tracing = store_trace || show_trace || extended_trace
  @newton_tr_trace

  x_converged, f_converged, gr_converged, converged
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

    trs = NewtonTRState(d, initial_x, initial_delta=initial_delta)

    tracing = store_trace || show_trace || extended_trace
    @newton_tr_trace

    # Iterate until convergence
    converged = x_converged = f_converged = gr_converged = false
    while !converged && trs.iteration <= iterations
      x_converged, f_converged, gr_converged, converged =
        take_newton_tr_step!(d,
                             trs,
                             delta_hat=delta_hat,
                             eta=eta,
                             xtol=xtol,
                             ftol=ftol,
                             grtol=grtol,
                             rho_lower=rho_lower,
                             rho_upper=rho_upper,
                             iterations=iterations,
                             store_trace=store_trace,
                             show_trace=show_trace,
                             extended_trace=extended_trace)
    end

    return MultivariateOptimizationResults("Newton's Method with Trust Region",
                                           initial_x,
                                           trs.x,
                                           @compat(Float64(trs.f_x)),
                                           trs.iteration,
                                           trs.iteration == iterations,
                                           x_converged,
                                           xtol,
                                           f_converged,
                                           ftol,
                                           gr_converged,
                                           grtol,
                                           trs.tr,
                                           trs.f_calls,
                                           trs.g_calls)
end
