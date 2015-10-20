# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

VERSION < v"0.4.0-dev" && using Docile

using NLopt
using CelesteTypes
using Transform

import ElboDeriv
import DataFrames
import ForwardDiff
import DualNumbers
import Optim

using DualNumbers.Dual
using DualNumbers.epsilon
using DualNumbers.real

export ObjectiveWrapperFunctions, WrapperState

#TODO: use Lumberjack.jl for logging
const debug = false

# Only include until this is merged with Optim.jl.
include(joinpath(Pkg.dir("Celeste"), "src", "newton_trust_region.jl"))

# The main reason we need this is to have a mutable type to keep
# track of function evaluations, but we can keep other metadata
# in it as well.
type WrapperState
    f_evals::Int64
    verbose::Bool
    print_every_n::Int64
    scale::Float64
end


type ObjectiveWrapperFunctions

    f_objective::Function
    f_value_grad::Function
    f_value_grad!::Function
    f_value::Function
    f_grad::Function
    f_grad!::Function
    f_ad_grad::Function
    f_ad_hessian::Function

    state::WrapperState
    transform::DataTransform
    mp::ModelParams{Float64}
    kept_ids::Array{Int64}
    omitted_ids::Array{Int64}

    ObjectiveWrapperFunctions(
      f::Function, mp::ModelParams{Float64}, transform::DataTransform,
      kept_ids::Array{Int64, 1}, omitted_ids::Array{Int64, 1}) = begin

        mp_dual = CelesteTypes.convert(ModelParams{DualNumbers.Dual}, mp);
        x_length = length(kept_ids) * mp.S
        x_size = (length(kept_ids), mp.S)

        state = WrapperState(0, false, 10, 1.0)
        function print_status{T <: Number}(
          iter_vp::VariationalParams{T}, value::T, grad::Array{T})
            if state.verbose || (state.f_evals % state.print_every_n == 0)
                println("f_evals: $(state.f_evals) value: $(value)")
            end
            if state.verbose
              S = length(iter_vp)
              if length(iter_vp[1]) == length(ids_names)
                state_df = DataFrames.DataFrame(names=ids_names)
              elseif length(iter_vp[1]) == length(ids_free_names)
                state_df = DataFrames.DataFrame(names=ids_free_names)
              else
                state_df = DataFrames.DataFrame(
                  names=[ "x$i" for i=1:length(iter_vp[1, :])])
              end
              for s=1:S
                state_df[symbol(string("val", s))] = iter_vp[s]
              end
              for s=1:mp.S
                state_df[symbol(string("grad", s))] = grad[:, s]
              end
              println(state_df)
              println("\n=======================================\n")
            end
        end

        # Takes a vector
        function f_objective(x_dual::Vector{DualNumbers.Dual{Float64}})
            state.f_evals += 1
            # Evaluate in the constrained space and then unconstrain again.
            transform.array_to_vp!(reshape(x_dual, x_size), mp_dual.vp, omitted_ids)
            f_res = f(mp_dual)
            f_res_trans = transform.transform_sensitive_float(f_res, mp_dual)
        end

        function f_objective(x::Vector{Float64})
            state.f_evals += 1
            # Evaluate in the constrained space and then unconstrain again.
            transform.array_to_vp!(reshape(x, x_size), mp.vp, omitted_ids)
            f_res = f(mp)

            # TODO: Add an option to print either the transformed or
            # free parameterizations.
            print_status(mp.vp, f_res.v, f_res.d)
            f_res_trans = transform.transform_sensitive_float(f_res, mp)
            #print_status(transform.from_vp(mp.vp), f_res_trans.v, f_res_trans.d)
            f_res_trans
        end

        function f_value_grad{T <: Number}(x::Vector{T})
            @assert length(x) == x_length
            res = f_objective(x)
            grad = zeros(T, length(x))
            if length(grad) > 0
                svs = [res.d[kept_ids, s] for s in 1:mp.S]
                grad[:] = reduce(vcat, svs)
            end
            state.scale * res.v, state.scale .* grad
        end

        # The remaining functions are scaled and take matrices.
        function f_value_grad!{T <: Number}(x::Vector{T}, grad::Vector{T})
            @assert length(x) == x_length
            @assert length(x) == length(grad)
            value, grad[:,:] = f_value_grad(x)
            value
        end

        # TODO: Add caching.
        function f_value{T <: Number}(x::Vector{T})
            @assert length(x) == x_length
            f_value_grad(x)[1]
       end

        function f_grad{T <: Number}(x::Vector{T})
            @assert length(x) == x_length
            f_value_grad(x)[2]
        end

        function f_grad!{T <: Number}(x::Vector{T}, grad::Vector{T})
            grad[:,:] = f_grad(x)
        end

        # Forward diff versions of the gradient and Hessian.
        f_ad_grad = ForwardDiff.forwarddiff_gradient(
          f_value, Float64, fadtype=:dual; n=x_length);

        function f_ad_hessian(x_vec::Array{Float64})
            @assert length(x_vec) == x_length
            x = reshape(x_vec, x_size)
            k = length(x)
            hess = zeros(Float64, k, k);
            x_dual = DualNumbers.Dual{Float64}[
              DualNumbers.Dual{Float64}(x[i, j], 0.) for
              i = 1:size(x)[1], j=1:size(x)[2]];
            print("Getting Hessian ($k components): ")
            for s in 1:size(x)[2], index in 1:size(x)[1]
                index == 1 ? print("o"): print(".")
                original_x = x[index, s]
                x_dual[index, s] = DualNumbers.Dual(original_x, 1.)
                deriv = f_grad(x_dual[:])
                # This goes through deriv in column-major order.
                hess[:, index] =
                  Float64[ ForwardDiff.epsilon(x_val) for x_val in deriv ]
                x_dual[index, s] = DualNumbers.Dual(original_x, 0.)
            end
            print("Done.\n")
            # Assure that the hessian is exactly symmetric.
            0.5 * (hess + hess')
        end

        new(f_objective, f_value_grad, f_value_grad!, f_value, f_grad, f_grad!,
            f_ad_grad, f_ad_hessian,
            state, transform, mp, kept_ids, omitted_ids)
    end
end


@doc """
Vectors of the row, column, and value of the Hessian entries.
The indices are tuples of (source, parameter) which will be
linearized later.
""" ->
function elbo_hessian(tiled_blob::TiledBlob,
                      x::Matrix{Float64},
                      mp_dual::ModelParams{DualNumbers.Dual{Float64}},
                      transform::Transform.DataTransform,
                      omitted_ids::Vector{Int64};
                      deriv_sources=1:mp_dual.S,
                      verbose=false)
  k = size(x)[1]
  @assert size(x)[2] == length(mp_dual.vp)

  hess_i = Tuple{Int64, Int64}[]
  hess_j = Tuple{Int64, Int64}[]
  hess_val = Float64[]

  accum = zero_sensitive_float(CanonicalParams, Dual{Float64}, mp_dual.S);
  x_dual = Dual{Float64}[ Dual{Float64}(x[i, j], 0.) for
                          i = 1:size(x)[1], j=1:size(x)[2] ];

  new_hess_time = time()
  for s1 in deriv_sources
    verbose && println("Source $s1")
    for index1=1:k
      verbose && print(".")
      original_val = real(x_dual[index1, s1])
      @assert epsilon(x_dual[index1, s1]) == 0.
      x_dual[index1, s1] = DualNumbers.Dual(original_val, 1.);
      transform.array_to_vp!(x_dual, mp_dual.vp, omitted_ids);
      ElboDeriv.elbo_hessian_term!(tiled_blob, mp_dual, accum, s1);
      accum_trans = transform.transform_sensitive_float(accum, mp_dual);
      @assert size(accum_trans.d) == (k, mp_dual.S)
      x_dual[index1, s1] = DualNumbers.Dual(original_val, 0.)

      # Record the hessian terms.
      for s2 in deriv_sources, index2=1:k
        this_hess_val = DualNumbers.epsilon(accum_trans.d[index2, s2])
        if (this_hess_val != 0)
          push!(hess_i, (s1, index1))
          push!(hess_j, (s2, index2))
          push!(hess_val, this_hess_val)
        end
      end
    end
    verbose && println("Done with source $s1.")
  end
  new_hess_time = time() - new_hess_time

  @assert length(hess_i) == length(hess_j) == length(hess_val)
  hess_i, hess_j, hess_val, new_hess_time
end


function unpack_hessian_vals(hess_i::@compat(Vector{Tuple{Int64, Int64}}),
                             hess_j::Vector{Tuple{Int64, Int64}},
                             hess_val::Vector{Float64},
                             dims::@compat(Tuple{Int64, Int64}))
  # TODO: make this function part of the transform.
  hess_i_vec = Array(Int64, length(hess_i));
  hess_j_vec = Array(Int64, length(hess_j));
  for entry in 1:length(hess_i)
    hess_i_vec[entry] = sub2ind(dims, hess_i[entry][2], hess_i[entry][1])
    hess_j_vec[entry] = sub2ind(dims, hess_j[entry][2], hess_j[entry][1])
  end
  new_hess_sparse =
    sparse(hess_i_vec, hess_j_vec, hess_val, prod(dims), prod(dims));
  new_hess = 0.5 * full(new_hess_sparse + new_hess_sparse')
end



function get_nlopt_unconstrained_bounds(vp::Vector{Vector{Float64}},
                                        omitted_ids::Vector{Int64},
                                        transform::DataTransform)
    # Set reasonable bounds for unconstrained parameters.
    #
    # vp: Variational parameters.
    # omitted_ids: Ids of _unconstrained_ variational parameters to be omitted.
    # transform: A DataTransform object.

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    lbs = fill(-15.0, length(ids_free), length(vp))
    ubs = fill(15.0, length(ids_free), length(vp))

    # Change the bounds to match the scaling
    for s=1:length(vp)
      for (param, bounds) in transform.bounds[s]
        if (bounds.ub == Inf)
          # Hack: higher bounds for upper-unconstrained params.
          ubs[collect(ids_free.(param)), s] = 20.0
        end
        lbs[collect(ids_free.(param)), s] .*= bounds.rescaling
        ubs[collect(ids_free.(param)), s] .*= bounds.rescaling
      end
    end
    reduce(vcat, lbs[kept_ids, :]), reduce(vcat, ubs[kept_ids, :])
end


@doc """
Optimizes f using Newton's method and exact Hessians.  For now, it is
not clear whether this or BFGS is better, so it is kept as a separate function.

Args:
  - f: A function that takes a tiled_blob and constrained coordinates
       (e.g. ElboDeriv.elbo)
  - tiled_blob: Input for f
  - mp: Constrained initial ModelParams
  - transform: The data transform to be applied before optimizing.
  - lbs: An array of lower bounds (in the transformed space)
  - ubs: An array of upper bounds (in the transformed space)
  - omitted_ids: Omitted ids from the _unconstrained_ parameterization
      (i.e. elements of free_ids).
  - xtol_rel: X convergence
  - ftol_abs: F convergence
  - verbose: Print detailed output
  - elbo_hessian: Use a more efficient autodiff function for the ELBO hessian.

Returns:
  - iter_count: The number of iterations taken
  - max_f: The maximum function value achieved
  - max_x: The optimal function input
  - ret: The return code of optimize()
""" ->
function maximize_f_newton(
  f::Function, tiled_blob::TiledBlob, mp::ModelParams,
  transform::Transform.DataTransform;
  omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose=false,
  max_iters=100, rho_lower=0.25, elbo_hessian=false)

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    optim_obj_wrap =
      OptimizeElbo.ObjectiveWrapperFunctions(
        mp -> f(tiled_blob, mp), mp, transform, kept_ids, omitted_ids);

    # For minimization, which is required by the linesearch algorithm.
    optim_obj_wrap.state.scale = -1.0
    optim_obj_wrap.state.verbose = verbose

    function f_hess_wrapper!(x, new_hess)
      hess = optim_obj_wrap.f_ad_hessian(x)
      new_hess[:,:] = hess
    end

    x0 = transform.vp_to_array(mp.vp, omitted_ids);
    d = Optim.TwiceDifferentiableFunction(
      optim_obj_wrap.f_value, optim_obj_wrap.f_grad!, f_hess_wrapper!)

    # TODO: use the Optim version after newton_tr is merged.
    nm_result = newton_tr(d,
                          x0[:],
                          xtol = xtol_rel,
                          ftol = ftol_abs,
                          grtol = 1e-8,
                          iterations = max_iters,
                          store_trace = verbose,
                          show_trace = false,
                          extended_trace = verbose,
                          initial_delta=10.0,
                          delta_hat=1e9,
                          rho_lower = rho_lower)

    iter_count = optim_obj_wrap.state.f_evals
    transform.array_to_vp!(reshape(nm_result.minimum, size(x0)),
                           mp.vp, omitted_ids);
    max_f = -1.0 * nm_result.f_minimum
    max_x = nm_result.minimum

    println("got $max_f at $max_x after $iter_count function evaluations ",
            "($(nm_result.iterations) Newton steps)\n")
    iter_count, max_f, max_x, nm_result
end

@doc """
Maximize using BFGS and unconstrained coordinates.

Args:
  - f: A function that takes a tiled_blob and constrained coordinates
       (e.g. ElboDeriv.elbo)
  - tiled_blob: Input for f
  - mp: Constrained initial ModelParams
  - transform: The data transform to be applied before optimizing.
  - lbs: An array of lower bounds (in the transformed space)
  - ubs: An array of upper bounds (in the transformed space)
  - omitted_ids: Omitted ids from the _unconstrained_ parameterization
       (i.e. elements of free_ids).
  - xtol_rel: X convergence
  - ftol_abs: F convergence
  - verbose: Print detailed output

Returns:
  - iter_count: The number of iterations taken
  - max_f: The maximum function value achieved
  - max_x: The optimal function input
  - ret: The return code of optimize()
""" ->
function maximize_f(
  f::Function, tiled_blob::TiledBlob, mp::ModelParams,
  transform::DataTransform,
  lbs::@compat(Union{Float64, Vector{Float64}}),
  ubs::@compat(Union{Float64, Vector{Float64}});
  omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose = false)

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    x0 = transform.vp_to_array(mp.vp, omitted_ids)[:]

    obj_wrapper = ObjectiveWrapperFunctions(
      mp -> f(tiled_blob, mp), mp, transform, kept_ids, omitted_ids);
    obj_wrapper.state.verbose = verbose

    opt = Opt(:LD_LBFGS, length(x0))
    for i in 1:length(x0)
        if !(lbs[i] <= x0[i] <= ubs[i])
            println("coordinate $i falsity: $(lbs[i]) <= $(x0[i]) <= $(ubs[i])")
            if x0[i] >= ubs[i]
              x0[i] = ubs[i] - 1e-6
            elseif x0[i] <= lbs[i]
              x0[i] = lbs[i] + 1e-6
            end
        end
    end
    lower_bounds!(opt, lbs)
    upper_bounds!(opt, ubs)
    max_objective!(opt, obj_wrapper.f_value_grad!)
    xtol_rel!(opt, xtol_rel)
    ftol_abs!(opt, ftol_abs)
    (max_f, max_x, ret) = optimize(opt, x0)

    obj_wrapper.state.f_evals, max_f, max_x, ret
end


function maximize_f(
  f::Function, tiled_blob::TiledBlob, mp::ModelParams, transform::DataTransform;
    omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose = false)
    # Default to the bounds given in get_nlopt_unconstrained_bounds.

    lbs, ubs = get_nlopt_unconstrained_bounds(mp.vp, omitted_ids, transform)
    maximize_f(f, tiled_blob, mp, transform, lbs, ubs;
      omitted_ids=omitted_ids, xtol_rel=xtol_rel,
      ftol_abs=ftol_abs, verbose = verbose)
end

function maximize_f(f::Function, tiled_blob::TiledBlob, mp::ModelParams;
    omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose = false)
    # Use the default transform.

    transform = get_mp_transform(mp);
    maximize_f(f, tiled_blob, mp, transform,
      omitted_ids=omitted_ids, xtol_rel=xtol_rel, ftol_abs=ftol_abs,
      verbose=verbose)
end

function maximize_elbo(tiled_blob::TiledBlob, mp::ModelParams, trans::DataTransform;
    xtol_rel = 1e-7, ftol_abs=1e-6, verbose = false)
    omitted_ids = setdiff(1:length(UnconstrainedParams),
                          [ids_free.r1, ids_free.r2,
                           ids_free.k[:], ids_free.c1[:]])
    maximize_f(ElboDeriv.elbo, tiled_blob, mp, trans, omitted_ids=omitted_ids,
        ftol_abs=ftol_abs, xtol_rel=xtol_rel, verbose=verbose)

    maximize_f(ElboDeriv.elbo, tiled_blob, mp, trans,
        ftol_abs=ftol_abs, xtol_rel=xtol_rel, verbose=verbose)
end

function maximize_elbo(tiled_blob::TiledBlob, mp::ModelParams; verbose = false)
    trans = get_mp_transform(mp)
    maximize_elbo(tiled_blob, mp, trans, verbose=verbose)
end

function maximize_likelihood(
  tiled_blob::TiledBlob, mp::ModelParams, trans::DataTransform;
  xtol_rel = 1e-7, ftol_abs=1e-6, verbose = false)
    omitted_ids = [ids_free.k[:], ids_free.c2[:], ids_free.r2]
    maximize_f(ElboDeriv.elbo_likelihood, tiled_blob, mp, trans,
               omitted_ids=omitted_ids, xtol_rel=xtol_rel,
               ftol_abs=ftol_abs, verbose=verbose)
end

end
