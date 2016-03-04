module OptimizeElbo


using CelesteTypes
using Transform

import ElboDeriv
import DataFrames
import Optim

export ObjectiveWrapperFunctions, WrapperState

#TODO: use Lumberjack.jl for logging
const debug = false

# Only include until this is merged with Optim.jl.
include(joinpath(Pkg.dir("Celeste"), "src", "newton_trust_region.jl"))

# The main reason we need this is to have a mutable type to keep
# track of function evaluations, but we can keep other metadata
# in it as well.
type WrapperState
    f_evals::Int
    verbose::Bool
    print_every_n::Int
    scale::Float64
end


type ObjectiveWrapperFunctions

    f_objective::Function
    f_value_grad::Function
    f_value_grad!::Function
    f_value::Function
    f_grad::Function
    f_grad!::Function
    f_hessian::Function
    f_hessian!::Function

    state::WrapperState
    transform::DataTransform
    mp::ModelParams{Float64}
    kept_ids::Array{Int}
    omitted_ids::Array{Int}

    # Caching
    last_sf::SensitiveFloat
    last_x::Vector{Float64}

    # Arguments:
    #  f: A function that takes in ModelParams and returns a SensitiveFloat.
    #  mp: Initial ModelParams
    #  transform: A DataTransform that matches ModelParams
    #  kept_ids: The free parameter ids to keep
    #  omitted_ids: The free parameter ids to omit (TODO: this is redundant)
    #  fast_hessian: Evaluate the forward autodiff Hessian using only
    #                mp.active_sources to speed up computation.
    ObjectiveWrapperFunctions(
      f::Function, mp::ModelParams{Float64}, transform::DataTransform,
      kept_ids::Array{Int, 1}, omitted_ids::Array{Int, 1};
      fast_hessian::Bool=true) = begin

        x_length = length(kept_ids) * transform.active_S
        x_size = (length(kept_ids), transform.active_S)
        @assert transform.active_sources == mp.active_sources

        last_sf =
          zero_sensitive_float(UnconstrainedParams, length(mp.active_sources))
        last_x = [ NaN ]

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
              for s=1:S
                state_df[symbol(string("grad", s))] = grad[:, s]
              end
              println(state_df)
              println("\n=======================================\n")
            end
        end

        function f_objective(x::Vector{Float64})
          if x == last_x
            # Return the cached result.
            return(last_sf)
          else
            state.f_evals += 1
            # Evaluate in the constrained space and then unconstrain again.
            transform.array_to_vp!(reshape(x, x_size), mp.vp, omitted_ids)
            f_res = f(mp)

            # TODO: Add an option to print either the transformed or
            # free parameterizations.
            print_status(mp.vp[mp.active_sources],
                         f_res.v[1], f_res.d)
            f_res_trans = transform.transform_sensitive_float(f_res, mp)

            # Cache the result.
            last_x = deepcopy(x)
            last_sf = deepcopy(f_res_trans)
            return(f_res_trans)
          end
        end

        function f_value_grad{T <: Number}(x::Vector{T})
            @assert length(x) == x_length
            res = f_objective(x)
            grad = zeros(T, length(x))
            if length(grad) > 0
                svs = [res.d[kept_ids, si] for si in 1:transform.active_S]
                grad[:] = reduce(vcat, svs)
            end
            state.scale * res.v[1], state.scale .* grad
        end

        # The remaining functions are scaled and take matrices.
        function f_value_grad!{T <: Number}(x::Vector{T}, grad::Vector{T})
            @assert length(x) == x_length
            @assert length(x) == length(grad)
            value, grad[:,:] = f_value_grad(x)
            value
        end

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

        function f_hessian{T <: Number}(x::Vector{T})
          @assert length(x) == x_length
          res = f_objective(x)
          all_kept_ids = Int[]
          for sa=1:transform.active_S
            append!(all_kept_ids, kept_ids + (sa - 1) * length(kept_ids))
          end
          sub_hess = res.h[all_kept_ids, all_kept_ids]
          state.scale .* 0.5 * (sub_hess + sub_hess')
      end

        function f_hessian!{T <: Number}(x::Vector{T}, hess::Matrix{T})
          hess[:, :] = f_hessian(x)
        end

        new(f_objective, f_value_grad, f_value_grad!,
            f_value, f_grad, f_grad!, f_hessian, f_hessian!,
            # f_ad_hessian!, f_ad_hessian_sparse,
            state, transform, mp, kept_ids, omitted_ids, last_sf)
    end
end


"""
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

Returns:
  - iter_count: The number of iterations taken
  - max_f: The maximum function value achieved
  - max_x: The optimal function input
  - ret: The return code of optimize()
"""
function maximize_f(
  f::Function, tiled_blob::TiledBlob, mp::ModelParams,
  transform::Transform.DataTransform;
  omitted_ids=Int[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose=false,
  max_iters=100, rho_lower=0.25, fast_hessian=true)

    # Make sure the model parameters are within the transform bounds
    enforce_bounds!(mp, transform)

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    optim_obj_wrap =
      OptimizeElbo.ObjectiveWrapperFunctions(
        mp -> f(tiled_blob, mp), mp, transform, kept_ids, omitted_ids,
        fast_hessian=fast_hessian);

    # For minimization, which is required by the linesearch algorithm.
    optim_obj_wrap.state.scale = -1.0
    optim_obj_wrap.state.verbose = verbose

    x0 = transform.vp_to_array(mp.vp, omitted_ids);
    d = Optim.TwiceDifferentiableFunction(
      optim_obj_wrap.f_value, optim_obj_wrap.f_grad!,
      optim_obj_wrap.f_hessian!)

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


function maximize_f(f::Function, tiled_blob::TiledBlob, mp::ModelParams;
    omitted_ids=Int[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose = false,
    max_iters = 100)
    # Use the default transform.

    transform = get_mp_transform(mp);
    maximize_f(f, tiled_blob, mp, transform,
      omitted_ids=omitted_ids, xtol_rel=xtol_rel, ftol_abs=ftol_abs,
      verbose=verbose, max_iters=max_iters)
end

function maximize_elbo(tiled_blob::TiledBlob, mp::ModelParams, trans::DataTransform;
    xtol_rel = 1e-7, ftol_abs=1e-6, verbose = false)
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
    omitted_ids = [ids_free.k[:]; ids_free.c2[:]; ids_free.r2]
    maximize_f(ElboDeriv.elbo_likelihood, tiled_blob, mp, trans,
               omitted_ids=omitted_ids, xtol_rel=xtol_rel,
               ftol_abs=ftol_abs, verbose=verbose)
end

end
