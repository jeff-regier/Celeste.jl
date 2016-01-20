module OptimizeElbo


using CelesteTypes
using Transform

import ElboDeriv
import DataFrames
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
    f_hessian::Function
    f_hessian!::Function
    # f_ad_grad::Function
    f_ad_hessian!::Function
    f_ad_hessian_sparse::Function

    state::WrapperState
    transform::DataTransform
    mp::ModelParams{Float64}
    kept_ids::Array{Int64}
    omitted_ids::Array{Int64}
    DualType::DataType

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
      kept_ids::Array{Int64, 1}, omitted_ids::Array{Int64, 1};
      fast_hessian::Bool=true) = begin

        x_length = length(kept_ids) * transform.active_S
        x_size = (length(kept_ids), transform.active_S)
        DualType = DualNumbers.Dual{Float64}
        mp_dual = CelesteTypes.convert(ModelParams{DualType}, mp);
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

        function f_objective(x_dual::Vector{DualType})
            state.f_evals += 1
            # Evaluate in the constrained space and then unconstrain again.
            transform.array_to_vp!(reshape(x_dual, x_size), mp_dual.vp, omitted_ids)
            f_res = f(mp_dual)
            f_res_trans = transform.transform_sensitive_float(f_res, mp_dual)
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
                         f_res.v, f_res.d)
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
            state.scale * res.v, state.scale .* grad
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
          all_kept_ids = Int64[]
          for sa=1:transform.active_S
            append!(all_kept_ids, kept_ids + (sa - 1) * length(kept_ids))
          end
          sub_hess = res.h[all_kept_ids, all_kept_ids]
          state.scale .* 0.5 * (sub_hess + sub_hess')
      end

        function f_hessian!{T <: Number}(x::Vector{T}, hess::Matrix{T})
          hess[:, :] = f_hessian(x)
        end

        # # Compute a forward AD gradient.  This is mostly useful
        # # for debugging and testing.
        # function f_ad_grad(x_vec::Array{Float64})
        #   # TODO: combine this and the Hessian into a single function.
        #   @assert length(x_vec) == x_length
        #   x = reshape(x_vec, x_size)
        #   k = length(x_vec)
        #
        #   x_dual = DualType[
        #     DualType(x[i, j], 0.) for i = 1:(x_size[1]), j=1:(x_size[2])];
        #
        #   grad = zeros(Float64, x_size...)
        #   print("Getting autodiff gradient ($k components): ")
        #   mp_dual.active_sources = mp.active_sources
        #   for si in 1:length(mp.active_sources)
        #     s = mp.active_sources[si]
        #     for index in 1:x_size[1]
        #       index == 1 ? print("+"): print("-")
        #       original_x = x[index, si]
        #       x_dual[index, si] = DualType(original_x, 1.)
        #
        #       value = f_value(x_dual[:])
        #       # This goes through deriv in column-major order.
        #       grad[index, si] = Float64(DualNumbers.epsilon(value))
        #       x_dual[index, si] = DualType(original_x, 0.)
        #     end
        #   end
        #   print("Done.\n")
        #   grad
        # end

        # Update <hess> in place with an autodiff hessian.
        function f_ad_hessian!(x_vec::Array{Float64}, hess::Matrix{Float64})
            @assert length(x_vec) == x_length
            x = reshape(x_vec, x_size)
            k = length(x_vec)

            x_dual = DualType[
              DualType(x[i, j], 0.) for i = 1:(x_size[1]), j=1:(x_size[2])];

            @assert size(hess) == (k, k)

            print("Getting Hessian ($k components): ")
            mp_dual.active_sources = mp.active_sources
            for si in 1:length(mp.active_sources)
              s = mp.active_sources[si]
              if fast_hessian
                # We only need to calculate the derivatives in tiles where
                # epsilon != 0.  The values of the derivatives themselves (that
                # is, the real part of the dual numbers) will be wrong but the
                # second derivatives with respect to source s will be right.
                mp_dual.active_sources = [s]
              end
              for index in 1:x_size[1]
                index == 1 ? print("o"): print(".")
                original_x = x[index, si]
                x_dual[index, si] = DualType(original_x, 1.)

                deriv = f_grad(x_dual[:])
                # This goes through deriv in column-major order.
                hess[:, sub2ind(x_size, index, si)] =
                  Float64[ DualNumbers.epsilon(x_val) for x_val in deriv ]
                x_dual[index, si] = DualType(original_x, 0.)
              end
            end
            print("Done.\n")
            # Assure that the hessian is exactly symmetric.
            hess[:,:] = 0.5 * (hess + hess')
        end

        # Returns the row and column indices and value of a sparse Hessian
        # computed with autodifferentiation.
        function f_ad_hessian_sparse(x_vec::Array{Float64})
            @assert length(x_vec) == x_length
            x = reshape(x_vec, x_size)
            k = length(x_vec)

            x_dual = DualType[DualType(x[i, j], 0.) for
                              i = 1:x_size[1], j=1:x_size[2]];

            # Vectors of the (source, component) indices for the rows
            # and columns of the Hessian.
            hess_i = Tuple{Int64, Int64}[]
            hess_j = Tuple{Int64, Int64}[]

            # Values of the hessian in the (hess_i, hess_j) locations.
            hess_val = Float64[]

            print("Getting sparse Hessian ($k components): ")
            mp_dual.active_sources = mp.active_sources
            for s1i in 1:length(mp.active_sources)
              s1 = mp.active_sources[s1i]
              if fast_hessian
                # We only need to calculate the derivatives in tiles where
                # epsilon != 0.  The values of the derivatives themselves (that
                # is, the real part of the dual numbers) will be wrong but the
                # second derivatives with respect to source s will be right.
                mp_dual.active_sources = [s1]
              end
              for index1 in 1:x_size[1]
                index1 == 1 ? print("o"): print(".")
                original_x = x[index1, s1i]
                x_dual[index1, s1i] = DualType(original_x, 1.)
                deriv = reshape(f_grad(x_dual[:]), x_size)
                x_dual[index1, s1i] = DualType(original_x, 0.)

                # Record the hessian terms.
                for s2i in 1:length(mp.active_sources), index2=1:size(x)[1]
                  s2 = mp.active_sources[s2i]
                  this_hess_val = DualNumbers.epsilon(deriv[index2, s2i])
                  if (this_hess_val != 0)
                    push!(hess_i, (s1i, index1))
                    push!(hess_j, (s2i, index2))
                    push!(hess_val, this_hess_val)
                  end # index2 for
                end # s2 for
              end # index for
            end # s1 for
            print("Done.\n")
            hess_i, hess_j, hess_val
        end

        new(f_objective, f_value_grad, f_value_grad!,
            f_value, f_grad, f_grad!, f_hessian, f_hessian!,
            f_ad_hessian!, f_ad_hessian_sparse,
            state, transform, mp, kept_ids, omitted_ids, DualType, last_sf)
    end
end


@doc """
Convert the indices and values of a sparse Hessian matrix to an actual
sparse matrix.

Args:
  - hess_i: A vector of (source, component) tuples for the Hessian rows
  - hess_j: A vector of (source, component) tuples for the Hessian columns
  - hess_val: The values of the Hessian corresponding to (hess_i, hess_j)
  - dims: The dimensions of the parameter matrix (#components, #sources)

Returns:
  - A symmetric sparse matrix corresponding to the inputs.
""" ->
function unpack_hessian_vals(hess_i::Vector{Tuple{Int64, Int64}},
                             hess_j::Vector{Tuple{Int64, Int64}},
                             hess_val::Vector{Float64},
                             dims::Tuple{Int64, Int64})
  # TODO: make this function part of the transform.
  hess_i_vec = Array(Int64, length(hess_i));
  hess_j_vec = Array(Int64, length(hess_j));
  for entry in 1:length(hess_i)
    hess_i_vec[entry] = sub2ind(dims, hess_i[entry][2], hess_i[entry][1])
    hess_j_vec[entry] = sub2ind(dims, hess_j[entry][2], hess_j[entry][1])
  end
  new_hess_sparse =
    sparse(hess_i_vec, hess_j_vec, hess_val, prod(dims), prod(dims));

  # Guarantee exact symmetry.
  new_hess = 0.5 * (new_hess_sparse + new_hess_sparse')
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

Returns:
  - iter_count: The number of iterations taken
  - max_f: The maximum function value achieved
  - max_x: The optimal function input
  - ret: The return code of optimize()
""" ->
function maximize_f(
  f::Function, tiled_blob::TiledBlob, mp::ModelParams,
  transform::Transform.DataTransform;
  omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose=false,
  max_iters=100, rho_lower=0.25, fast_hessian=true)

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
    omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose = false,
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
