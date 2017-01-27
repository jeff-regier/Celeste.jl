using Base.Test
using Celeste.DeterministicVIImagePSF: lanczos_kernel, sinc_with_derivatives,
    lanczos_kernel_with_derivatives_nocheck, lanczos_kernel_with_derivatives,
    bspline_kernel_with_derivatives, cubic_kernel_with_derivatives

for x in linspace(0.01, 3, 4), y in linspace(0.01, 3, 4), T in (Float32, Float64)
    xx, yy = T(x), Float64(y)

    @inferred lanczos_kernel(xx, yy)
    @inferred sinc_with_derivatives(xx)
    @inferred lanczos_kernel_with_derivatives_nocheck(xx, yy)
    @inferred lanczos_kernel_with_derivatives(xx, yy)
    @inferred bspline_kernel_with_derivatives(xx)
    @inferred cubic_kernel_with_derivatives(xx, yy)
end