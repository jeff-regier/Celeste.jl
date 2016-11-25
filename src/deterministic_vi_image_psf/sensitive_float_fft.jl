"""
Convolve a matrix of sensitive floats (represented by conv_fft) with a matrix of reals.

Args:
  - sf_matrix: A matrix of sensitive floats arranged spatially
  - conv_fft: Pre-allocated memory the same size as sf_matrix
  - sf_matrix_out: The FFT of the signal you want to convolve, same size as sf_matrix
"""
function convolve_sensitive_float_matrix!(
            sf_matrix::Matrix{SensitiveFloat{Float64}},
            conv_fft::Matrix{Complex{Float64}},
            sf_matrix_out::Matrix{SensitiveFloat{Float64}})

    @assert size(sf_matrix) == size(conv_fft)

    # Pre-allocate memory.
    fft_matrix = zeros(Complex{Float64}, size(sf_matrix))
    n_active_sources = size(sf_matrix[1].d, 2)

    h_range = 1:size(sf_matrix, 1)
    w_range = 1:size(sf_matrix, 2)

    for h in h_range, w in w_range
      fft_matrix[h, w] = sf_matrix[h, w].v[]
    end
    fft!(fft_matrix)
    fft_matrix .*= conv_fft
    ifft!(fft_matrix)
    for h in h_range, w in w_range
        sf_matrix_out[h, w].v[] = real(fft_matrix[h, w]);
    end

    for sa_d in 1:n_active_sources, ind in 1:sf_matrix[1,1].local_P
        for h in h_range, w in w_range
          fft_matrix[h, w] = sf_matrix[h, w].d[ind, sa_d]
        end
        fft!(fft_matrix)
        fft_matrix .*= conv_fft
        ifft!(fft_matrix)
        for h in h_range, w in w_range
            sf_matrix_out[h, w].d[ind, sa_d] = real(fft_matrix[h, w]);
        end
    end

    for ind1 in 1:size(sf_matrix[1].h, 1), ind2 in 1:ind1
        for h in h_range, w in w_range
          # TOOD: avoid this copy?
          fft_matrix[h, w] = sf_matrix[h, w].h[ind1, ind2]
        end
        fft!(fft_matrix)
        fft_matrix .*= conv_fft
        ifft!(fft_matrix)
        for h in h_range, w in w_range
            sf_matrix_out[h, w].h[ind1, ind2] = sf_matrix_out[h, w].h[ind2, ind1] =
                real(fft_matrix[h, w]);
        end
    end

    sf_matrix_out
end
