using Dates
using Base.Test

# Pull out the tests from ElboJuMP.jl

############################
# Test the brightness
jump_e_l_a = Float64[ ReverseDiffSparse.getvalue(E_l_a[s, b, a], celeste_m.colVal)
                      for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

celeste_e_l_a = [ ElboDeriv.SourceBrightness(mp.vp[s]).E_l_a[b, a].v
                  for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

jump_e_ll_a = Float64[ ReverseDiffSparse.getvalue(E_ll_a[s, b, a], celeste_m.colVal)
                       for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

celeste_e_ll_a = [ ElboDeriv.SourceBrightness(mp.vp[s]).E_ll_a[b, a].v
                  for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

@test_approx_eq jump_e_l_a celeste_e_l_a
@test_approx_eq jump_e_ll_a celeste_e_ll_a

############################
# Test the bivariate normal functions

star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blobs[1].psf, mp);

### Check the stars:
celeste_star_mean = [
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[1][k, s].the_mean[row]
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2 ];
jump_star_mean =
	[ ReverseDiffSparse.getvalue(star_mean[b, s, k, row],
   							  celeste_m.colVal)
       for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2 ];
@test_approx_eq celeste_star_mean jump_star_mean

celeste_star_precision = [
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[1][k, s].precision[row, col]
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2 ];
jump_star_precision =
	[ ReverseDiffSparse.getvalue(star_precision[b, s, k, row, col],
   							     celeste_m.colVal)
       for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2 ];

@test_approx_eq celeste_star_precision jump_star_precision

celeste_star_z = [
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[1][k, s].z
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp ];
jump_star_z =
	[ ReverseDiffSparse.getvalue(star_z[b, s, k],
   							     celeste_m.colVal)
       for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp ];
@test_approx_eq celeste_star_z jump_star_z




############
# Test the galaxies

jump_galaxy_type1_precision = Float64[
	ReverseDiffSparse.getvalue(galaxy_type1_precision[b, s, k, g_k, row, col],
  						       celeste_m.colVal)
		for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal1_comp,
		row=1:2, col=1:2 ];

celeste_galaxy_type1_precision = Float64[
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[2][k, g_k, 1, s].bmc.precision[row, col]
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal1_comp,
    row=1:2, col=1:2 ];

@test_approx_eq celeste_galaxy_type1_precision jump_galaxy_type1_precision


jump_galaxy_type2_precision = Float64[
	ReverseDiffSparse.getvalue(galaxy_type2_precision[b, s, k, g_k, row, col],
  						       celeste_m.colVal)
		for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal2_comp,
		row=1:2, col=1:2 ];

celeste_galaxy_type2_precision = Float64[
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[2][k, g_k, 2, s].bmc.precision[row, col]
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal2_comp,
    row=1:2, col=1:2 ];

@test_approx_eq celeste_galaxy_type2_precision jump_galaxy_type2_precision


jump_galaxy_type1_z = Float64[
	ReverseDiffSparse.getvalue(galaxy_type1_z[b, s, k, g_k],
			    			   celeste_m.colVal)
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal1_comp ];

celeste_galaxy_type1_z = Float64[
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[2][k, g_k, 1, s].bmc.z
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal1_comp ];

@test_approx_eq celeste_galaxy_type1_z jump_galaxy_type1_z

jump_galaxy_type2_z = Float64[
	ReverseDiffSparse.getvalue(galaxy_type2_z[b, s, k, g_k],
			    			   celeste_m.colVal)
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal2_comp ];

celeste_galaxy_type2_z = Float64[
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[2][k, g_k, 2, s].bmc.z
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal2_comp ];

@test_approx_eq celeste_galaxy_type2_z jump_galaxy_type2_z


# For deeper debugging, get a var_s from the original code:
#
# rho = mp.vp[s][ids.rho]
# phi = mp.vp[s][ids.phi]
# sigma = mp.vp[s][ids.sigma]
# pc = blobs[b].psf[k]
# gc = galaxy_prototypes[1][g_k]
# XiXi = Util.get_bvn_cov(rho, phi, sigma)
# mean_s = [0 0]
# var_s = pc.SigmaBar + gc.sigmaTilde * XiXi
# weight = pc.alphaBar * gc.alphaTilde  # excludes theta

# jump_xixi = 
# 	[ ReverseDiffSparse.getvalue(galaxy_xixi_mat[s, row, col],
#    							     celeste_m.colVal)
#        for row=1:2, col=1:2 ];

# jump_var_s = 
# 	[ ReverseDiffSparse.getvalue(galaxy_type1_var_s[b, s, k, g_k, row, col],
#    							     celeste_m.colVal)
#        for row=1:2, col=1:2 ];
# jump_var_s - var_s

# jump_galaxy_type1_det =
# 	ReverseDiffSparse.getvalue(galaxy_type1_det[b, s, k, g_k],
#    							     celeste_m.colVal)
# det(var_s) - jump_galaxy_type1_det


############################
# Test the galaxy and star functions

# Get the Celeste values:
celeste_star_pdf_f = zeros(Float64, CelesteTypes.B, mp.S, n_pcf_comp, img_w, img_h);
celeste_gal1_pdf_f = zeros(Float64, CelesteTypes.B, mp.S,
	                       n_pcf_comp, n_gal1_comp,
						   img_w, img_h);
celeste_gal2_pdf_f = zeros(Float64, CelesteTypes.B, mp.S,
	                       n_pcf_comp, n_gal2_comp,
						   img_w, img_h);

for img=1:CelesteTypes.B
	blob_img = blobs[img]
	star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob_img.psf, mp)
    for w in 1:img_w, h in 1:img_h
        m_pos = Float64[h, w]
        for s in 1:mp.S, k in 1:n_pcf_comp
        	if pixel_source_indicators[s, w, h] == 1
		    	py1, py2, f = ElboDeriv.ret_pdf(star_mcs[k, s], m_pos)
		    	celeste_star_pdf_f[img, s, k, w, h] = f
		    	for g_k in 1:n_gal1_comp
		    		py1, py2, f = ElboDeriv.ret_pdf(gal_mcs[k, g_k, 1, s].bmc, m_pos)
		    		celeste_gal1_pdf_f[img, s, k, g_k, w, h] = f
		    	end
		    	for g_k in 1:n_gal2_comp
		    		py1, py2, f = ElboDeriv.ret_pdf(gal_mcs[k, g_k, 2, s].bmc, m_pos)
		    		celeste_gal2_pdf_f[img, s, k, g_k, w, h] = f
		    	end
		    end
        end
    end
end

jump_star_pdf_f = Float64[
	ReverseDiffSparse.getvalue(star_pdf_f[img, s, k, pw, ph], celeste_m.colVal) 
	for img=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp,
	           pw=1:img_w, ph=1:img_h ];

jump_gal1_pdf_f = Float64[
	ReverseDiffSparse.getvalue(galaxy_type1_pdf_f[img, s, k, g_k, pw, ph], celeste_m.colVal) 
	for img=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal1_comp,
	           pw=1:img_w, ph=1:img_h ];

jump_gal2_pdf_f = Float64[
	ReverseDiffSparse.getvalue(galaxy_type2_pdf_f[img, s, k, g_k, pw, ph], celeste_m.colVal) 
	for img=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, g_k=1:n_gal2_comp,
	           pw=1:img_w, ph=1:img_h ];

@test_approx_eq celeste_star_pdf_f jump_star_pdf_f
@test_approx_eq celeste_gal1_pdf_f jump_gal1_pdf_f
@test_approx_eq celeste_gal2_pdf_f jump_gal2_pdf_f


#######################
# Test the G terms

jump_fs0m = Float64[ ReverseDiffSparse.getvalue(fs0m[b, s, pw, ph], celeste_m.colVal)
			         for b=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];
celeste_fs0m = Float64[ sum(celeste_star_pdf_f[img, s, :, pw, ph])
                        for img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];
@test_approx_eq	celeste_fs0m jump_fs0m

jump_fs1m = Float64[ ReverseDiffSparse.getvalue(fs1m[b, s, pw, ph], celeste_m.colVal)
			         for b=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];
celeste_fs1m_gal1 = Float64[ sum(celeste_gal1_pdf_f[img, s, :, :, pw, ph])
                             for img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];
celeste_fs1m_gal2 = Float64[ sum(celeste_gal2_pdf_f[img, s, :, :, pw, ph])
                             for img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];

# Very strangely, this gives an error if the line is broken after the + sign.
# ERROR: syntax: invalid comprehension syntax
celeste_fs1m = Float64[ celeste_fs1m_gal1[img, s, pw, ph] * mp.vp[s][ids.theta] + celeste_fs1m_gal2[img, s, pw, ph] * (1 - mp.vp[s][ids.theta])
                        for img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];

@test_approx_eq	celeste_fs1m jump_fs1m

##################################
# Test the G terms more directly

jump_E_G_s = Float64[ ReverseDiffSparse.getvalue(E_G_s[b, s, pw, ph], celeste_m.colVal)
			          for b=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h ];

sbs = [ ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S ];

star_mcs_array = [ ElboDeriv.load_bvn_mixtures(blobs[img].psf, mp)[1] for img=1:CelesteTypes.B ];
gal_mcs_array = [ ElboDeriv.load_bvn_mixtures(blobs[img].psf, mp)[2] for img=1:CelesteTypes.B ];

raw_celeste_fs0m = zeros(Float64, CelesteTypes.B, mp.S, img_w, img_h);
raw_celeste_fs1m = zeros(Float64, CelesteTypes.B, mp.S, img_w, img_h);
raw_celeste_e_g = zeros(Float64, CelesteTypes.B, mp.S, img_w, img_h);

for img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h
	all_sources = 1:mp.S
	these_sources = Bool[ pixel_source_indicators[s, pw, ph] == 1 for s=1:mp.S ]
	these_local_sources = all_sources[these_sources]

	this_fs0m = zero_sensitive_float([-1], star_pos_params)
	this_fs1m = zero_sensitive_float([-1], galaxy_pos_params)

	E_G = zero_sensitive_float(these_local_sources, all_params)
	var_G = zero_sensitive_float(these_local_sources, all_params)
	# Note that each pixel gets only one epsilon term, so it is
	# not included in E_G_s.

	ElboDeriv.accum_pixel_source_stats!(sbs[s],
								        star_mcs_array[img],
								        gal_mcs_array[img],
								        mp.vp[s],
								        1, s,
								        Float64[ph, pw], img,
								        this_fs0m, this_fs1m,
								        E_G, var_G)
	raw_celeste_fs0m[img, s, pw, ph] = this_fs0m.v
	raw_celeste_fs1m[img, s, pw, ph] = this_fs1m.v
	raw_celeste_e_g[img, s, pw, ph] = E_G.v

end

@test_approx_eq raw_celeste_fs0m jump_fs0m
@test_approx_eq raw_celeste_fs1m jump_fs1m
@test_approx_eq raw_celeste_e_g jump_E_G_s


#############################
# Test the log likelihood

celeste_time = now()
celeste_elbo_lik = ElboDeriv.elbo_likelihood(blobs, mp).v
now()
celeste_time = now() - celeste_time

jump_time = now()
jump_elbo_lik = ReverseDiffSparse.getvalue(elbo_log_likelihood, celeste_m.colVal)
now()
jump_time = now() - jump_time

@test_approx_eq	celeste_elbo_lik jump_elbo_lik






#################################
# Optimize?

@setNLObjective(celeste_m, Max, elbo_log_likelihood)

solve_time = now()
solve(celeste_m)
now()
solve_time = now() - solve_time
