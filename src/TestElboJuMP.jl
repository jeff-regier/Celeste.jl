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


#################################
# Optimize?

@setNLObjective(celeste_m, Max, elbo_log_likelihood)

solve_time = now()
solve(celeste_m)
now()
solve_time = now() - solve_time
