-- This query, run in casjobs, selects ground truth data from stripe 82
-- for a region containing field (4263, 5, 119)

declare @BRIGHT bigint set @BRIGHT=dbo.fPhotoFlags('BRIGHT')    
declare @EDGE bigint set @EDGE=dbo.fPhotoFlags('EDGE')  
declare @SATURATED bigint set @SATURATED=dbo.fPhotoFlags('SATURATED')   
declare @NODEBLEND bigint set @NODEBLEND=dbo.fPhotoFlags('NODEBLEND')   
declare @bad_flags bigint set   
@bad_flags=(@SATURATED|@BRIGHT|@EDGE|@NODEBLEND)    
  
  
select *
from (
select
  objid, rerun, run, camcol, field, flags,
  ra, dec, probpsf,
  psfmag_u, psfmag_g, psfmag_r, psfmag_i, psfmag_z,
  devmag_u, devmag_g, devmag_r, devmag_i, devmag_z,
  expmag_u, expmag_g, expmag_r, expmag_i, expmag_z,
  fracdev_r,
  devab_r, expab_r,
  devphi_r, expphi_r,
  devrad_r, exprad_r
into coadd_field_catalog
from stripe82.photoobj
where
  run in (106, 206) and
  ra between 0.449 and 0.599 and
  dec between 0.417 and 0.629) as tmp
where
  ((psfmag_i < 22 and probpsf = 1) or (probpsf = 0 and (expmag_i < 22 or devmag_i < 22))) and
  (flags & @bad_flags) = 0

