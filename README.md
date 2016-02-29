Celeste.jl
========

[![Build Status](https://travis-ci.org/jeff-regier/Celeste.jl.svg?branch=master)](https://travis-ci.org/jeff-regier/Celeste.jl)
[![Coverage Status](https://coveralls.io/repos/jeff-regier/Celeste.jl/badge.svg?branch=master)](https://coveralls.io/r/jeff-regier/Celeste.jl?branch=master)


Celeste.jl finds and characterizes stars and galaxies in astronomical images.
It implements variational inference for the generative model described in

> [Jeffrey Regier, Andrew Miller, Jon McAuliffe, Ryan Adams, Matt Hoffman,
> Dustin Lang, David Schlegel, and Prabhat. “Celeste: Variational inference for
> a generative model of astronomical images”. In: *Proceedings of the 32nd 
> International Conference on Machine Learning (ICML)*. 2015.](
> http://www.stat.berkeley.edu/~jeff/publications/regier2015celeste.pdf)


Usage
-----

The main entry point is `bin/celeste.jl`. Run `celeste.jl --help` for detailed
usage information.

Note that in the `score` mode, the script requires data downloaded from
the CasJobs Stripe82 database in a given RA, Dec range. Here's an example query
in the RA, Dec range [0, 1], [0, 1]:

```sql

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
into mydb.s82_0_1_0_1
from stripe82.photoobj
where
  run in (106, 206) and
  ra between 0. and 1. and
  dec between 0. and 1.
```

Then download the `mydb.s82_0_1_0_1` table as a FITS file.

When scoring, one must use RUN, CAMCOL, FIELD combinations that are entirely
within the RA, Dec range selected above. To find such fields, run the following
query:

```sql
select distinct run, camcol, field
from dr8.frame
where
  rerun = 301 and
  ramin > 0 and ramax < 1 and
  decmin > 0 and decmax < 1
order by run
```



#### License

Celeste.jl is free software, licensed under version 2.0 of the Apache
License.

