![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


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

Note that in the `score` mode, the script requires data downloaded from the CasJobs Stripe82
database in a given RA, Dec range. [See here][1] for information on downloading this data from the
SDSS CasJobs server.

[1]: https://github.com/jeff-regier/Celeste.jl/wiki/About-SDSS-and-Stripe-82#how-to-get-ground-truth-data-for-stripe-82

#### License

Celeste.jl is free software, licensed under version 2.0 of the Apache
License.

