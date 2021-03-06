# Lomb-Scargle periodograms for astropy

**Note: the code here is now part of astropy and this repository is no longer maintained.**

[![build status](http://img.shields.io/travis/jakevdp/lombscargle/master.svg?style=flat)](https://travis-ci.org/jakevdp/lombscargle)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/jakevdp/lombscargle/blob/master/LICENSE.rst)
[![powered by astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)

This repository contains an implementation of the Lomb-Scargle periodogram for
use with [astropy](http://astropy.org). The implementation is based on the
[gatspy](http://astroml.org/gatspy/) package, but is enhanced to work within
the ``astropy.units`` framework.

The fast periodogram functionality requires numpy 1.8 or newer.

The documentation build can be found at http://jakevdp.github.io/lombscargle/
