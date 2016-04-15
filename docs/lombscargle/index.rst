.. _lombscargle:

*******************************************
Lomb-Scargle Periodograms (``lombscargle``)
*******************************************

The Lomb-Scargle Periodogram (after Lomb [1]_, and Scargle [2]_)
is a commonly-used statistical tool designed to detect periodic signals
in unevenly-spaced observations.
The ``lombscargle`` package contains a unified interface to several
implementations of the Lomb-Scargle periodogram, including a fast *O[NlogN]*
implementation following the algorithm presented by Press & Rybicki [3]_.

The code here is adapted from the `gatspy`_ package,
described in [4]_ and [5]_.

.. _gatspy: http://astroml.org/gatspy

.. toctree::
  :maxdepth: 1

  quickstart.rst
  implementations.rst
  example.rst



Literature References
=====================
.. [1] Lomb, N.R. "Least-squares frequency analysis of unequally spaced data".
       Ap&SS 39 pp. 447-462 (1976)
.. [2] Scargle, J. D. "Studies in astronomical time series analysis. II -
       Statistical aspects of spectral analysis of unevenly spaced data".
       ApJ 1:263 pp. 835-853 (1982)
.. [3] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
       of unevenly sampled data". ApJ 1:338, p. 277 (1989)
.. [4] VanderPlas, J. "Gatspy: General Tools for Astronomical Time Series
       in Python" (2015) http://dx.doi.org/10.5281/zenodo.14833
.. [5] VanderPlas, J. & Ivezic, Z. "Periodograms for Multiband Astronomical
       Time Series". ApJ 812.1:18 (2015)


API Reference
=============

.. automodapi:: lombscargle

.. automodapi:: lombscargle.implementations

.. automodapi:: lombscargle.heuristics
