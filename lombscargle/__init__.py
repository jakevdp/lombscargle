# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
lombscargle
===========
AstroPy-compatible implementation of the classic lomb-scargle periodogram.
"""

__all__ = ['lombscargle', 'LombScargle']

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from ._lombscargle import LombScargle
    from .implementations import lombscargle
