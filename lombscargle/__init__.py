# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from ._lombscargle_slow import lombscargle_slow
    from ._lombscargle_scipy import lombscargle_scipy
    from ._lombscargle_matrix import lombscargle_matrix
    from ._lombscargle_fast import lombscargle_fast
