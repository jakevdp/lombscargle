"""Various implementations of the Lomb-Scargle Periodogram"""

from .main import lombscargle
from .matrix_impl import lombscargle_matrix
from .scipy_impl import lombscargle_scipy
from .slow_impl import lombscargle_slow
from .fast_impl import lombscargle_fast
