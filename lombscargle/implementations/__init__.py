"""Various implementations of the Lomb-Scargle Periodogram"""

from ._lombscargle_matrix import lombscargle_matrix
from ._lombscargle_scipy import lombscargle_scipy
from ._lombscargle_slow import lombscargle_slow
from ._lombscargle_fast import lombscargle_fast
