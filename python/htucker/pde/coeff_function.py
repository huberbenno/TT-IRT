import numpy as np
from numba import jit, njit


class coeff:
  def __init__(self, Nx, Ny, offset, var, decay = 2):
    self.Nx = Nx
    self.Ny = Ny
    self.offset = offset
    self.var = var
    self.decay = float(decay)

  def __call__(self, X):
    return self._call_static(X, self.offset, self.var, self.Nx, self.Ny, self.decay)

  @staticmethod
  @njit
  def _call_static(X, offset, var, Nx, Ny, decay):
    c = np.full(len(X), offset, dtype=float)
    for i, x in enumerate(X):
      k = np.arange(1,len(x))
      c[i] += np.sum(k**-decay * np.sin(np.pi*k*x[0]/(Nx-1)) \
                      * (x[k]/(Ny-1) - 0.5) * var)
    return np.exp(c)

class coeff_convdiff:
  def __init__(self, Nx, Ny, offset, var, decay = 2):
    self.Nx = Nx
    self.Ny = Ny
    self.offset = offset
    self.var = var
    self.decay = float(decay)

  def __call__(self, X):
    return self._call_static(X, self.offset, self.var, self.Nx, self.Ny, self.decay)

  @staticmethod
  @njit
  def _call_static(X, offset, var, Nx, Ny, decay):
    c = np.full(len(X), offset, dtype=float)
    for i, x in enumerate(X):
      k = np.arange(2,len(x))
      c[i] += np.sum(k**-decay * np.sin(np.pi*k*x[0]/(Nx-1)) \
                      * (x[k]/(Ny-1) - 0.5) * var)
    return np.exp(c)