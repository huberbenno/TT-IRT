import numpy as np
import torch
from numba import njit

class Diffusion1D:
  """
  Solve simple diffusion problem on [0,1] using P1 finite elements.
  """

  def __init__(self, N):
    self.N = N

  def matrix(self, coeff):
    """
    Assemble components of matrix.
    """

    if len(coeff) != 2:
      raise Exception(f"Wrong number of matrix components ({len(coeff)} != ).")

    A0 = self._matrix0(coeff[0], self.N)
    A1 = self._matrix1(coeff[1], self.N)

    return [A0, A1]

  @staticmethod
  # @njit
  def _matrix0(coeff, N):
    # diffusion component
    [nx, I] = np.shape(coeff)
    a = coeff
    A = []
    for i in range(I):        
      # assemble A, a piecewise constant on grid
      # linearly dependent on a
      Ai = np.zeros((N+1)**2)
      Ai[N+2:-1:N+2] = (a[1:,i] + a[:-1, i]) * N # set diag
      Ai[1::N+2] = -a[:,i] * N # set first upper diag
      Ai[N+1::N+2] = -a[:,i] * N # set first lower diag
      Ai = Ai.reshape((N+1, N+1))
      Ai[0] = 0 # for dirichlet boundary
      Ai[-1] = 0 # for dirichlet boundary
      A.append(Ai)

    return A

  @staticmethod
  @njit
  def _matrix1(coeff, N):
    # constant part (from boundary)
    [nx, I] = np.shape(coeff)
    A = []
    for i in range(I):
      # constant part of matrix
      Ai = np.zeros((N+1,N+1))
      Ai[0,0] = 1
      Ai[-1,-1] = 1
      A.append(Ai)

    return A

  def rhs(self, coeff):
    """
    Assemble components of rhs.
    """

    if len(coeff) != 1:
      raise Exception(f"Wrong number of rhs components ({len(coeff)} != 1).")

    F0 = self._rhs0(coeff[0], self.N)

    return [F0]

  @staticmethod
  @njit
  def _rhs0(coeff, N):
    [nx, I] = np.shape(coeff)
    
    # constant rhs
    F = []
    for i in range(I):
      # constant part of rhs
      Fi = np.full((N+1,1), 10/N) # assume RHS = 1 
      Fi[0] = 0 # boundary condition
      Fi[-1] = 0 # boundary condition
      F.append(Fi)

    return F

  def solve(self, coeff_A, coeff_b):
    """
    Assemble and solve FE problem for coeff.
    """

    I = np.shape(coeff_A[0])[1]
    
    U = []

    for i in range(I):
      A = self.matrix(tuple(coeff_A[k][:,i, np.newaxis] for k in range(len(coeff_A))))
      b = self.rhs(tuple(coeff_b[k][:,i, np.newaxis] for k in range(len(coeff_b))))
      U.append(np.linalg.solve(A[0][0] + A[1][0], b[0][0]).reshape((-1,1)))
    
    return U

class Diffusion1D_torch(Diffusion1D):

  def matrix(self, coeff):
    """
    Assemble components of matrix.
    """

    if len(coeff) != 2:
      raise Exception(f"Wrong number of matrix components ({len(coeff)} != 2).")

    A0 = [torch.from_numpy(a) for a in self._matrix0(coeff[0].numpy(force=True), self.N)]
    A1 = [torch.from_numpy(a) for a in self._matrix1(coeff[1].numpy(force=True), self.N)]

    return [A0, A1]

  def rhs(self, coeff):
    """
    Assemble components of rhs.
    """

    if len(coeff) != 1:
      raise Exception(f"Wrong number of rhs components ({len(coeff)} != 1).")

    F0 = [torch.from_numpy(a) for a in self._rhs0(coeff[0].numpy(force=True), self.N)]

    return [F0]

  def solve(self, coeff_A, coeff_b):
    """
    Assemble and solve FE problem for coeff.
    """

    I = np.shape(coeff_A[0])[1]
    
    U = []

    for i in range(I):
      A = self.matrix(tuple(coeff_A[k][:,i, np.newaxis] for k in range(len(coeff_A))))
      b = self.rhs(tuple(coeff_b[k][:,i, np.newaxis] for k in range(len(coeff_b))))
      U.append(torch.linalg.solve(A[0][0] + A[1][0], b[0][0]).reshape((-1,1)))
    
    return U