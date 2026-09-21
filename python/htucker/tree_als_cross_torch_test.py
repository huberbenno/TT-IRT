import numpy as np
import matplotlib.pyplot as plt

class SimplePDE:
  """
  Solve simple diffusion problem on [0,1] using P1 finite elements.
  """

  def __init__(self, N):
    self.N = N

  def matrix(self, coeff):
    """
    Assemble components of matrix.
    """

    if len(coeff) != 3:
      raise Exception(f"Wrong number of matrix components ({len(coeff)} != 3).")

    # diffusion component
    [nx, I] = np.shape(coeff[0])
    a = coeff[0][:,:]
    A0 = []
    for i in range(I):        
      # assemble A, a piecewise constant on grid
      # linearly dependent on a
      Ai = np.zeros((self.N+1,self.N+1))
      Ai.flat[self.N+2::self.N+2] = (a[1:,i] + a[:-1, i]) * self.N # set diag
      Ai.flat[1::self.N+2] = -a[:,i] * self.N # set first upper diag
      Ai.flat[self.N+1::self.N+2] = -a[:,i] * self.N # set first lower diag
      Ai[[0,-1]] = 0 # for dirichlet boundary
      A0.append(Ai)

    # convection component
    [nx, I] = np.shape(coeff[1])
    v = coeff[1][0,:]
    # v= np.zeros_like(v)
    A1 = []
    for i in range(I):
      # linearly dependent on v
      Ai = np.zeros((self.N+1,self.N+1))
      Ai.flat[1::self.N+2] = v[i]# set first upper diag
      Ai.flat[self.N+1::self.N+2] = -v[i] # set first lower diag
      Ai[[0,-1]] = 0 # for dirichlet boundary
      A1.append(Ai)

    # constant part (from boundary)
    [nx, I] = np.shape(coeff[2])
    A2 = []
    for i in range(I):
      # constant part of matrix
      Ai = np.zeros((self.N+1,self.N+1))
      Ai[0,0] = 1
      Ai[-1,-1] = 1
      A2.append(Ai)

    return [A0, A1, A2]
  
  def rhs(self, coeff):
    """
    Assemble components of rhs.
    """

    if len(coeff) != 1:
      raise Exception(f"Wrong number of rhs components ({len(coeff)} != 1).")
    
    [nx, I] = np.shape(coeff[0])
    
    # constant rhs
    F0 = []
    for i in range(I):
      # constant part of rhs
      Fi = np.full((self.N+1,1), 10/(self.N)) # assume RHS = 1 
      Fi[[0,-1]] = 0 # boundary condition
      F0.append(Fi)

    return [F0]

  def solve(self, coeff):
    """
    Assemble and solve FE problem for coeff.
    """
    I = np.shape(coeff[0][0])[1]
    
    U = []

    for i in range(I):
      A = self.matrix([coeff[0][k][:,i, np.newaxis] for k in range(len(coeff[0]))])
      b = self.rhs([coeff[1][k][:,i, np.newaxis] for k in range(len(coeff[1]))])
      U.append(np.linalg.solve(A[0][0] + A[1][0] + A[2][0], b[0][0]).reshape((-1,1)))
    
    return U

class diffusion_coeff:

  def __init__(self, Nx, Ny, offset, var):
    self.Nx = Nx
    self.Ny = Ny
    self.offset = offset
    self.var = var

  def __call__(self, X):
    c = np.full(len(X), self.offset,dtype=float)
    for i, x in enumerate(X):
      k = np.arange(1,len(x)-1)
      c[i] += np.sum(k**-2. * np.sin(np.pi*k*x[0]/(self.Nx-1)) \
                      * (x[k]/(self.Ny-1) - 0.5) * self.var)
      # c[i] += np.sum(k**-2. * np.sin(np.pi*k*x[0]/(self.Nx-1)) \
                      # * (x[k+1]/(self.Ny-1) - 0.5) * self.var)
    return np.exp(c)

Nx = 200    # spatial resolution
Ny = 5       # parameter resolution
n_a_param = 100 # number of parameters for diffusion coeff
n_param = n_a_param + 1
offset = 3   # parameter mean
var = 4       # 'variance' of the parameters
v_min, v_max = -10, 10

# get random params
rng = np.random.default_rng()
y = rng.uniform(0, Ny-1, n_a_param)
v = rng.uniform(v_min, v_max)
X = np.hstack([np.arange(Nx).reshape((-1,1)), np.zeros((Nx,1)), np.tile(y, [Nx,1])])
# X = np.hstack([np.arange(Nx).reshape((-1,1)), np.tile(y, [Nx,1])])

cfun = diffusion_coeff(Nx, Ny, offset, var)

C_true = cfun(X)

PDE_fun = SimplePDE(Nx)

import torch
from tree import Tree
from python.htucker.tree_tensor.tree_tensor_torch import TreeBasedTensor
from python.htucker.tree_cross.tree_cross_torch import TreeCross
from python.htucker.tree.tree_util import balanced_binary_tupletree, linear_tupletree

tol = 1e-3
param_shape = tuple([Ny] * n_param)

tree = Tree.from_tupletree(balanced_binary_tupletree(n_param))
# tree = Tree.from_tupletree((0,(1,2)))
tree.print()
C_a = TreeBasedTensor.randn(tree, (Nx,) + param_shape, Ny, dtype=torch.float64)

cross = TreeCross(C_a)
cross.run(cfun, n_iter=50, eps=tol, kickrank=1, verbose=True)
C_a.round(tol=tol)

# tensor for convection
C_v = TreeBasedTensor.ones(tree, (1,) + param_shape)
C_v.cores[C_v.tree.dim2leaf(1)] = torch.linspace(v_min,v_max, Ny, dtype=torch.float64).reshape(-1,1)
# C_v.cores[C_v.tree.dim2leaf(1)] = np.zeros(Ny).reshape(-1,1)

# helper tensor that is constant 1
C_const = TreeBasedTensor.ones(tree, (1,) + param_shape)
C_const2 = TreeBasedTensor.ones(tree, (1,) + param_shape)

from python.htucker.tree_als_cross.tree_als_cross_torch import TreeALSCross
import cProfile

test = TreeALSCross(
  [C_a, C_v, C_const],
  [C_const],
  PDE_fun,
  rinit=0,
  verbose=1,
  kickrank=5
  )


import subprocess
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

stats = cProfile.run('test.run(n_iter=5)', os.path.join(dir_path, 'profile.prof'))
# test.run(n_iter=2)

ps = subprocess.run(['gprof2dot', '-f', 'pstats', os.path.join(dir_path, 'profile.prof'),],capture_output=True, check=True)
subprocess.run(['dot', '-Tpng', '-o', os.path.join(dir_path, 'output.png')], input=ps.stdout, check=True)