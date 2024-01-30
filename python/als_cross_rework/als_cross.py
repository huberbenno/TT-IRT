import numpy as np
import tt
import warnings
from time import perf_counter

from localcross import localcross


class als_cross:
  """
  Class implementing the ALS cross algorithm.
  """

  class profiler:
    """
    Profiler providing timers and counters.
    """
    def __init__(self, timers=None, counters=None):
      """
      Parameters
      ----------
      timers : sequence of strings
        List of named timers.
      counters : sequence of strings
        List of named counters.
      """
      self.t_start = {}
      self.t_total = {}
      for timer in timers:
        self.t_total[timer] = 0.0

      self.counters = {}
      for counter in counters:
        self.counters[counter] = 0
    
    def start(self, timer):
      self.t_start[timer] = perf_counter()

    def stop(self, timer):
      self.t_total[timer] += perf_counter() - self.t_start[timer]

    def increment(self, counter, n=1):
      self.counters[counter] += n

    def get_data(self):
      return self.t_total | self.counters

  def parse_args(self, args) -> None:
    """
    Parse list of (argname,val) tuples and set corresponding parameters.

    Supported arguments
    -------------------
    nswp : integer
      number of iterations (forward + backward)
    kickrank: integer
      enrichment rank
    random_init: integer
      If >0, number of random indices for init. If 0, indices of parameter are
      used.
    """
    # default values for optional parameters
    self.nswp = 5
    self.kickrank = 10 
    self.random_init = 0
    self.verbose = 1

    # parse parameters
    for (arg, val) in args.items():
      match arg:
        case 'nswp':
          self.nswp = val
        case 'kickrank':
          self.kickrank = val
        case 'random_init':
          self.random_init = val
        case 'verbose':
          self.verbose = val
        case _:
          warnings.warn('unknown argument \'' + arg + '\'', SyntaxWarning)
  

  def orthogonalize_tensor(C0, cores, return_indices=True):
    """
    Orthogonalize TT tensor.

    Parameters
    ----------
    cores: sequence of ndarrays
      list of the TT cores with i-th core in shape (r_i, n_i, r_{i+1}).
    return_indices: bool
      whether to return the maxvol indices. Default is `True`.

    Returns
    -------
    C0: ndarray
      updated first core
    cores: sequence of ndarrays
      list with rest of updated TT cores
    r : ndarray
      array containing updated TT ranks
    (indices) : sequence of ndarrays, optional
      list with maxvol indices
    """
    d = len(cores)
    r = np.ones(d, dtype=np.int32)
    v = np.ones((1,1))
    if return_indices:
      indices = []

    for i in reversed(range(1, d)):
      # cast non-orth block from previous iteration into core
      r[i-1], n = cores[i].shape[:2]
      cr = cores[i].reshape((r[i-1]*n, -1))
      cr = cr @ v
      # QR core (note transpose)
      cr = cr.reshape((r[i-1], -1)).T
      cr, v = np.linalg.qr(cr)
      # update rank (in case core has rank deficiency)
      r[i] = cr.shape[1]
      # compute cross approximation
      ind = tt.maxvol.maxvol(cr)
      if return_indices:
        indices = [ind] + indices
      
      cr = cr.T
      CC = cr[:, ind]
      cr = np.linalg.solve(CC, cr)
      v = v.T @ CC
      # update core
      cores[i] = cr.reshape((r[i-1],n,r[i]))
    
    M, N, R = C0.shape
    C0 = C0.reshape(-1, R) @ v
    C0.reshape(M,N,r[0])

    if return_indices:
      return C0, cores, r, indices
    else:
      return C0, cores, r
  
  def spatial_core_forward(self):
    # store previous U
    U_prev = self.U0

    self.prof.start('t_solve')

    # get matrix and rhs in first iteration
    if self.swp == 1:
      self.U0, self.A0, self.F0 = self.assem_solve_fun(self.Ju)
      self.Nx = self.U0[0].shape[0]
      self.F0 = [np.hstack(F0k) for F0k in self.F0] 
    else:
      self.U0 = self.assem_solve_fun(self.Ju, linear_system=False)

    self.U0 = np.hstack(self.U0)

    self.prof.stop('t_solve')
    self.prof.increment('n_PDE_eval')

    # check error
    dx = 1 
    if U_prev is not None:
      dx = np.linalg.norm(self.U0 - U_prev) / np.linalg.norm(self.U0)

    self.max_dx = max(self.max_dx, dx)

    if self.verbose > 0:
      print(f'=swp={self.swp} core 0, max_dx={self.max_dx:.3e}, max_rank = {max(self.ru)}')

    # exit if tolerance is met
    if self.max_dx < self.tol:
      return True

    # truncate U0
    self.U0, v = localcross(self.U0, self.tol/np.sqrt(self.n_param))
    self.ru[0] = self.U0.shape[1]
    # cast non-orth factor to next core
    if self.swp > 1:
      self.u[0] = v @ self.u[0].reshape(v.shape[-1], -1)
    
    # rank adaption
    if self.kickrank > 0: # TODO and (random_init==0 or swp>1)
      # compute residual at Z indices
      self.Z0 = np.zeros((self.Nx, self.rz[0]))
      cru = self.U0 @ v @ self.ZU[0]
      for k in range(self.Mc):
        for j in range(self.rz[0]):
          crA = np.zeros((self.Nx, self.Nx))
          for l in range(self.rc[0]):
            crA += self.A0[k][l] * self.ZC[0][l,j]

          self.Z0[:,j] += crA @ cru[:,j]

        self.Z0 -= self.F0[k] @ self.ZC[k][0]

      # QR residual
      self.Z0 = np.linalg.qr(self.Z0)[0]
      self.rz[0] = self.Z0.shape[1]
      # append residual to U core
      cru = np.hstack((self.U0, self.Z0))

      # QR enriched core
      self.U0, v = np.linalg.qr(cru)
      if self.swp > 1:
        self.u[0] = v[:,:self.ru[0]] @ self.u[0].reshape(self.ru[0], -1)
      
      self.ru[0] = self.U0.shape[1]
    
    # TODO evaluate if loop for projection are necessary
    # Project onto solution basis U0
    self.prof.start('t_project')
   
    UAU_new = [[None] * self.Mc] * self.rc[0]
    Uprev = self.U0 # TODO technically dont need to copy here
    for k in range(self.Mc):
      for j in range(self.rc[0]):
        UAU_new[k][j] = np.conjugate(Uprev.T) @ self.A0[k][j] @ Uprev
        UAU_new[k][j] = UAU_new[k][j].reshape(-1,1)

    self.UAU[0] = [np.hstack(UAUk) for UAUk in UAU_new]
    self.UF[0] = [np.conjugate(Uprev.T) @ F0k for F0k in self.F0]

    self.prof.stop('t_project')

    # Project onto residual
    if self.kickrank > 0:
      ZU_new = [[None] * self.Mc] * self.rc[0]
      for k in range(self.Mc):
        for j in range(self.rc[0]):
          ZU_new[j] = np.conjugate(self.Z0.T) @ self.A0[j] @ Uprev
          ZU_new[j] = ZU_new[j].reshape(-1,1)
      
      self.ZU[0] = [np.hstack(ZUk) for ZUk in ZU_new]
      # TODO why ZC and not ZF
      self.ZC[0] = [np.conjugate(self.Z0.T) @ F0k for F0k in self.F0]

    return False

  def step_backward(self, i):
    pass

  def __init__(self, params, assem_solve_fun, tol, **args):
    """
    Initialize ALS cross algorithm.

    Parameters
    ----------
    params:
      list of parameters in TT format.
    assem_solve_fun: callable
      function implementing FE solver.
    tol: scalar
      truncation and stopping tolerance
    **args: 
      optional named arguments described in `parse_args`
    """
      
    # store parameters
    self.params = params # TODO treatment of multiple parameters
    self.assem_solve_fun = assem_solve_fun
    self.tol = tol

    # parse optional arguments (set defaults if not given)
    self.parse_args(args)

    # init rng (set seed to get deterministic result)
    self.rng = np.random.default_rng()

    # grid sizes
    self.n_param = self.params[0].n[1:]     # parametric grid sizes
    self.d_param = self.params[0].d - 1     # parameter dimension
    self.rc = [p.r[1:] for p in params]     # TT ranks of the parameters
    self.Mc = len(self.params)    # number of parameter TTs
    self.ru = None                # TT ranks of the solution
    self.rz = None                # TT ranks of the residual
    self.Nx = None

    # orthogonalize parameter TT
    self.c_cores = [None] * len(self.params)
    self.C_core = [None] * len(self.params)
    for i, param in self.params:
      # keep maxvol indices of first param TT 
      if i == 0:
        self.C_core[i], self.c_cores[i], self.rc[i], indices = \
          self.orthogonalize_tensor(tt.vector.to_list(param))
      else:
        self.C_core[i], self.c_cores[i], self.rc[i] = \
          self.orthogonalize_tensor(tt.vector.to_list(param), return_indices=False)

    # init index set
    self.Ju = np.empty((1,0), dtype=np.int32)       # u indices (for PDE eval)
    
    # get random indices
    if self.random_init > 0:
      xi = np.ones((1, self.random_init))
      for i in reversed(range(self.d_param)):
        indices[i] = self.rng.integers(self.n_param[i], size=(self.random_init))
        self.Ju = np.hstack(indices[i], self.Ju)
        self.ru[i] = self.random_init

    # OR use indices derived from (first) param
    else:
      # TODO original code only to i=1. Why?
      for i in reversed(range(self.d_param)):
        indices[i] = indices[i] % self.n_param[i]
        self.Ju = np.hstack(indices[i], self.Ju)
        self.ru = self.rc[0]

    # Init right samples of param at indices Ju
    self.UC = [np.ones((1,1)) for p in self.params]
    for j in range(self.Mc):
      xi = np.ones((1, self.rc[j][-2]))
      for i in reversed(range(self.d_param)):
        xi = np.einsum('i...j,j...->i...', self.c_cores[j][i][:, indices[i], :], xi)
        self.UC[j] = [xi] + self.UC[j]

    # init residual
    if self.kickrank > 0:
      self.ZU = [np.ones((1,1))]  # right samples of sol at residual indices
      # residual ranks are relative to (first) param ranks
      self.rz = np.round(self.kickrank * self.rc[0] / np.max(self.rc[0]))
      self.rz = np.clip(self.rz, min=1).astype(np.int32)
      self.rz[-1] = 1
      xi = np.ones((1, self.rz[-2]))
      for i in reversed(range(self.d_param)):
        # random initial indices
        indices[i] = self.rng.integers(self.n_param[i], size=(self.rz[i]))
        # no solution yet, intialize with random data
        self.ZU = [self.rng.standard_normal((self.ru[i], self.rz[i]))] + self.ZU

      # right samples of param at residual indices
      self.ZC = [[np.ones((1,1))] for p in self.params] 
      for j in range(self.Mc):
        xi = np.ones((1, self.rc[j][-2]))
        for i in reversed(range(self.d_param)):
          xi = np.einsum('i...j,j...->i...', self.c_cores[j][i][:, indices[i], :], xi)
          self.ZC[j] = [xi] + self.ZC[j]

    # init solution variables
    self.U0 = None
    self.u = [None] * self.d

    # init matrix and rhs variables
    self.A0 = None
    self.F0 = None

    # init projection variables
    self.UAU = [None] * (self.n_param + 1)
    self.UF = [None] * (self.n_param + 1)

    # init main loop flags and counters
    self.forward_is_next = True
    self.swp = 1                # iteration counter
    self.max_dx = 0             # tracks max error over all cores
    self.tol_reached = False    # set if tolerance check passes

    
    # init profiler
    self.prof = self.profiler(['t_solve, t_project'], ['n_PDE_eval'])

  def iterate(self, nswp=1):
    for k in range(nswp):
      self.swp += 1

      # alternate forward/backward iteration
      if self.forward_is_next:
        tol_reached =  self.spatial_core_forward()
        if tol_reached:
          break

        for i in range(1, self.d_param+1):
          self.step_forward(i)
        
        self.forward_is_next = False
      else:
        for i in reversed(range(1, self.d_param+1)):
          self.step_backward(i)

        self.forward_is_next = True

  def get_tensor(self):
    """
    Get the output of the algorithm as a TT tensor.

    Returns
    -------
    output: tt.tensor
    """
    return tt.tensor.from_list(
      [self.U0.reshape((1, self.Nxu, self.ru[0]))] + self.u
      )  
  
  def get_stats(self):
    """
    Get stats collected by profiler.

    Returns
    -------
    data : dict
      Dict containing (name, value) pairs of metrics collected by profiler.
    """
    return self.prof.get_data()      