import numpy as np
import tt
import warnings
from time import perf_counter


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

    # parse parameters
    for (arg, val) in args.items():
      match arg:
        case 'nswp':
          self.nswp = val
        case 'kickrank':
          self.kickrank = val
        case 'random_init':
          self.random_init = val
        case _:
          warnings.warn('unknown argument \'' + arg + '\'', SyntaxWarning)
  

  def orthogonalize_tensor(cores, return_indices=True):
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
    cores: sequence of ndarrays
      list with updated TT cores
    v : ndarray
      non-orthogonal factor cast from first core
    r : ndarray
      array containing updated TT ranks
    (indices) : sequence of ndarrays, optional
      list with maxvol indices
    """
    d = len(cores)
    r = np.ones(d+1, dtype=np.int32)
    v = np.ones((1,1))
    if return_indices:
      indices = []

    for i in reversed(range(d)):
      # cast non-orth block from previous iteration into core
      r[i], n = cores[i].shape[:2]
      cr = cores[i].reshape((r[i]*n, -1))
      cr = cr @ v
      # QR core (note transpose)
      cr = cr.reshape((r[i], -1)).T
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
      cores[i] = cr.reshape((r[i],n,r[i+1]))

    if return_indices:
      return cores, v, r, indices
    else:
      return cores, v, r
  
  def spatial_core_forward(self):
    pass

  def step_forward(self):
    pass

  def step_backward(self):
    pass

  def __init__(self, params, assem_solve_fun, tol, **args):
    """
    Initialize ALS cross algorithm.

    Parameters
    ----------
    params:
      parameters in TT format.
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
    self.n_param = params.n[1:]   # parametric grid sizes
    self.d_param = params.d - 1   # parameter dimension
    self.rc = params.r[1:]        # TT ranks of the parameter
    self.ru = None                # TT ranks of the solution
    self.rz = None                # TT ranks of the residual

    # parametric cores of the parameter
    self.c_cores = tt.vector.to_list(self.params)[1:]

    # orthogonalize parameter TT
    self.c_cores, v, self.rc, indices = \
      self.orthogonalize_tensor(self.c_cores)
    
    # TODO
    # - init residual stuff
    # - 

    # init index set
    self.Ju = np.empty((1,0), dtype=np.int32) # u indices (for PDE eval)
    self.UC = [np.ones((1,1))]                # right samples of param at Ju
    # get random indices
    if self.random_init > 0:
      xi = np.ones((1, self.random_init))
      for i in reversed(range(self.d_param)):
        ind = self.rng.integers(self.n_param[i], size=(self.random_init))
        self.Ju = np.hstack(ind, self.Ju)
        # sample param
        xi = np.einsum('i...j,j...->i...', self.c_cores[i][:, ind, :], xi)
        self.UC = [xi] + self.UC
        self.ru[i] = self.random_init
    # OR use indices derived from param
    else:
      xi = np.ones((1, self.rc[-2]))
      # TODO original code only to i=1. Why?
      for i in reversed(range(self.d_param)):
        ind = indices[i] % self.n_param[i]
        self.Ju = np.hstack(ind, self.Ju)
        # sample param
        xi = np.einsum('i...j,j...->i...', self.c_cores[i][:, ind, :], xi)
        self.UC = [xi] + self.UC
        # TODO this was original code. Why?
        # self.UC = [np.eye(self.rc[i])] + self.UC 
      self.ru = self.rc

    # init residual
    if self.kickrank > 0:
      self.ZU = [np.ones((1,1))]  # right samples of sol at residual indices
      self.ZC = [np.ones((1,1))]  # right samples of param at residual indices
      # residual ranks are relative to param ranks
      self.rz = np.round(self.kickrank * self.rc / np.max(self.rc))
      self.rz = np.clip(self.rz, min=1).astype(np.int32)
      self.rz[-1] = 1
      xi = np.ones((1, self.rz[-2]))
      for i in reversed(range(self.d_param)):
        # random initial indices
        ind = self.rng.integers(self.n_param[i], size=(self.rz[i]))
        xi = np.einsum('i...j,j...->i...', self.c_cores[i][:, ind, :], xi)
        self.ZC = [xi] + self.ZC
        # no solution yet, intialize with random data
        self.ZU = [self.rng.standard_normal((self.ru[i], self.rz[i]))] + self.ZU

    # init solution variables
    self.U0 = None
    self.u = [None] * self.d

    # init main loop flags and counters
    self.forward_is_next = True
    self.swp = 1
    
    # init profiler
    self.prof = self.profiler(['t_solve, t_project'], ['n_PDE_eval'])

  def iterate(self, nswp=1):
    for k in range(nswp):
      self.swp += 1

      # alternate forward/backward iteration
      if self.forward_is_next:
        self.spatial_core_forward()
        for i in range(1, self.d_param+1):
          self.step_forward()
        
        self.forward_is_next = False
      else:
        for i in reversed(range(1, self.d_param+1)):
          self.step_backward()

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