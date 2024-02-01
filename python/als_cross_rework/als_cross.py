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
    use_indices: bool
      Whether assem_solve_fun takes values or indices
    verbose: integer
      Amount of info printed. Set to 0 for no output
    """
    # default values for optional parameters
    self.nswp = 5
    self.kickrank = 10 
    self.random_init = 0
    self.verbose = 1
    self.use_indices = False

    # parse parameters
    for (arg, val) in args.items():
      match arg:
        case 'nswp':
          self.nswp = val
        case 'kickrank':
          self.kickrank = val
        case 'random_init':
          self.random_init = val
        case 'use_indices':
          self.use_indices = val
        case 'verbose':
          self.verbose = val
        case _:
          warnings.warn('unknown argument \'' + arg + '\'', SyntaxWarning)
  

  def orthogonalize_tensor(self, cores, return_indices=True):
    # TODO dont need self
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
      r[i-1] = cr.shape[1]
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
    
    C0 = cores[0]
    M, N, R = C0.shape
    C0 = C0.reshape(-1, R) @ v
    C0 = C0.reshape(M,N,r[0])

    if return_indices:
      return C0, cores[1:], r, indices
    else:
      return C0, cores[1:], r
  
  def spatial_core_forward(self):
    # store previous U
    U_prev = self.U0

    self.prof.start('t_solve')

    # input for assem_solve_fun
    if self.use_indices:
      arg = self.Ju
    else:
      # construct coeff
      arg = [None] * self.Mc
      for k in range(self.Mc):
        arg[k] = self.C_core[k] @ self.UC[k][0] 

    # get matrix and rhs in first iteration
    if self.swp == 1:
      self.U0, self.A0, self.F0 = self.assem_solve_fun(arg)
      self.Nx = self.U0[0].shape[0]
      self.F0 = [np.hstack(F0k) for F0k in self.F0] 
    else:
      self.U0 = self.assem_solve_fun(arg, linear_system=False)

    self.U0 = np.hstack(self.U0)

    self.prof.stop('t_solve')
    self.prof.increment('n_PDE_eval')

    # check error
    self.dx = 1 
    if U_prev is not None:
      self.dx = np.linalg.norm(self.U0 - U_prev) / np.linalg.norm(self.U0)

    self.max_dx = max(self.max_dx, self.dx)

    if self.verbose > 0:
      print(f'= swp={self.swp} core 0, max_dx={self.max_dx:.3e}, max_rank = {max(self.ru)}')

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
      for k in range(self.Mc):
        cru = self.U0 @ v @ self.ZU[k][0]
        for j in range(self.rz[0]):
          crA = np.zeros((self.Nx, self.Nx))
          for l in range(self.rc[0]):
            crA += self.A0[k][l] * self.ZC[k][0][l,j]

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
      for k in range(self.Mc):
        ZU_new = [None]  * self.rc[0]
        for j in range(self.rc[0]):
          ZU_new[j] = np.conjugate(self.Z0.T) @ self.A0[k][j] @ Uprev
          ZU_new[j] = ZU_new[j].reshape(-1,1)
      
        self.ZU[k][0] = np.hstack(ZU_new)
        # TODO why ZC and not ZF
        self.ZC[k][0] = np.conjugate(self.Z0.T) @ self.F0[k]

    return False
  
  def solve_reduced_system(self,i):
    """
    Solve (block diagonal) reduced systems.
    """
    # right hand interface projection
    crC = [None] * self.Mc
    for k in range(self.Mc):
      crC[k] = self.c_cores[k][i-1].reshape(-1, self.rc[i]) @ self.UC[k][i]

    # compute RHS projection
    crF = np.zeros((self.ru[i-1], self.n_param[i-1] * self.ru[i]))
    for k in range(self.Mc):
      crF += self.UF[i-1][k] @ crC[k]

    # assemble and solve blocks
    # TODO significant speedup (especially for small ranks) available 
    crA = [UAUk.reshape(-1, self.rc[i-1]) for UAUk in self.UAU[i-1]]
    cru = np.empty((self.ru[i-1], self.n_param[i-1] * self.ru[i]))
    for j in range(self.n_param[i-1] * self.ru[i]):
      Ai = np.zeros(self.ru[i-1] * self.ru[i-1])
      for k in range(self.Mc):
        Ai += np.matmul(crA[k], crC[k][:,j])

      Ai = np.reshape(Ai, (self.ru[i-1], self.ru[i-1]))
      cru[:,j] = np.linalg.solve(Ai, crF[:,j])

    # check error
    self.dx = 1 
    if self.u[i-1] is not None:
      self.dx = np.linalg.norm(cru.flatten() - self.u[i-1].flatten()) / np.linalg.norm(cru)

    self.max_dx = max(self.max_dx, self.dx) 

    # update solution
    self.u[i-1] = cru.reshape((self.ru[i-1], self.n_param[i-1],  self.ru[i]))

  def project_blockdiag(self, i, k=0):
    """
    Update right interface projections.
    """
    UAU_new = np.zeros((self.ru[i], self.ru[i] * self.rc[k][i]))

    UAU = self.UAU[i-1][k].reshape(self.ru[i-1], -1)
    crC = np.transpose(self.c_cores[i-1][k], (0,2,1))
    cru = np.transpose(self.u[i-1], (0,2,1))
    for j in range(self.n_param[i-1]):
      v = np.conjugate(cru[:,:,j].T)
      crA = v @ UAU
      crA = crA.reshape(-1, self.rc[k][i-1]) @ crC[:,:,j]
      crA = crA.reshape(self.ru[i], -1).T.reshape(self.ru[i-1], -1)
      crA = v @ crA
      UAU_new += crA.reshape(-1, self.ru[i]).T

    return UAU_new
  
  def step_forward(self, i):
    # solve (block diagonal) reduced system
    self.solve_reduced_system(i)

    # truncate solution core
    cru, rv = localcross(
      self.u[i-1].reshape(-1, self.ru[i]), 
      self.tol/np.sqrt(self.d_param)
      )

    # Rank adaption
    if self.kickrank > 0: # and (random_init==0 or swp>1)
      # compute residual
      
      crz = np.zeros(((self.ru[i-1], self.n_param[i-1] * self.rz[i])))
      for k in range(self.Mc):
        # solution interface at U indices
        U_prev = (cru @ rv @ self.ZU[k][i]).reshape(self.ru[i-1], -1)
        # right interface at Z indices
        crC = self.c_cores[i-1].reshape(-1, self.rc[k][i]) @ self.ZC[k][i]
        crC = crC.reshape(self.rc[k][i-1], -1)
        
        crA = self.UAU[i-1].reshape(-1, self.rc[k][i-1])
        for j in range(self.n_param[i-1] * self.rz[i]):
          Ai = (crA @ crC[:,j]).reshape(-1, self.ru[i-1])
          crz[:,j] += Ai @ U_prev[:,j]

        crz -= self.UF[i-1][k] @ crC

      # enrich by combining solution and residual
      cru = np.hstack((cru, crz.reshape(-1, self.rz[i])))
      # QR enriched core
      cru, v = np.linalg.qr(cru)
      rv = v[:, :rv.shape[0]] @ rv

      # Update residual
      crz = np.zeros((self.rz[i-1], self.n_param[i-1] * self.rz[i]))
      for k in range(self.Mc):
        crA = self.ZU[k][i-1].reshape(-1, self.rc[k][i-1])
        for j in range(self.n_param[i-1] * self.rz[i]):
          Ai = (crA @ crC[:,j]).reshape(self.rz[i-1], self.ru[i-1])
          crz[:,j] += Ai @ U_prev[:,j]

        crz -= self.ZC[k][i-1] @ crC
      
      crz = crz.reshape(-1, self.rz[i])

    # cast non orthogonal factor to the next block
    if self.swp > 1:
      self.u[i] = (rv @ self.u[i].reshape(self.ru[i], -1))
      self.u[i].reshape((self.ru[i], self.n_param[i], self.ru[i+1]))

    # update solution core
    self.ru[i] = cru.shape[1]
    self.u[i-1] = cru.reshape((self.ru[i-1], self.n_param[i-1], self.ru[i]))

    # update left interface projections
    self.UAU[i] = [self.project_blockdiag(i,k) for k in range(self.Mc)]

    # update RHS projection interfaces
    for k in range(self.Mc):
      UFik = self.UF[i-1][k] @ self.c_cores[k][i-1].reshape(self.rc[k][i-1], -1)
      self.UF[i][k] = np.conjugate(cru.T) @ UFik.reshape(-1, self.rc[k][i])

    # Projections with the residual
    if self.kickrank > 0:
      # TODO
      # if random_init>0 and swp==1:
      #       crz = rng.standard_normal((rz[i-1] * ny[i-1], rz[i]))
      # orthogonalize residual core
      crz = np.linalg.qr(crz)[0]
      self.rz[i] = crz.shape[1]
      crz = crz.reshape((self.rz[i-1], self.n_param[i-1], self.rz[i]))
      
      # TODO verify just changing the slicing works
      # crC = np.transpose(self.c_cores[i-1], (0,2,1))
      # cru = np.transpose(self.u[i-1], (0,2,1))
      # crz = np.transpose(crz, (0,2,1))
      for k in range(self.Mc):
        # matrix
        self.ZU[k][i] = np.zeros(self.rz[i], self.ru[i] * self.rc[k][i])
        for j in range(self.n_param[i-1]):
          crA = np.conjugate(crz[:,j,:].T) @ self.ZU[k][i-1].reshape(self.rz[i-1], -1)
          crA = crA.reshape(-1, self.rc[k][i-1]) @ self.c_cores[k][i-1][:,j,:]
          crA = crA.reshape(self.rz[i], -1).T.reshape(self.ru[i-1])
          crA = np.conjugate(self.u[i-1][:,j,:].T) @ crA
          self.ZU[k][i] += crA.reshape(-1, self.rz[i]).T 

        # RHS
        self.ZC[k][i] = self.ZC[k][i-1] @ self.c_cores[k][i-1].reshape(self.rc[k][i-1], -1)
        self.ZC[k][i] = np.conjugate(crz.T) @ self.ZC[k][i].reshape(-1, self.rc[k][i])


    if self.verbose > 0:
      print(f'= swp={self.swp} core {i}>, dx={self.dx:.3e}, rank = [{self.ru[i-1]}, {self.ru[i]}]')


  def step_backward(self, i):
    # solve (block diagonal) reduced system
    self.solve_reduced_system(i)

    # truncate solution core (note cru is not orthogonal)
    rv, cru = localcross(
      self.u[i-1].reshape(-1, self.ru[i]), 
      self.tol/np.sqrt(self.d_param)
      )

    # rank adaption
    if self.kickrank > 0:
      # enrichment
      U_prev = (rv @ cru).reshape(self.ru[i-1], -1)
      crz = np.zeros((self.rz[i-1], self.n_param[i-1] * self.ru[i]))
      for k in range(self.Mc):
        crC = self.c_cores[k][i-1].reshape(-1, self.rc[k][i]) @ self.UC[i]
        crC = crC.reshape(self.rc[k][i-1], -1)
        crA = self.ZU[k][i-1].reshape(-1, self.rc[k][i-1])
        for j in range(self.n_param[i-1] * self.ru[i]):
          Ai = (crA @ crC[:,j]).reshape(-1, self.ru[i-1])
          crz[:,j] += Ai @ U_prev[:,j]

        crz -= self.ZC[i-1][k] @ crC

      cru = np.vstack((cru, crz))

      # Residual
      crz = np.zeros((self.rz[i-1], self.n_param[i-1] * self.rz[i]))
      for k in range(self.Mc):
        crC = self.c_cores[k][i-1].reshape(-1, self.rc[k][i]) @ self.ZC[k][i]
        crC = crC.reshape(self.rc[k][i-1], -1)
        U_prev = U_prev.reshape(-1, self.ru[i]) @ self.ZU[k][i]
        U_prev = U_prev.reshape(self.ru[i-1], 1)
        for j in range(self.n_param[i-1] * self.rz[i]):
          Ai = (crA @ crC[:,j]).reshape(-1, self.ru[i-1])
          crz[:,j] += Ai @ U_prev[:,j]

        crz -= self.ZC[k][i-1] @ crC

      crz = crz.reshape(-1, self.rz[i])

    # QR solution core
    # TODO why qr again if kickrank==0
    cru, v = np.linalg.qr(cru.T)
    rv = rv @ v.T[:rv.shape[1]]
    self.ru[i-1] = rv.shape[1]

    # maxvol to find local indices
    ind = tt.maxvol.maxvol(cru)
    cru = np.linalg.solve(cru[ind].T, cru.T)

    # cast non orthogonal factor to next core
    if i > 1:
      self.u[i-2] = self.u[i-2].reshape(-1, rv.shape[0]) @ rv
      self.u[i-2].reshape((self.ru[i-2], self.n_param[i-2], self.ru[i-1]))
    else:
      self.U0 = self.U0 @ rv
    
    # update solution core
    self.u[i-1] = cru.reshape((self.ru[i-1], self.n_param[i-1], self.ru[i]))

    # update indices
    ind = ind % self.n_param[i-1]
    self.Ju = np.hstack(ind.reshape(-1,1), self.Ju)

    # right interface projection (sample param on U indices)
    for k in range(self.Mc):
      self.UC[i-1][k] = self.c_cores[k][i-1].reshape(-1, self.rc[k][i]) @ self.UC[i][k]
      self.UC[i-1][k] = self.UC[i-1][k].reshape(self.rc[k][i-1], -1)[:, ind]

    if self.kickrank > 0:
      # QR and maxvol residual core
      crz = np.linalg.qr(crz.reshape(self.rz[i-1]).T)[0]
      self.rz[i-1] = crz.shape[1]
      ind = tt.maxvol.maxvol(crz)
      for k in range(self.Mc):
        # sample C at Z indices
        self.ZC[k][i-1][k] = self.cores[k][i-1].reshape(-1, self.rc[k][i]) @ self.ZC[k][i]
        self.ZC[k][i-1][k] = self.ZC[k][i-1].reshape(self.rc[k][i-1], -1)[:, ind]
        # sample U at Z indices
        self.ZU[k][i-1] = self.u[i-1].reshape(-1, self.ru[i]) @ self.ZU[k][i]
        self.ZU[k][i-1] = self.ZU[k][i-1].reshape(self.ru[i-1], -1)[:, ind]

    if self.verbose > 0:
      print(f'= swp={self.swp} core <{i}, dx={self.dx:.3e}, rank = [{self.ru[i-1]}, {self.ru[i]}]')


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
    self.Mc = len(self.params)              # number of parameter TTs
    self.Rc = [p.r[0] for p in params]      # components per param TT
    self.ru = None                # TT ranks of the solution
    self.rz = None                # TT ranks of the residual
    self.Nx = None

    # orthogonalize parameter TT
    self.c_cores = [None] * len(self.params)
    self.C_core = [None] * len(self.params)
    for k, param in enumerate(self.params):
      # keep maxvol indices of first param TT 
      if k == 0:
        self.C_core[k], self.c_cores[k], self.rc[k], indices = \
          self.orthogonalize_tensor(tt.vector.to_list(param))
      else:
        self.C_core[k], self.c_cores[k], self.rc[k] = \
          self.orthogonalize_tensor(tt.vector.to_list(param), return_indices=False)
    
    ind_rem = [None] * self.d_param
    ind_quot = [None] * self.d_param      

    # init index set    
    # EITHER get random indices
    if self.random_init > 0:
      self.Ju = np.empty((self.random_init,0), dtype=np.int32)
      xi = np.ones((1, self.random_init))
      # TODO original code only to i=1. Why?
      for i in reversed(range(self.d_param)):
        indices = self.rng.choice(
          self.n_param[i]*self.rc[0][i+1], 
          size=(self.random_init),
          replace=False, shuffle=False
          )
        ind_quot[i], ind_rem[i] = np.divmod(indices, self.rc[0][i+1])
        self.rng.integers(self.n_param[i], size=(self.random_init))
        self.Ju = np.hstack([indices[i], self.Ju])
        self.ru[i] = self.random_init

    # OR use indices derived from (first) param
    else:
      self.Ju = np.empty((self.rc[0][-2],0), dtype=np.int32) 
      for i in reversed(range(self.d_param)):
        ind_quot[i], ind_rem[i] = np.divmod(indices[i], self.rc[0][i+1])
        self.Ju = np.hstack([ind_quot[i].reshape(-1,1), self.Ju[ind_rem[i]]])

      self.ru = self.rc[0]

    # Init right samples of param at indices Ju
    self.UC = [[np.ones((1,1))] for p in self.params]
    for k in range(self.Mc):
      xi = np.ones((1, self.rc[k][-2]))
      for i in reversed(range(self.d_param)):    
        xi = np.einsum('i...j,j...->i...', 
                       self.c_cores[k][i][:, ind_quot[i], :], 
                       xi[:, ind_rem[i]])
        self.UC[k] = [xi] + self.UC[k]

    # init residual
    if self.kickrank > 0:
      self.ZU = [np.ones((1,1))]  # right samples of sol at residual indices
      # residual ranks are relative to (first) param ranks
      self.rz = np.round(self.kickrank * self.rc[0] / np.max(self.rc[0]))
      self.rz = np.clip(self.rz, a_min=1, a_max=None).astype(np.int32)
      self.rz[-1] = 1
      xi = np.ones((1, self.rz[-2]))
      for i in reversed(range(self.d_param)):
        # random initial indices
        indices = self.rng.choice(
          self.n_param[i]*self.rz[i+1], 
          size=(self.rz[i]),
          replace=False, shuffle=False
          )
        ind_quot[i], ind_rem[i] = np.divmod(indices, self.rz[i+1])
        # no solution yet, intialize with random data
        self.ZU = [self.rng.standard_normal((self.ru[i], self.rz[i]))] + self.ZU

      # right samples of param at residual indices
      self.ZC = [[np.ones((1,1))] for p in self.params] 
      for k in range(self.Mc):
        xi = np.ones((1, self.rz[-2]))
        for i in reversed(range(self.d_param)):
          xi = np.einsum('i...j,j...->i...', 
                         self.c_cores[k][i][:, ind_quot[i], :], 
                         xi[:, ind_rem[i]])
          self.ZC[k] = [xi] + self.ZC[k]

    # init solution variables
    self.U0 = None
    self.u = [None] * self.d_param

    # init matrix and rhs variables
    self.A0 = None
    self.F0 = None

    # init projection variables
    self.UAU = [None] * (self.d_param + 1)
    self.UF = [None] * (self.d_param + 1)

    # init main loop flags and counters
    self.forward_is_next = True
    self.swp = 1                # iteration counter
    self.max_dx = 0             # tracks max error over all cores
    self.tol_reached = False    # set if tolerance check passes

    # init profiler
    self.prof = self.profiler(['t_solve', 't_project'], ['n_PDE_eval'])

  def iterate(self, nswp=1):
    for k in range(nswp):
      self.swp += 1

      # alternate forward/backward iteration
      if self.forward_is_next:
        tol_reached =  self.spatial_core_forward()
        if tol_reached:
          break
        
        # TODO dont need to do last core when iterating back
        for i in range(1, self.d_param+1):
          self.step_forward(i)
        
        self.forward_is_next = False
      else:
        for i in reversed(range(1, self.d_param+1)):
          self.step_backward(i)

        self.Ju = np.empty((1, 0), dtype=np.int32)
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