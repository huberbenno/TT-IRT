import numpy as np
import tt
import warnings
from time import perf_counter
from scipy.sparse import csr_matrix, issparse

from .localcross import localcross


class als_cross_lstsq:
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

  def special_core(self):
    # store previous U
    U_prev = self.U0

    self.prof.start('t_solve')

    # input for assem_solve_fun
    if self.use_indices:
      arg = self.Ju
    else:
      # construct coeff
      arg = [[None] * self.M_A, [None] * self.M_b]
      for k in range(self.M_A):
        arg[0][k] = self.A_core[k] @ self.UA[k][0]
      for k in range(self.M_b):
        arg[1][k] = self.b_core[k] @ self.Ub[k][0]

    # solve deterministic PDE at u indices
    self.U0 = self.assem_solve_fun.solve(arg)
    if self.swp == 1:
      # TODO maybe get via attibute of assem_solve_fun
      self.Nx = self.U0[0].shape[0]

    self.U0 = np.hstack(self.U0)

    self.prof.stop('t_solve')
    self.prof.increment('n_PDE_eval', self.ru[0])

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

    # reset
    self.max_dx = 0

    # truncate U0
    self.U0, v = localcross(self.U0, self.tol/np.sqrt(self.d_param))
    self.ru[0] = self.U0.shape[1]
    # cast non-orth factor to next core
    if self.swp > 1:
      self.u[0] = v @ self.u[0].reshape(v.shape[-1], -1)

    # rank adaption
    if self.kickrank > 0: # TODO and (random_init==0 or swp>1)
      self.prof.start('t_amen')
      # compute residual at Z indices
      self.Z0 = np.zeros((self.Nx, self.rz[0]), dtype=self.ScalarType)
      for k in range(self.M_A):
        cru = self.U0 @ v @ self.ZU[k][0]
        for j in range(self.rz[0]):
          if issparse(self.A0[k][0]):
            crA = csr_matrix((self.Nx, self.Nx))
          else:
            crA = np.zeros((self.Nx, self.Nx))
          for l in range(self.rc_A[k][0]):
            crA += self.A0[k][l] * self.ZA[k][0][l,j]

          self.Z0[:,j] += np.ravel(crA @ cru[:,j])

      for k in range(self.M_b):
        self.Z0 -= self.F0[k] @ self.Zb[k][0]

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

      self.prof.stop('t_amen')

    # TODO evaluate if loop for projection are necessary
    # Project onto solution basis U0
    self.prof.start('t_project')

    Uprev = self.U0 # TODO technically dont need to copy here

    for k in range(self.M_AtA):
      k1, k2 = k, k % self.M_A
      UAtAUk_new = [None] * self.rc_AtA[k][0]
      for j in range(self.rc_AtA[k][0]):
        j1, j2 = j, j % self.rc_A[k1][0]
        UAtAUk_new[j] = np.conjugate(Uprev.T) @ self.A0[k1][j1].T @ self.A0[k2][j2] @ Uprev
        UAtAUk_new[j] = UAtAUk_new[j].reshape(-1,1)

      self.UAtAU[k][0] = np.hstack(UAtAUk_new)
    
    for k in range(self.M_Atb):
      k1, k2 = k , k % self.M_A
      for j in range(self.rc_Atb[k][0]):
        j1, j2 = j, j % self.rc_A[k1][0]
        self.UF[k][0] = (np.conjugate(Uprev.T) @ self.A0[k1][j1].T @ self.F0[k2][j2])

    self.prof.stop('t_project')

    # Project onto residual
    if self.kickrank > 0:
      self.prof.start('t_amen')
      for k in range(self.M_A):
        ZU_new = [None] * self.rc_A[k][0]
        for j in range(self.rc_A[k][0]):
          ZU_new[j] = np.conjugate(self.Z0.T) @ self.A0[k][j] @ Uprev
          ZU_new[j] = ZU_new[j].reshape(-1,1)

        self.ZU[k][0] = np.hstack(ZU_new)
      
      for k in range(self.M_b):
        self.Zb[k][0] = np.conjugate(self.Z0.T) @ self.F0[k]
      
      self.prof.stop('t_amen')

    return False

  def solve_reduced_system(self,i):
    """
    Solve (block diagonal) reduced systems.
    """
    # right hand interface projection
    crAtA = [None] * self.M_AtA
    for k in range(self.M_AtA):
      crAtA[k] = self.A_cores[k][i-1].reshape(-1, self.rc_A[k][i]) @ self.UA[k][i]
      crAtA[k] = crAtA[k].reshape(self.rc_AtA[k][i-1], -1)

    # compute RHS projection
    crF = np.zeros((self.ru[i-1], self.n_param[i-1] * self.ru[i]), dtype=self.ScalarType)
    for k in range(self.M_b):
      crb = self.b_cores[k][i-1].reshape(-1, self.rc_b[k][i]) @ self.Ub[k][i]
      crF += self.UF[k][i-1] @ crb.reshape(self.rc_b[k][i-1], -1)
    # assemble and solve blocks
    # TODO significant speedup (especially for small ranks) available
    cru = np.empty((self.ru[i-1], self.n_param[i-1] * self.ru[i]), dtype=self.ScalarType)
    for j in range(self.n_param[i-1] * self.ru[i]):
      Ai = np.zeros(self.ru[i-1] * self.ru[i-1], dtype=self.ScalarType)
      for k in range(self.M_A):
        Ai += self.UAU[k][i-1].reshape(-1, self.rc_A[k][i-1]) @ crA[k][:,j]

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
    UAU_new = np.zeros((self.ru[i], self.ru[i] * self.rc_A[k][i]), dtype=self.ScalarType)

    UAU = self.UAU[k][i-1].reshape(self.ru[i-1], -1)
    crC = np.transpose(self.A_cores[k][i-1], (0,2,1))
    cru = np.transpose(self.u[i-1], (0,2,1))
    for j in range(self.n_param[i-1]):
      v = cru[:,:,j].T
      crA = np.conjugate(v) @ UAU
      crA = crA.reshape(-1, self.rc_A[k][i-1]) @ crC[:,:,j]
      crA = crA.reshape(self.ru[i], -1).T.reshape(self.ru[i-1], -1)
      crA = v @ crA
      UAU_new += crA.reshape(-1, self.ru[i]).T

    return UAU_new

  def step_forward(self, i):
    # truncate solution core
    cru, rv = localcross(
      self.u[i-1].reshape(-1, self.ru[i]),
      self.tol/np.sqrt(self.d_param)
      )

    # Rank adaption
    if self.kickrank > 0: # and (random_init==0 or swp>1)
      # compute residual
      crz = np.zeros(((self.ru[i-1], self.n_param[i-1] * self.rz[i])), dtype=self.ScalarType)
      crz_new = np.zeros((self.rz[i-1], self.n_param[i-1] * self.rz[i]), dtype=self.ScalarType)
      for k in range(self.M_A):
        # solution interface at U indices
        U_prev = (cru @ rv @ self.ZU[k][i]).reshape(self.ru[i-1], -1)
        # right interface at Z indices
        crC = self.A_cores[k][i-1].reshape(-1, self.rc_A[k][i]) @ self.ZA[k][i]
        crC = crC.reshape(self.rc_A[k][i-1], -1)

        crA = self.UAU[k][i-1].reshape(-1, self.rc_A[k][i-1])
        for j in range(self.n_param[i-1] * self.rz[i]):
          Ai = (crA @ crC[:,j]).reshape(-1, self.ru[i-1])
          crz[:,j] += Ai @ U_prev[:,j]

        # Update residual
        crA = self.ZU[k][i-1].reshape(-1, self.rc_A[k][i-1])
        for j in range(self.n_param[i-1] * self.rz[i]):
          Ai = (crA @ crC[:,j]).reshape(self.rz[i-1], self.ru[i-1])
          crz_new[:,j] += Ai @ U_prev[:,j]

      for k in range(self.M_b):
        crb = self.b_cores[k][i-1].reshape(-1, self.rc_b[k][i]) @ self.Zb[k][i]
        crb = crb.reshape(self.rc_b[k][i-1], -1)

        crz_new -= self.Zb[k][i-1] @ crb
        crz -= self.UF[k][i-1] @ crb

      # enrich by combining solution and residual
      cru = np.hstack((cru, crz.reshape(-1, self.rz[i])))
      # QR enriched core
      cru, v = np.linalg.qr(cru)
      rv = v[:, :rv.shape[0]] @ rv

      crz_new = crz_new.reshape(-1, self.rz[i])

    # cast non orthogonal factor to the next block
    if self.swp > 1:
      self.u[i] = (rv @ self.u[i].reshape(self.ru[i], -1))
      self.u[i] = self.u[i].reshape((-1, self.n_param[i], self.ru[i+1]))

    # update solution core
    self.ru[i] = cru.shape[1]
    self.u[i-1] = cru.reshape((self.ru[i-1], self.n_param[i-1], self.ru[i]))

    # update left interface projections
    for k in range(self.M_A):
      self.UAU[k][i] = self.project_blockdiag(i,k)

    # update RHS projection interfaces
    for k in range(self.M_b):
      UFik = self.UF[k][i-1] @ self.b_cores[k][i-1].reshape(self.rc_b[k][i-1], -1)
      self.UF[k][i] = np.conjugate(cru.T) @ UFik.reshape(-1, self.rc_b[k][i])

    # Projections with the residual
    if self.kickrank > 0:
      # TODO
      # if random_init>0 and swp==1:
      #       crz_new = rng.standard_normal((rz[i-1] * ny[i-1], rz[i]))
      # orthogonalize residual core
      crz_new = np.linalg.qr(crz_new)[0]
      self.rz[i] = crz_new.shape[1]
      crz_new = crz_new.reshape((self.rz[i-1], self.n_param[i-1], self.rz[i]))

      # TODO verify just changing the slicing works
      # crC = np.transpose(self.c_cores[i-1], (0,2,1))
      # cru = np.transpose(self.u[i-1], (0,2,1))
      # crz_new = np.transpose(crz_new, (0,2,1))
      for k in range(self.M_A):
        # matrix
        self.ZU[k][i] = np.zeros((self.rz[i], self.ru[i] * self.rc_A[k][i]), dtype=self.ScalarType)
        for j in range(self.n_param[i-1]):
          crA = np.conjugate(crz_new[:,j,:].T) @ self.ZU[k][i-1].reshape(self.rz[i-1], -1)
          crA = crA.reshape(-1, self.rc_A[k][i-1]) @ self.A_cores[k][i-1][:,j,:]
          crA = crA.reshape(self.rz[i], -1).T.reshape(self.ru[i-1],-1)
          crA = self.u[i-1][:,j,:].T @ crA
          self.ZU[k][i] += crA.reshape(-1, self.rz[i]).T

        # TODO do i need these
        # self.ZA[k][i] = self.ZA[k][i-1] @ self.A_cores[k][i-1].reshape(self.rc_A[k][i-1], -1)
        # self.ZA[k][i] = self.ZA[k][i].reshape(-1, self.rc_A[k][i])
        # self.ZA[k][i] = np.conjugate(crz_new.reshape(-1, self.rz[i]).T) @ self.ZA[k][i]

      for k in range(self.M_b):
        # RHS
        self.Zb[k][i] = self.Zb[k][i-1] @ self.b_cores[k][i-1].reshape(self.rc_b[k][i-1], -1)
        self.Zb[k][i] = self.Zb[k][i].reshape(-1, self.rc_b[k][i])
        self.Zb[k][i] = np.conjugate(crz_new.reshape(-1, self.rz[i]).T) @ self.Zb[k][i]

    if self.verbose > 0:
      print(f'= swp={self.swp} core {i}>, dx={self.dx:.3e}, rank = [{self.ru[i-1]}, {self.ru[i]}]')


  def step_backward(self, i):
    # truncate solution core (note cru is not orthogonal)
    rv, cru = localcross(
      self.u[i-1].reshape(self.ru[i-1], -1),
      self.tol/np.sqrt(self.d_param)
      )

    # rank adaption
    if self.kickrank > 0:

      U_prev = (rv @ cru).reshape(self.ru[i-1], -1)
      crz = np.zeros((self.rz[i-1], self.n_param[i-1] * self.ru[i]), dtype=self.ScalarType)
      crz_new = np.zeros((self.rz[i-1], self.n_param[i-1] * self.rz[i]), dtype=self.ScalarType)
      for k in range(self.M_A):
        crA = self.ZU[k][i-1].reshape(-1, self.rc_A[k][i-1])
        # compute residual
        crC = self.A_cores[k][i-1].reshape(-1, self.rc_A[k][i]) @ self.UA[k][i]
        crC = crC.reshape(self.rc_A[k][i-1], -1)
        for j in range(self.n_param[i-1] * self.ru[i]):
          Ai = (crA @ crC[:,j]).reshape(-1, self.ru[i-1])
          crz[:,j] += Ai @ U_prev[:,j]

        # update residual
        crC = self.A_cores[k][i-1].reshape(-1, self.rc_A[k][i]) @ self.ZA[k][i]
        crC = crC.reshape(self.rc_A[k][i-1], -1)
        UZ = U_prev.reshape(-1, self.ru[i]) @ self.ZU[k][i]
        UZ = UZ.reshape(self.ru[i-1], -1)
        for j in range(self.n_param[i-1] * self.rz[i]):
          Ai = (crA @ crC[:,j]).reshape(-1, self.ru[i-1])
          crz_new[:,j] += Ai @ UZ[:,j]

      for k in range(self.M_b):
        crC = self.b_cores[k][i-1].reshape(-1, self.rc_b[k][i]) @ self.Ub[k][i]
        crC = crC.reshape(self.rc_b[k][i-1], -1)
        crz -= self.Zb[k][i-1] @ crC

        crC = self.b_cores[k][i-1].reshape(-1, self.rc_b[k][i]) @ self.Zb[k][i]
        crC = crC.reshape(self.rc_b[k][i-1], -1)
        crz_new -= self.Zb[k][i-1] @ crC

      # enrichment
      cru = np.vstack((cru, crz))

      crz_new = crz_new.reshape(-1, self.rz[i])

    # QR solution core
    # TODO why qr again if kickrank==0
    cru, v = np.linalg.qr(cru.T)
    rv = rv @ v.T[:rv.shape[1]]

    # maxvol to find local indices
    ind = tt.maxvol.maxvol(cru)
    UU = cru[ind].T
    cru = np.linalg.solve(UU, cru.T)
    rv = rv @ UU
    self.ru[i-1] = rv.shape[1]

    # cast non orthogonal factor to next core
    if i > 1:
      self.u[i-2] = self.u[i-2].reshape(-1, rv.shape[0]) @ rv
      self.u[i-2] = self.u[i-2].reshape((self.ru[i-2], self.n_param[i-2], self.ru[i-1]))
    else:
      self.U0 = self.U0 @ rv

    # update solution core
    self.u[i-1] = cru.reshape((self.ru[i-1], self.n_param[i-1], self.ru[i]))

    # update indices
    ind_quot, ind_rem = np.divmod(ind, self.ru[i])
    self.Ju = np.hstack([ind_quot.reshape(-1,1), self.Ju[ind_rem]])

    # right interface projection (sample param on U indices)
    for k in range(self.M_A):
      self.UA[k][i-1] = self.A_cores[k][i-1].reshape(-1, self.rc_A[k][i]) @ self.UA[k][i]
      self.UA[k][i-1] = self.UA[k][i-1].reshape(self.rc_A[k][i-1], -1)[:, ind]

    for k in range(self.M_b):
      self.Ub[k][i-1] = self.b_cores[k][i-1].reshape(-1, self.rc_b[k][i]) @ self.Ub[k][i]
      self.Ub[k][i-1] = self.Ub[k][i-1].reshape(self.rc_b[k][i-1], -1)[:, ind]

    if self.kickrank > 0:
      # QR and maxvol residual core
      crz_new = np.linalg.qr(crz_new.reshape(self.rz[i-1], -1).T)[0]
      self.rz[i-1] = crz_new.shape[1]
      ind = tt.maxvol.maxvol(crz_new)
      for k in range(self.M_A):
        # sample C at Z indices
        self.ZA[k][i-1] = self.A_cores[k][i-1].reshape(-1, self.rc_A[k][i]) @ self.ZA[k][i]
        self.ZA[k][i-1] = self.ZA[k][i-1].reshape(self.rc_A[k][i-1], -1)[:, ind]
        # sample U at Z indices
        self.ZU[k][i-1] = self.u[i-1].reshape(-1, self.ru[i]) @ self.ZU[k][i]
        self.ZU[k][i-1] = self.ZU[k][i-1].reshape(self.ru[i-1], -1)[:, ind]
      
      for k in range(self.M_b):
        # sample C at Z indices
        self.Zb[k][i-1] = self.b_cores[k][i-1].reshape(-1, self.rc_b[k][i]) @ self.Zb[k][i]
        self.Zb[k][i-1] = self.Zb[k][i-1].reshape(self.rc_b[k][i-1], -1)[:, ind]

    if self.verbose > 0:
      print(f'= swp={self.swp} core <{i}, dx={self.dx:.3e}, rank = [{self.ru[i-1]}, {self.ru[i]}]')


  def __init__(self, A_params, b_params, assem_solve_fun, tol, ScalarType=np.float64, **args):
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
    self.ScalarType = ScalarType

    # store parameters
    self.A_params = A_params
    self.b_params = b_params
    self.assem_solve_fun = assem_solve_fun
    self.tol = tol

    # parse optional arguments (set defaults if not given)
    self.parse_args(args)

    # init rng (set seed to get deterministic result)
    self.rng = np.random.default_rng()

    # grid sizes
    self.n_param = self.A_params[0].n[1:]     # parametric grid sizes
    self.d_param = self.A_params[0].d - 1     # parameter dimension
    # for matrix A
    self.rc_A = [p.r[1:] for p in A_params]     # TT ranks of the parameters
    self.M_A = len(self.A_params)               # number of parameter TTs
    self.Rc_A = [p.r[0] for p in A_params]      # components per param TT
    # for RHS b
    self.rc_b = [p.r[1:] for p in b_params]     # TT ranks of the parameters
    self.M_b = len(self.b_params)               # number of parameter TTs
    self.Rc_b = [p.r[0] for p in b_params]      # components per param TT
    # other 
    self.ru = None                # TT ranks of the solution
    self.rz = None                # TT ranks of the residual
    self.Nx = None

    # orthogonalize parameter TT
    self.A_cores = [None] * self.M_A
    self.A_core = [None] * self.M_A
    for k, param in enumerate(self.A_params):
      # keep maxvol indices of first param TT
      if k == 0:
        self.A_core[k], self.A_cores[k], self.rc_A[k], indices = \
          self.orthogonalize_tensor(tt.vector.to_list(param))
      else:
        self.A_core[k], self.A_cores[k], self.rc_A[k] = \
          self.orthogonalize_tensor(tt.vector.to_list(param), return_indices=False)
    
    self.b_cores = [None] * self.M_b
    self.b_core = [None] * self.M_b
    for k, param in enumerate(self.b_params):
      self.b_core[k], self.b_cores[k], self.rc_b[k] = \
        self.orthogonalize_tensor(tt.vector.to_list(param), return_indices=False)

    # init matrix and rhs variables
    # self.A0 = assem_solve_fun.matrix(self.A_core)
    self. A0 = [np.stack(A0k, axis=-1) for A0k in assem_solve_fun.matrix(self.A_core)]
    self.F0 = [np.hstack(F0k) for F0k in assem_solve_fun.rhs(self.b_core)]

    self.Nx = self.A0[0].shape[1]

    # for A_t @ A
    self.rc_AtA = []
    # self.Rc_AtA = []
    self.AtA0 = []
    self.M_AtA = self.M_A * self.M_A    
    # self.A_cores = []
    # self.A_core = []
    for i in range(self.M_A):
      for j in range(self.M_A):
        self.rc_A.append(self.rc_A[i] * self.rc_A[j]) 
        # self.Rc_A.append(Rc_A[i] * Rc_A[j])
        # self.AtA0.append(np.einsum('kis,kjt->ijst', self.A0[i], self.A0[j]).reshape((self.Nx, self.Nx, -1)))
        # c_list = []
        # for k in range(self.d_param):
        #   c_list.append(np.einsum('mis,nit->mnist', A_cores[i], A_cores[j]).reshape(self.rc_A[i][k]*self.rc_A[j][k], self.n_param[k],-1))
        # self.A_cores.append(c_list)

    # for A_t @ b
    self.rc_Atb = []
    # self.Rc_Atb = []
    self.M_Atb = self.M_A * self.M_b  
    for i in range(self.M_A):
      for j in range(self.M_b):
        self.rc_b.append(self.rc_A[i] * self.rc_b[j]) 
    #     self.Rc_b.append(Rc_A[i] * Rc_b[j])
    #     self.F0.append(np.einsum('kis,kt->ist', A0[i], F0[j]).rehsape((self.Nx, -1)))
    

    ind_rem = [None] * self.d_param
    ind_quot = [None] * self.d_param

    # init index set
    # EITHER get random indices
    if self.random_init > 0:
      self.ru = np.full(self.d_param+1, self.random_init)
      self.ru[-1] = 1
      self.Ju = np.empty((self.random_init,0), dtype=np.int32)
      # TODO original code only to i=1. Why?
      for i in reversed(range(self.d_param)):
        indices = self.rng.choice(
          self.n_param[i]*self.random_init,
          size=(self.random_init),
          replace=False, shuffle=False
          )
        ind_quot[i], ind_rem[i] = np.divmod(indices, self.random_init)
        self.Ju = np.hstack([ind_quot[i].reshape(-1,1), self.Ju[ind_rem[i]]])

    # OR use indices derived from (first) param
    else:
      self.Ju = np.empty((self.rc_A[0][-2],0), dtype=np.int32)
      for i in reversed(range(self.d_param)):
        ind_quot[i], ind_rem[i] = np.divmod(indices[i], self.rc_A[0][i+1])
        self.Ju = np.hstack([ind_quot[i].reshape(-1,1), self.Ju[ind_rem[i]]])

      self.ru = self.rc_A[0].copy()

    # Init right samples of matrix and rhs param at indices Ju
    self.UA = [[np.ones((1,1))] for p in self.A_params]
    self.UAtA = [[np.ones((1,1))] for k in range(self.M_AtA)]
    for k1 in range(self.M_A):
      if self.random_init > 0:
        xi = np.ones((1, self.random_init))
      else:
        xi = np.ones((1, self.ru[-2]))
      for i in reversed(range(self.d_param)):
        xi = np.einsum('i...j,j...->i...',
                       self.A_cores[k1][i][:, ind_quot[i], :],
                       xi[:, ind_rem[i]])
        self.UA[k1] = [xi] + self.UA[k1]

      for k2 in range(self.M_A):
        k = k1 * self.M_A + k2
        if self.random_init > 0:
          xi = np.ones((1, self.random_init))
        else:
          xi = np.ones((1, self.ru[-2]))
        for i in reversed(range(self.d_param)):
          AitAi = np.einsum('mis,nit->mnist',
                  self.A_cores[k1][i][:, ind_quot[i], :],
                  self.A_cores[k2][i][:, ind_quot[i], :]
                  ).reshape(self.rc_A[k1]*self.rc_A[k2], self.rz[i], -1)
          xi = np.einsum('i...j,j...->i...',
                        AitAi,
                        xi[:, ind_rem[i]])
          self.UAtA[k] = [xi] + self.UAtA[k]

    self.Ub = [[np.ones((1,1))] for p in self.b_params]
    self.UAtb = [[np.ones((1,1))] for k in range(self.M_Atb)]
    for k1 in range(self.M_b):
      if self.random_init > 0:
        xi = np.ones((1, self.random_init))
      else:
        xi = np.ones((1, self.ru[-2]))
      for i in reversed(range(self.d_param)):
        xi = np.einsum('i...j,j...->i...',
                       self.b_cores[k1][i][:, ind_quot[i], :],
                       xi[:, ind_rem[i]])
        self.Ub[k1] = [xi] + self.Ub[k1]

      for k2 in range(self.M_A):
        k = k1 * self.M_A + k2
        if self.random_init > 0:
          xi = np.ones((1, self.random_init))
        else:
          xi = np.ones((1, self.ru[-2]))
        for i in reversed(range(self.d_param)):
          Aitbi = np.einsum('mis,nit->mnist',
                  self.A_cores[k1][i][:, ind_quot[i], :],
                  self.b_cores[k2][i][:, ind_quot[i], :]
                  ).reshape(self.rc_A[k1]*self.rc_b[k2], self.rz[i], -1)
          xi = np.einsum('i...j,j...->i...',
                        Aitbi,
                        xi[:, ind_rem[i]])
          self.UAtb[k] = [xi] + self.UAtb[k]

    # init residual
    if self.kickrank > 0:
      # right proj ZAZ at residual indices
      self.ZU = [[np.ones((1,1))] for p in self.A_params]
      # right proj of coeffs at residual indices
      self.ZA = [[np.ones((1,1))] for p in self.A_params]
      self.Zb = [[np.ones((1,1))] for p in self.b_params]
      # residual ranks are relative to (first) param ranks
      self.rz = np.round(self.kickrank * self.rc_A[0] / np.max(self.rc_A[0]))
      self.rz = np.clip(self.rz, a_min=1, a_max=None).astype(np.int32)
      self.rz[-1] = 1
      # random initial indices
      for i in reversed(range(self.d_param)):
        indices = self.rng.choice(
          self.n_param[i]*self.rz[i+1],
          size=(self.rz[i]),
          replace=False, shuffle=False
          )
        ind_quot[i], ind_rem[i] = np.divmod(indices, self.rz[i+1])

      for k in range(self.M_AtA):
        k1, k2 = k, k % self.M_A
        xi = np.ones((1, self.rz[-2]))
        for i in reversed(range(self.d_param)):
          # no solution yet, intialize with random data
          self.ZU[k] = [self.rng.standard_normal((self.ru[i], self.rz[i]))] + self.ZU[k]
          AitAi = np.einsum('mis,nit->mnist',
                            self.A_cores[k1][i][:, ind_quot[i], :],
                            self.A_cores[k2][i][:, ind_quot[i], :]
                            ).reshape(self.rc_A[k1]*self.rc_A[k2], self.rz[i], -1)
          xi = np.einsum('i...j,j...->i...',
                          AitAi,
                          xi[:, ind_rem[i]])
          self.ZA[k] = [xi] + self.ZA[k]
      
      for k in range(self.M_b):
        xi = np.ones((1, self.rz[-2]))
        for i in reversed(range(self.d_param)):
          xi = np.einsum('i...j,j...->i...',
                          self.b_cores[k][i][:, ind_quot[i], :],
                          xi[:, ind_rem[i]])
          self.Zb[k] = [xi] + self.Zb[k]

    # init solution variables
    self.U0 = None
    self.u = [None] * self.d_param

    # init projection variables
    self.UAtAU = [[None] * (self.d_param + 1) for k in range(self.M_AtA)]
    self.UAtF = [[None] * (self.d_param + 1) for k in range(self.M_Atb)]

    # init main loop flags and counters
    self.forward_is_next = True
    self.swp = 1                # iteration counter
    self.max_dx = 0             # tracks max error over all cores
    self.tol_reached = False    # set if tolerance check passes

    # init profiler
    self.prof = self.profiler(
      ['t_solve', 't_project', 't_amen'], 
      ['n_PDE_eval']
      )

  def iterate(self, nswp=1):
    for k in range(nswp):
      # alternate forward/backward iteration
      if self.forward_is_next:
        tol_reached =  self.special_core()
        if tol_reached:
          break

        # TODO dont need to do last core when iterating back
        for i in range(1, self.d_param+1):
          # solve (block diagonal) reduced system
          self.solve_reduced_system(i)
          # compute projections, rank adaption and more
          if i < self.d_param:
            self.step_forward(i)

        if self.verbose > 0:
          print(f'= swp={self.swp} fwd finish, max_dx={self.max_dx:.3e}, max_rank = {max(self.ru)}')

        # reset
        self.max_dx = 0

        self.forward_is_next = False

      else:
        for i in reversed(range(1, self.d_param+1)):
          # solve (block diagonal) reduced system
          self.solve_reduced_system(i)
          # compute projections, rank adaption and more
          self.step_backward(i)

        self.Ju = np.empty((1, 0), dtype=np.int32)
        self.forward_is_next = True

      self.swp += 1

  def get_tensor(self):
    """
    Get the output of the algorithm as a TT tensor.

    Returns
    -------
    output: tt.tensor
    """
    return tt.tensor.from_list(
      [self.U0.reshape((1, self.Nx, self.ru[0]))] + self.u
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