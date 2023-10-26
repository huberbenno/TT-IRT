import warnings
import tt
import numpy
import time
from localcross import localcross
from utils import solve_blockdiag, project_blockdiag


def als_cross_parametric(coeff, assem_solve_fun, tol , **varargin):
  """
  TT ALS-Cross algorithm.

  :param coeff: (d+1)-dimensional block TT format storing coefficients, with the
  first TT rank Mc corresponding to different coefficients, and the 
  entire first TT block corresponding to the deterministic part with
  Nxc degrees of freedom in the deterministic variable. The other 
  TT blocks correspond to the d (stochastic) parameters.

  :param assem_solve_fun: a handle to the function which should take the
    coefficients Ci of size ``[Mc, Nxc, r]`` and return lists ``U,A,F``, 
    respectively solutions, matrices and RHS at the given Ci. 
    Here r is the TT rank of the solution in the first TT block.
    ``U,A,F`` must be cell arrays of size 1 x r, each snapshot ``U[i]`` must be
    a column vector of size Nxu x 1, each snapshot ``F[i]`` must be a 
    column vector of size Nxa x 1, and each matrix ``A[i]`` must be a
    Nxa x Nxa matrix.

    Alternatively, if ``use_indices`` is enabled, assem_solve_fun 
    should take indices of parameters where
    the systems must be solved, in the form of r x d integer matrix,
    with elements in the k-th column ranging from 1 to n_k, the number
    of grid points in the k-th parameter. The output format is the same.

    !!! In both cases, A and F should depend near linearly on coeff !!!

  :param tol: cross truncation and stopping tolerance

  :param **varargin: Optional parameters:
    - ``Pua``: a matrix to project spatial block of solution to spat. block of matrix
      For good performance, Pua should be a full rank projector, with
      size(Pua)==[Nxa,Nxu] with Nxu>=Nxa.
      Default empty Pua assumes Nxu==Nxa.
    - ``nswp``: max number of iterations (default 5)
    - ``kickrank``: max TT rank of the residual/enrichment (default 10)
    - ``random_init``: if greater than 0, take random_init random indices at
      start; if 0 (default), take maxvol indices of coeff
    - ``use_indices``: selects the type of input for assem_solve_fun:
      ``False`` (default) assumes that the function takes values of
      the coefficients,
      ``True`` assumes that the function takes indices of parameters

  :returns:
    u: solution in TT format
    time_extern: list [time solve, time project] of cpu times
    funevals: total number of deterministic solves

  """

  # default values for optional parameters
  nswp = 5 # number of iterations
  kickrank = 10 # The base for enrichment rank. The actual ranks are scaled to the coeff. ranks
  Pua = []  # A matrix that maps spatial DOFS of the solution to the spatial DOFs of the matrix/rhs
  random_init = 0 # If >0, start with random indices, instead of the coefficient
  use_indices = False # whether assem_solve_fun takes indices (or values)

  # parse parameters
  for (arg, val) in varargin.items():
    match arg:
      case 'nswp':
        nswp = val
      case 'kickrank':
        kickrank = val
      case 'pua':
        Pua = val
      case 'random_init':
        random_init = val
      case 'use_indices':
        use_indices = val
      case _:
        warnings.warn('unknown argument \'' + arg + '\'', SyntaxWarning)

  # rng
  rng = numpy.random.default_rng(0)
  
  # all grid sizes
  Nxc = coeff.n[0]  # spatial grid size
  ny = coeff.n[1:]  # parametric grid sizes
  d = coeff.d - 1   # d is parameter dimension only
  Mc = coeff.r[0]   # number of coefficient components
  rc = coeff.r[1:]  
  ru = numpy.copy(rc)          # these will be TT ranks of the solution
  coeff = tt.vector.to_list(coeff)
  C0 = coeff[0]
  coeff = coeff[1:]

  # Prepare storage for reduction/sampling matrices
  if nswp > 1:
    UAU = [None] * (d+1)  # left Galerkin reductions, matrix
    UF = [None] * (d+1)   # left Galerkin reductions, RHS

  UC = [None] * (d+1) # right cross samples of C on U-indices
  UC[-1] = numpy.ones((1,1))

  # Prepare storage for the residual
  if kickrank > 0:
    ZU = [None] * (d+1) # ZAU from the left, ZU from the right
    ZU[-1] = numpy.ones((1,1))
    ZC = ZU.copy()
    rz = numpy.round(kickrank * rc / numpy.max(rc)).astype(int)
    rz[rz<1] = 1
    rz[-1] = 1

  xi = numpy.ones((1, random_init))
  if use_indices: # Initialise global indices if the user function works with them
    Ju = numpy.empty((rc[-1],0), dtype=numpy.int32)

  # First, orthogonalize the coefficient.
  # We can derive its optimal indices (as an initial guess), or use random
  # ones
  v = numpy.ones((1,1))
  for i in reversed(range(d)):
    crc = numpy.reshape(coeff[i], (rc[i]*ny[i], -1))
    crc = numpy.matmul(crc, numpy.transpose(v))
    crc = numpy.reshape(crc, (rc[i], ny[i]*rc[i+1]))
    crc = numpy.transpose(crc)
    crc,v = numpy.linalg.qr(crc)
    rc[i] = numpy.shape(crc)[1]
    ind = tt.maxvol.maxvol(crc) 
    crc = numpy.transpose(crc)
    CC = crc[:, ind]
    crc = numpy.linalg.solve(CC, crc)
    v = numpy.matmul(numpy.transpose(CC), v)
    coeff[i] = numpy.reshape(crc, (rc[i], ny[i], rc[i+1]))

    if use_indices: # TODO verify correctness
      Ju = numpy.hstack(
        (numpy.repeat(numpy.arange(ny[i]), rc[i+1]).reshape(-1,1),
        numpy.tile(Ju, [ny[i],1]))
      )
      Ju = Ju[ind, :]
      
    if random_init > 0 and i > 0:
      # Random sample coeff from the right
      ind = rng.integers(ny[i], size=(random_init))
      xi = numpy.einsum('i...j,j...->i...', coeff[i][:, ind, :], xi)
      UC[i] = xi
      ru[i] = random_init
    else:
      UC[i] = numpy.eye(rc[i])
      ru[i] = rc[i]

    if kickrank > 0:
      # Initialize the residual
      crz = rng.standard_normal((ny[i] * rz[i+1], rz[i]))
      crz = numpy.linalg.qr(crz)[0]
      rz[i] = numpy.shape(crz)[1]
      ind = tt.maxvol.maxvol(crz)
      # Sample the coefficient and solution at Z-indices
      ZC[i] = numpy.matmul(numpy.reshape(coeff[i], (rc[i]*ny[i], rc[i+1])), ZC[i+1])
      ZC[i] = numpy.reshape(ZC[i], (rc[i], ny[i] * rz[i+1]))
      ZC[i] = ZC[i][:, ind]
      if ru[i] > rc[i]: # TODO is this a good idea
        ZU[i] = numpy.vstack((ZC[i], numpy.zeros((ru[i]-rc[i], rz[i]))))
      else:
        ZU[i] = ZC[i][:ru[i]]
  
  # Init empty solution storage
  u = [None] * d
  U0 = []

  # The coefficient+rhs at sampled indices
  C0 = numpy.reshape(C0, (Mc*Nxc,-1)) # size Mc*Nxc, rc1
  C0 = numpy.matmul(C0, numpy.transpose(v))
  # This is the spatial block of the coefficient, in the representation when
  # all parametric blocks contain identity matrices.
  # The coeff will not change anymore
  C0 = numpy.reshape(C0, (Mc, Nxc, rc[0]))

  # Initialise cost profilers
  time_solve = 0
  time_project = 0
  funevals = 0 # init number of det. solves

  # Start iterating
  i = 0
  max_dx = 0
  dir = 1
  swp = 1
  while swp <= nswp:
    if i == 0:
      ##### Work on spatial block ###################################
      # Previous guess (if we have one)
      Uprev = U0
      # solve deterministic problems at the U-indices
      if use_indices:
        Ci = Ju
      else:
        # Construct the coeff there
        Ci = numpy.matmul(numpy.reshape(C0, (Mc*Nxc,-1)), UC[0]) # size Mc*Nxc, rc[0]
        Ci = numpy.reshape(Ci, (Mc, Nxc, ru[0]))

      t1__uc = time.perf_counter()
      

      if kickrank > 0:
        U0, V0, UAUs, UFs, ZU_new, ZC_new = assem_solve_fun(Ci,ru[0],rc[0],rz[0],ZU[0],ZC[0],rankAdaption=True, firstIter=(swp==1))
        # Nxa = numpy.shape(A0s[0])[0]
        # F0 = numpy.hstack(F0)
        # In the first sweep, Ci==C0, and we need the corresponding
        # matrices (A0s) and RHS (F0), since we'll use them in
        # subsequent iterations ...
      else:
        # ... where it's enough to compute the solution only
        U0, V0, UAUs, UFs = assem_solve_fun(Ci, ru[0],rc[0], rankAdaption=False, firstIter=(swp==1))

      time_solve += time.perf_counter() - t1__uc
      del Ci # memory saving
      funevals += ru[0]
      # U0 = numpy.hstack(U0)
      # Nxu = numpy.shape(U0)[0] # this, again, can differ from Nxa or Nxc
      Nxu = assem_solve_fun.sol_size
      # if Nxu != Nxa and len(Pua) == 0:
      #   raise RuntimeError('Numbers of spatial DOFs in u and A differ, and no transformation matrix is given. Unable to reduce model')
      
      # check the error
      # TODO fix
      # if len(Uprev) > 0:
      #   dx = numpy.linalg.norm(U0 - Uprev) / numpy.linalg.norm(U0)
      # else:
      #   dx = 1
      dx=1
      max_dx = max(max_dx, dx)

      print(f'=als_cross_parametric= 0 swp={swp}, max_dx={max_dx:.3e}, max_rank = {max(ru)}')

      # Unusual ALS exit condition: after solving for the spatial block,
      # to have the non-orth center here.
      if max_dx < tol:
        break

      max_dx = 0

      # cast non orth factor to next block
      if swp > 1:
        u[0] = numpy.reshape(u[0], (ru[0], ny[0]*ru[1]))
        u[0] = numpy.matmul(V0, u[0])

      ru[0] = numpy.shape(U0)[1]

      # if kickrank > 0:
      #   # Compute residual
      #   # Compute A * U at Z indices
      #   cru = numpy.linalg.multi_dot((U0, v, ZU[0]))
      #   if Nxa != Nxu:
      #     cru = numpy.matmul(Pua, cru)

      #   Z0 = numpy.empty((Nxa, rz[0]))
      #   for j in range(rz[0]):
      #     crA = A0s[0] * ZC[0][0,j]
      #     for k in range(1,rc[0]):
      #       crA += A0s[k] * ZC[0][k,j]

      #     Z0[:,j] = crA @ cru[:,j]
        
      #   Z0 -= numpy.matmul(F0, ZC[0])
      #   Z0 = numpy.linalg.qr(Z0)[0]
      #   rz[0] = numpy.shape(Z0)[1]
      #   if Nxa != Nxu:
      #     cru = numpy.hstack((U0, numpy.matmul(numpy.conjugate(numpy.transpose(Pua)), Z0)))
      #   else:
      #     cru = numpy.hstack((U0, Z0))

      #   # QR U0
      #   U0, v = numpy.linalg.qr(cru)
      #   v = v[:,:ru[0]]
      #   if swp > 1:
      #     u[0] = numpy.reshape(u[0], (ru[0], ny[0]*ru[1]))
      #     u[0] = numpy.matmul(v, u[0])

      #   ru[0] = numpy.shape(U0)[1]

      # Project the model onto the solution basis U0
      # UAU is very large here
      # Need to run some loops to save mem
      t1__uc = time.perf_counter()
      # UAU_new = [None] * rc[0]
      Uprev = U0
      # if Nxa != Nxu:
      #   Uprev = numpy.matmul(Pua, U0)
      
      # for j in range(rc[0]):
      #   UAU_new[j] = numpy.conjugate(numpy.transpose(Uprev)) @ A0s[j] @ Uprev
      #   UAU_new[j] = numpy.reshape(UAU_new[j], (-1,1))
      
      if nswp == 1:
        # we don't need to save UAU projections for all blocks, if we
        # will never iterate back
        # UAU = numpy.hstack(UAU_new)
        UAU = UAUs.reshape(rc[0],-1).T
        UF = UFs
      else:
        # UAU[0] = numpy.hstack(UAU_new)
        UAU[0] = UAUs.reshape(rc[0],-1).T
        UF[0] = UFs

      time_project += time.perf_counter() - t1__uc
      # del UAU_new

      # Project onto residual
      if kickrank > 0:
        # Project onto residual basis
        # ZU_new = [None] * rc[0]
        # for j in range(rc[0]):
        #   ZU_new[j] = numpy.conjugate(numpy.transpose(Z0)) @ A0s[j] @ Uprev
        #   ZU_new[j] = numpy.reshape(ZU_new[j], (-1,1))
        
        # ZU[0] = numpy.hstack(ZU_new)
        # ZC[0] = numpy.matmul(numpy.conjugate(numpy.transpose(Z0)), F0)
        ZU[0] = ZU_new.reshape(rc[0], -1).T
        ZC[0] = ZC_new
        rz[0] = ZC_new.shape[0]
        del ZU_new
      
      # Save some memory if we only have 1 iteration
      if nswp == 1:
        del UAUs
        # del F0

    else:  ##### End i == 0, Loop for reduced system ################
      # Solve block-diagonal reduced system
      crC = numpy.reshape(coeff[i-1], (rc[i-1]*ny[i-1], rc[i]))
      crC = numpy.matmul(crC, UC[i]) # now these are indices from the right, and Galerkin from the left
      crC = numpy.reshape(crC, (rc[i-1], ny[i-1]*ru[i]))
      if nswp == 1:
        UAUi = UAU # cell-free local reduction from the left
        UFi = UF
      else:
        UAUi = UAU[i-1]
        UFi = UF[i-1]
      
      crF = numpy.matmul(UFi, crC)

      cru = solve_blockdiag(UAUi,crC,crF,ru[i-1],ru[i],rc[i-1],ny[i-1])

      # error check
      if u[i-1] is None:
        dx = 1
      else:
        dx = numpy.linalg.norm(cru - numpy.reshape(u[i-1], (ru[i-1]*ny[i-1], ru[i]))) / numpy.linalg.norm(cru)
      max_dx = max(max_dx, dx)
      u[i-1] = numpy.reshape(cru, (ru[i-1], ny[i-1], ru[i])) 
      # if we're in the d-th block (i == d-1), we're done

      if i < d and dir > 0: ##### left-right sweep ################
        # Truncate cru with cross
        cru, rv = localcross(cru, tol/numpy.sqrt(d))
        if kickrank > 0:
          # Update the residual and enrichment
          crC = numpy.reshape(coeff[i-1],(rc[i-1]*ny[i-1], rc[i]))
          crC = numpy.matmul(crC, ZC[i]) # now these are indices from the right, and Galerkin from the left
          crC = numpy.reshape(crC, (rc[i-1], ny[i-1]*rz[i]))
          Uprev = numpy.linalg.multi_dot((cru, rv, ZU[i]))
          Uprev = numpy.reshape(Uprev, (ru[i-1],ny[i-1]*rz[i]))
          # first enrichment
          crA = numpy.reshape(UAUi, (ru[i-1]*ru[i-1], rc[i-1]))
          crz = numpy.empty((ru[i-1], ny[i-1]*rz[i]))
          for j in range(ny[i-1]*rz[i]):
            Ai = numpy.matmul(crA, crC[:,j])
            Ai = numpy.reshape(Ai, (ru[i-1], ru[i-1]))
            crz[:,j] = numpy.matmul(Ai, Uprev[:,j])

          crz -= numpy.matmul(UFi, crC)
          crz = numpy.reshape(crz, (ru[i-1]*ny[i-1],rz[i]))
          cru = numpy.hstack((cru, crz))
          # QR u
          cru, v = numpy.linalg.qr(cru)
          v = v[:, :numpy.shape(rv)[0]]
          rv = numpy.matmul(v, rv)
          # Now the residual itself
          crA = numpy.reshape(ZU[i-1], (rz[i-1]*ru[i-1],rc[i-1]))
          crz = numpy.empty((rz[i-1],ny[i-1]*rz[i]))
          for j in range(len(crz)):
            Ai = numpy.matmul(crA, crC[:,j])
            Ai = numpy.reshape(Ai, (rz[i-1],ru[i-1]))
            crz[:,j] = numpy.matmul(Ai, Uprev[:,j])

          crz -= numpy.matmul(ZC[i-1], crC)
          crz = numpy.reshape(crz, (rz[i-1]*ny[i-1], rz[i]))
        
        # cast the non-orth factor to the next block
        if swp > 1:
          u[i] = numpy.reshape(u[i], (ru[i], ny[i]*ru[i+1]))
          u[i] = numpy.matmul(rv, u[i])
          u[i] = numpy.reshape(u[i], (-1, ny[i], ru[i+1]))
        
        ru[i] = numpy.shape(cru)[1]
        u[i-1] = numpy.reshape(cru, (ru[i-1], ny[i-1], ru[i]))

        # projection from the left -- Galerkin
        # with matrix
        crC = numpy.reshape(coeff[i-1], (rc[i-1],ny[i-1],rc[i]))
        cru = numpy.reshape(cru, (ru[i-1], ny[i-1], ru[i]))

        UAU_new = project_blockdiag(UAUi,crC,cru,ru[i-1],ru[i],rc[i-1],rc[i],ny[i-1])
        
        # Save some mem
        if nswp == 1:
          UAU = UAU_new
        else:
          UAU[i] = UAU_new
        
        del UAU_new
        del UAUi

        # with RHS
        crC = numpy.reshape(coeff[i-1], (rc[i-1], ny[i-1]*rc[i]))
        UFi = numpy.matmul(UFi, crC)
        UFi = numpy.reshape(UFi, (ru[i-1]*ny[i-1], rc[i]))
        cru = numpy.reshape(cru, (ru[i-1]*ny[i-1], ru[i]))
        UFi = numpy.matmul(numpy.conjugate(numpy.transpose(cru)), UFi)
        
        if nswp == 1:
          UF = UFi
        else:
          UF[i] = UFi
        
        del UFi

        # Projections with Z
        if kickrank > 0:
          crz = numpy.linalg.qr(crz)[0]
          rz[i] = numpy.shape(crz)[1]
          # with matrix
          crC = numpy.reshape(crC, (rc[i-1], ny[i-1], rc[i]))
          cru = numpy.reshape(cru, (ru[i-1], ny[i-1], ru[i]))
          crz = numpy.reshape(crz, (rz[i-1], ny[i-1], rz[i]))
          ZU[i] = numpy.zeros((rz[i], ru[i]*rc[i]))
          ZU[i-1] = numpy.reshape(ZU[i-1], (rz[i-1], ru[i-1]*rc[i-1]))
          crC = numpy.transpose(crC, (0,2,1))
          cru = numpy.transpose(cru, (0,2,1))
          crz = numpy.transpose(crz, (0,2,1))
          for j in range(ny[i-1]):
            v = crz[:,:,j]
            crA = numpy.matmul(numpy.conjugate(numpy.transpose(v)), ZU[i-1])
            crA = numpy.reshape(crA, (rz[i]*ru[i-1], rc[i-1]))
            crA = numpy.matmul(crA, crC[:,:,j])
            crA = numpy.reshape(crA, (rz[i], ru[i-1]*rc[i]))
            crA = numpy.transpose(crA)
            crA = numpy.reshape(crA, (ru[i-1], rz[i]*rc[i]))
            v = cru[:,:,j]
            crA = numpy.matmul(numpy.conjugate(numpy.transpose(v)), crA)
            crA = numpy.reshape(crA, (ru[i]*rc[i], rz[i]))
            crA = numpy.transpose(crA)
            ZU[i] += crA

          crz = numpy.transpose(crz, (0,2,1))
          crz = numpy.reshape(crz, (rz[i-1]*ny[i-1], rz[i]))
          # with RHS
          crC = numpy.reshape(coeff[i-1], (rc[i-1],ny[i-1]*rc[i]))
          ZC[i] = numpy.matmul(ZC[i-1], crC)
          ZC[i] = numpy.reshape(ZC[i], (rz[i-1]*ny[i-1], rc[i]))
          ZC[i] = numpy.matmul(numpy.conjugate(numpy.transpose(crz)), ZC[i])
          if nswp == 1:
            ZC[i-1] = None
            ZU[i-1] = None

      elif dir < 0: ##### right-left sweep
        cru = numpy.reshape(cru, (ru[i-1], ny[i-1]*ru[i]))
        # truncate cru with cross, now cru is not orthogonal
        rv, cru = localcross(cru, tol/numpy.sqrt(d))
        if kickrank > 0:
          # Update the residual and enrichment
          # enrichment first
          crC = numpy.reshape(coeff[i-1], (rc[i-1]*ny[i-1], rc[i]))
          crC = numpy.matmul(crC, UC[i])
          crC = numpy.reshape(crC, (rc[i-1], ny[i-1]*ru[i]))
          Uprev = numpy.matmul(rv, cru)
          Uprev = numpy.reshape(Uprev, (ru[i-1],ny[i-1]*ru[i]))
          crA = numpy.reshape(ZU[i-1], (rz[i-1]*ru[i-1], rc[i-1]))
          crz = numpy.empty((rz[i-1],ny[i-1]*ru[i]))
          for j in range(ny[i-1]*ru[i]):
            Ai = numpy.matmul(crA, crC[:,j])
            Ai = numpy.reshape(Ai, (rz[i-1], ru[i-1]))
            crz[:, j] = numpy.matmul(Ai, Uprev[:,j])

          crz -= numpy.matmul(ZC[i-1], crC)
          cru = numpy.vstack((cru, crz))
          # now the residual itself
          crC = numpy.reshape(coeff[i-1], (rc[i-1]*ny[i-1], rc[i]))
          crC = numpy.matmul(crC, ZC[i]) # now these are indices from the right, and Galerkin from the left
          crC = numpy.reshape(crC, (rc[i-1], ny[i-1]*rz[i]))
          Uprev = numpy.reshape(Uprev, (ru[i-1]*ny[i-1], ru[i]))
          Uprev = numpy.matmul(Uprev, ZU[i])
          Uprev = numpy.reshape(Uprev, (ru[i-1], ny[i-1]*rz[i]))
          crz = numpy.empty((rz[i-1],ny[i-1]*rz[i]))
          for j in range(ny[i-1]*rz[i]):
            Ai = numpy.matmul(crA, crC[:,j])
            Ai = numpy.reshape(Ai, (rz[i-1], ru[i-1]))
            crz[:,j] = numpy.matmul(Ai, Uprev[:,j])

          crz -= numpy.matmul(ZC[i-1], crC)
          crz = numpy.reshape(crz, (rz[i-1]*ny[i-1],rz[i]))

        ru[i-1] = numpy.shape(rv)[1]
        # QR u
        cru, v = numpy.linalg.qr(numpy.transpose(cru))
        v = v[:, :ru[i-1]]
        rv = numpy.matmul(rv, numpy.transpose(v))
        # maxvol to determine new indices
        ind = tt.maxvol.maxvol(cru)
        cru = numpy.transpose(cru)
        UU = cru[:, ind]
        cru = numpy.linalg.solve(UU, cru)
        rv = numpy.matmul(rv, UU)
        ru[i-1] = numpy.shape(rv)[1]
        # Cast non-orth factor to the next block
        if i > 1:
          u[i-2] = numpy.reshape(u[i-2], (ru[i-2]*ny[i-2],-1))
          u[i-2] = numpy.matmul(u[i-2], rv)
          u[i-2] = numpy.reshape(u[i-2], (ru[i-2], ny[i-2], -1))
        else:
          U0 = numpy.matmul(U0, rv)

        ru[i-1] = numpy.shape(cru)[0]
        u[i-1] = numpy.reshape(cru, (ru[i-1], ny[i-1], ru[i]))

        if use_indices: # TODO verify correctness
          Ju = numpy.hstack(
            (numpy.repeat(numpy.arange(ny[i-1]), ru[i]).reshape(-1,1),
            numpy.tile(Ju, [ny[i-1],1]))
          )
          Ju = Ju[ind, :]

        # Projection from the right -- sample C on U indices
        UC[i-1] = numpy.reshape(coeff[i-1], (rc[i-1]*ny[i-1], rc[i]))
        UC[i-1] = numpy.matmul(UC[i-1], UC[i])
        UC[i-1] = numpy.reshape(UC[i-1], (rc[i-1], ny[i-1]*ru[i]))
        UC[i-1] = UC[i-1][:, ind]

        # reductions with Z
        if kickrank > 0:
          # QR and maxvol Z
          crz = numpy.reshape(crz, (rz[i-1], ny[i-1]*rz[i]))
          crz = numpy.linalg.qr(numpy.transpose(crz))[0]
          rz[i-1] = numpy.shape(crz)[1]
          ind = tt.maxvol.maxvol(crz)
          # Sample C and U
          ZC[i-1] = numpy.reshape(coeff[i-1], (rc[i-1]*ny[i-1], rc[i]))
          ZC[i-1] = numpy.matmul(ZC[i-1], ZC[i])
          ZC[i-1] = numpy.reshape(ZC[i-1], (rc[i-1], ny[i-1]*rz[i]))
          ZC[i-1] = ZC[i-1][:, ind]
          ZU[i-1] = numpy.reshape(u[i-1], (ru[i-1]*ny[i-1], ru[i]))
          ZU[i-1] = numpy.matmul(ZU[i-1], ZU[i])
          ZU[i-1] = numpy.reshape(ZU[i-1], (ru[i-1], ny[i-1]*rz[i]))
          ZU[i-1] = ZU[i-1][:, ind]

      print(
        f'=als_cross_parametric= swp={swp} ({dir}), i={i}, dx={dx:.3e}, rank =[{ru[i-1]}, {ru[i]}]'
      )

    i += dir

    if dir > 0 and i == d+1 and swp == nswp:
      break # Last block & last sweep
    elif dir > 0 and i == d and swp < nswp:
      # turn at the right end
      print(f'=als_cross_parametric= fwd swp={swp}, max_dx={max_dx:.3e}, max_rank ={max(ru)}')
      dir = -1
      swp += 1
      max_dx = 0
      if use_indices:
        Ju = numpy.empty((rc[-1],0), dtype=numpy.int32)
    elif i == 0 and dir < 0:
      # turn at the left end
      dir = 1
      swp += 1
  
  U0 = numpy.reshape(U0, (1, Nxu, ru[0]))
  u = [U0] + u
  u = tt.tensor.from_list(u)     

  return (u,[time_solve, time_project],funevals)
