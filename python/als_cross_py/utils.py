import numpy as np
import ctypes as ct
import os

# try to load fast implementation
_have_lutils = False
try:
  _lutils = np.ctypeslib.load_library('libutils', "./utils")
  _lutils.solve_blockdiag.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    ct.c_int, ct.c_int, ct.c_int, ct.c_int
  )
  _lutils.solve_blockdiag_parallel.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int
  )

  _lutils.project_blockdiag.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int
  )
  _lutils.project_blockdiag_parallel.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, 
    ct.c_int
  )
  _have_lutils = True
except:
  pass


def solve_blockdiag(UAU, crC, crF, ru1, ru2, rc1, n1, fast=False):

  crC = np.reshape(crC, (rc1, n1*ru2))
  # fast implementation
  if _have_lutils and fast:
    crA = np.reshape(UAU, (ru1, ru1, rc1))
    UAU = np.asfortranarray(crA)
    crC = np.asfortranarray(crC)
    cru = np.copy(crF, order='F')
    # TODO find a better way to do this
    if ru2*n1 < os.cpu_count(): # no threading for small number of blocks 
      _lutils.solve_blockdiag(UAU, crC, cru, ru1, ru2, rc1, n1)
    elif ru1 < 150:
      _lutils.solve_blockdiag_parallel(UAU, crC, cru, ru1, ru2, rc1, n1, os.cpu_count())
    else: # for ru1>=150 LAPack is parallel (on my machine TM)
      _lutils.solve_blockdiag_parallel(UAU, crC, cru, ru1, ru2, rc1, n1, 2)
  else:
    # fallback
    crA = np.reshape(UAU, (ru1*ru1, rc1))
    cru = np.empty((ru1, n1 * ru2))
    for j in range(n1 * ru2):
      Ai = np.matmul(crA, crC[:,j])
      Ai = np.reshape(Ai, (ru1, ru1))
      cru[:,j] = np.linalg.solve(Ai, crF[:,j])

  return cru.reshape((ru1*n1,ru2))


def project_blockdiag(UAU,crC,cru,ru1,ru2,rc1,rc2,n1, fast=True):

  UAU_new = np.zeros((ru2, ru2*rc2), order='C')

  # fast implementation
  if _have_lutils and fast:
    UAU = np.reshape(UAU, (ru1, ru1, rc1))
    UAU = np.asfortranarray(UAU)
    crC = np.asfortranarray(crC)
    cru = np.asfortranarray(cru)
    if n1 < 100:
      _lutils.project_blockdiag(UAU,crC,cru,UAU_new,ru1,ru2,rc1,rc2,n1)
    else:
      _lutils.project_blockdiag_parallel(UAU,crC,cru,UAU_new,ru1,ru2,rc1,rc2,n1,2)
  else:
    # fallback
    UAU = np.reshape(UAU, (ru1, ru1*rc1))
    crC = np.transpose(crC, (0,2,1))
    cru = np.transpose(cru, [0,2,1])
    for j in range(n1):
      v = cru[:,:,j]
      crA = np.matmul(np.conjugate(np.transpose(v)), UAU)
      crA = np.reshape(crA, (ru2*ru1, rc1))
      crA = np.matmul(crA, crC[:,:,j])
      crA = np.reshape(crA, (ru2, ru1*rc2))
      crA = np.transpose(crA)
      crA = np.reshape(crA, (ru1, ru2*rc2))
      crA = np.matmul(np.conjugate(np.transpose(v)), crA)
      crA = np.reshape(crA, (ru2*rc2, ru2))
      crA = np.transpose(crA)
      UAU_new += crA

  return UAU_new