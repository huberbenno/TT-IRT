import numpy
import ctypes
import os

# try to load fast implementation
_have_lutils = False
try:
  _lutils = numpy.ctypeslib.load_library('libutils', "./utils")
  _lutils.solve_blockdiag.argtypes = (
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
  )
  _lutils.solve_blockdiag_parallel.argtypes = (
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
  )

  _lutils.project_blockdiag.argtypes = (
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
  )
  _lutils.project_blockdiag_parallel.argtypes = (
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, flags='C'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.c_int
  )
  _have_lutils = True
except:
  pass


def solve_blockdiag(UAU, crC, crF, ru1, ru2, rc1, n1):
  
  # fast implementation
  if _have_lutils:
    cru = numpy.empty((ru1*n1,ru2))
    # TODO find a better way to do this
    if ru2*n1 < os.cpu_count(): # no threading for small number of blocks 
      _lutils.solve_blockdiag(UAU, crC, crF, cru, ru1, ru2, rc1, n1)
    elif ru1 < 150:
      _lutils.solve_blockdiag_parallel(UAU, crC, crF, cru, ru1, ru2, rc1, n1, os.cpu_count())
    else: # for ru1>=150 LAPack is parallel (on my machine TM)
      _lutils.solve_blockdiag_parallel(UAU, crC, crF, cru, ru1, ru2, rc1, n1, 2)
    return cru

  # fallback
  crA = numpy.reshape(UAU, (ru1*ru1, rc1))
  crC = numpy.reshape(crC, (rc1, n1*ru2))
  cru = numpy.empty((ru1, n1 * ru2))
  for j in range(n1 * ru2):
    Ai = numpy.matmul(crA, crC[:,j])
    Ai = numpy.reshape(Ai, (ru1, ru1))
    cru[:,j] = numpy.linalg.solve(Ai, crF[:,j])

  return numpy.reshape(cru, (ru1*n1,ru2))


def project_blockdiag(UAU,crC,cru,ru1,ru2,rc1,rc2,n1):

  # fast implementation
  if _have_lutils:
    UAU_new = numpy.empty((ru2, ru2*rc2))
    if n1 < 100:
      _lutils.project_blockdiag(UAU,crC,cru,UAU_new,ru1,ru2,rc1,rc2,n1)
    else:
      _lutils.project_blockdiag_parallel(UAU,crC,cru,UAU_new,ru1,ru2,rc1,rc2,n1,2)
    return UAU_new

  # fallback
  UAU_new = numpy.zeros((ru2, ru2*rc2))
  UAU = numpy.reshape(UAU, (ru1, ru1*rc1))
  crC = numpy.transpose(crC, (0,2,1))
  cru = numpy.transpose(cru, [0,2,1])
  for j in range(n1):
    v = cru[:,:,j]
    crA = numpy.matmul(numpy.conjugate(numpy.transpose(v)), UAU)
    crA = numpy.reshape(crA, (ru2*ru1, rc1))
    crA = numpy.matmul(crA, crC[:,:,j])
    crA = numpy.reshape(crA, (ru2, ru1*rc2))
    crA = numpy.transpose(crA)
    crA = numpy.reshape(crA, (ru1, ru2*rc2))
    crA = numpy.matmul(numpy.conjugate(numpy.transpose(v)), crA)
    crA = numpy.reshape(crA, (ru2*rc2, ru2))
    crA = numpy.transpose(crA)
    UAU_new += crA

  return UAU_new