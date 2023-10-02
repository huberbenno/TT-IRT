import numpy
import ctypes

# try to load fast implementation
_have_llc= False
try:
  _llc = numpy.ctypeslib.load_library('liblocalcross', "./utils")
  _llc.localcross_f64.argtypes = (
    numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2),
    ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char
  )
  _have_llc = True
except:
  pass


def localcross(Y,tol, return_indices=False):
  """
  Full-pivoted cross for truncating one ket TT block instead of SVD

  :param Y: TT block
  :param tol: truncation tolerance
  :param return_indices: (optional) return indices if true

  :return: 
    if ``return_indices=True`` returns tuple u,v,I; else returns tuple u,v
  """
  if len(numpy.shape(Y)) == 2:
    n,m = numpy.shape(Y)
    b = 1
  else:
    n,m,b = numpy.shape(Y)


  if _have_llc:
    uptr = ctypes.POINTER(ctypes.c_double)()
    vptr = ctypes.POINTER(ctypes.c_double)()
    Iptr = ctypes.POINTER(ctypes.c_double)()
    r = ctypes.c_int(-1)
    if Y.flags.f_contiguous:
      _llc.localcross_f64(Y, n,m*b, tol, uptr, vptr, Iptr,r,ctypes.c_char(ord('F')))
    else:
      _llc.localcross_f64(Y, n,m*b, tol, uptr, vptr, Iptr,r,ctypes.c_char(ord('C')))

    u = numpy.ctypeslib.as_array(uptr, (n, r.value))
    v = numpy.ctypeslib.as_array(vptr, (r.value,m*b))

    if return_indices:
      I = numpy.ctypeslib.as_array(Iptr, (r.value,))
      return u,v,I
    else:
      return u,v
  
  # fallback
  minsize = min(n, m*b)
  u = numpy.zeros((n,minsize))
  v = numpy.zeros((minsize, m*b))
  res = numpy.reshape(Y, (n, m*b)).copy()
    
  I = numpy.zeros((minsize)) # also return indices

  val_max = numpy.max(numpy.abs(Y))
  r_tol = 1 # rank after truncation (0 tensor is also rank 1)
  for r in range(minsize):
    piv = numpy.argmax(numpy.abs(res))
    piv = numpy.unravel_index(piv, numpy.shape(res))
    val = numpy.abs(res[piv])
    if val <= tol*val_max:
      break

    r_tol = r+1
    u[:,r] = res[:,piv[1]]
    v[r,:] = res[piv[0],:] / res[piv]
    res -= numpy.outer(u[:,r], v[r, :])
    I[r] = piv[0]

  I = I[:r_tol]
  u = u[:,:r_tol]
  v = v[:r_tol,:]

  # QR u
  u, rv = numpy.linalg.qr(u)
  v = numpy.matmul(rv, v)

  if return_indices:
    return u,v,I
  else:
    return u,v
  
