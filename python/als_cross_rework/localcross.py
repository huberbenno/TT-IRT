import numpy as np
import ctypes as ct

# try to load fast implementation
_have_llc= False
try:
  _llc = np.ctypeslib.load_library('liblocalcross', ".")
  _llc.localcross_f64.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
    ct.c_int, ct.c_int, ct.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
    ct.POINTER(ct.c_int)
  )
  _llc.localcross_c128.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2),
    ct.c_int, ct.c_int, ct.c_double,
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='F'),
    np.ctypeslib.ndpointer(dtype=np.complex128, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1),
    ct.POINTER(ct.c_int)
  )
  _have_llc = True
except:
  pass

def localcross(Y,tol, return_indices=False, fast=True):
  """
  Full-pivoted cross for truncating one ket TT block instead of SVD

  :param Y: TT block
  :param tol: truncation tolerance
  :param return_indices: (optional) return indices if True

  :return: 
    if ``return_indices=True`` returns tuple u,v,I; else returns tuple u,v
  """
  assert(Y.dtype == np.float64 or Y.dtype == np.complex128)

  if len(np.shape(Y)) == 2:
    n,m = np.shape(Y)
    b = 1
  else:
    n,m,b = np.shape(Y)

  minsize = min(n, m*b)
  u = np.zeros((n,minsize), order='F', dtype=Y.dtype)
  v = np.zeros((minsize, m*b), order='C', dtype=Y.dtype)
    
  I = np.zeros((minsize), dtype=np.float64) # also return indices

  if _have_llc and fast:
    Y_cp = Y.copy(order='F')
    r = ct.c_int(-1)
    if Y.dtype is np.float64:
      _llc.localcross_f64(Y_cp, n,m*b, tol, u, v, I,r)
    else:
      _llc.localcross_c128(Y_cp, n,m*b, tol, u, v, I,r)

    u = u[:, :r.value]
    v = v[:r.value, :]
    I = I[:r.value]

  else:
    # fallback
    res = np.reshape(Y, (n, m*b)).copy()
    val_max = np.max(np.abs(Y))
    r_tol = 1 # rank after truncation (0 tensor is also rank 1)
    for r in range(minsize):
      piv = np.argmax(np.abs(res))
      piv = np.unravel_index(piv, np.shape(res))
      val = np.abs(res[piv])
      if val <= tol*val_max:
        break

      r_tol = r+1
      u[:,r] = res[:,piv[1]]
      v[r,:] = res[piv[0],:] / res[piv]
      res -= np.outer(u[:,r], v[r, :])
      I[r] = piv[0]

    I = I[:r_tol]
    u = u[:,:r_tol]
    v = v[:r_tol,:]

    # QR u
    u, rv = np.linalg.qr(u)
    v = np.matmul(rv, v)

  if return_indices:
    return u,v,I
  else:
    return u,v
  
