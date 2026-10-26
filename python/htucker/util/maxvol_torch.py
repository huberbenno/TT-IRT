# Torch port of the square and rectangular maximum-volume submatrix routines
# `maxvol` and `rect_maxvol` from ttpy 2.0.0rc3 (tt/algs/maxvol.py),
# https://github.com/oseledets/ttpy, by Ivan Oseledets and the ttpy contributors.
#
# Changes with respect to the original: the numpy/scipy LAPACK (getrf) and BLAS
# (solve, ger) calls are replaced by torch operations, so the computation stays on
# the device of the input and in torch's thread pool. Mixing numpy's BLAS threads
# with torch's threads oversubscribes the CPU on the small matrices of the cross
# algorithms. The `info` dict, `top_k_index` and most of the diagnostics are not
# ported.
#
# Algorithms:
#   S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov,
#   N. L. Zamarashkin, "How to find a good submatrix", in Matrix Methods: Theory,
#   Algorithms, Applications, 2010, pp. 247-256.
#   A. Mikhalev, I. V. Oseledets, "Rectangular maximum-volume submatrices and their
#   applications", Linear Algebra and its Applications 538 (2018) 187-211,
#   arXiv:1502.07838.
#
# License of the original:
#
#   MIT License
#
#   Copyright (c) 2024-2026 Ivan Oseledets
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import warnings

import numpy as np
import torch


def _as_matrix(a, device=None):
  """
  Torch matrix from a on device (unchanged if None), floating point and finite.
  """
  a = torch.as_tensor(a)
  if device is not None:
    a = a.to(device)
  if a.ndim != 2:
    raise ValueError(f'maxvol expects a 2d matrix, got shape {tuple(a.shape)}')
  if not (a.is_floating_point() or a.is_complex()):
    a = a.to(torch.float64)
  # every stopping test is a '>' comparison, which would silently be false for a NaN
  if not torch.isfinite(a).all():
    raise ValueError('maxvol: the input matrix has non-finite entries')
  return a


def _maxvol(A, tol, max_iters, rcond=None):
  """
  Square maxvol of the floating point matrix A (N > r), see maxvol. Returns the
  pivots as a torch tensor, C and whether max |C| <= tol was reached.
  """
  n, r = A.shape

  # starting row set: LU with partial pivoting
  LU, pivots = torch.linalg.lu_factor(A)
  diag = LU.diagonal()[:r].abs()
  top, low = float(diag.max()), float(diag.min())
  if rcond is None:
    rcond = r * torch.finfo(A.dtype).eps
  if top == 0. or low <= rcond * top:
    raise torch.linalg.LinAlgError(
      f'maxvol: the matrix is rank deficient / numerically singular (smallest '
      f'|U_ii| = {low:.3e}, largest = {top:.3e}, threshold = {rcond:.3e} * largest)')
  # LAPACK pivots are 1-based row interchanges
  perm = list(range(n))
  for i, p in enumerate(pivots.tolist()[:r]):
    perm[i], perm[p - 1] = perm[p - 1], perm[i]
  piv = torch.tensor(perm[:r], device=A.device)

  # C = A A[piv]^-1
  C = torch.linalg.solve(A[piv].T, A.T).T.contiguous()

  # swap rows while some |C[i, j]| > tol: replacing piv[j] by i multiplies
  # |det A[piv]| by |C[i, j]|, C is updated by Sherman-Morrison
  for _ in range(max_iters):
    pos = int(C.abs().argmax())
    i, j = divmod(pos, r)
    pivot = C[i, j].item()
    if abs(pivot) <= tol:
      return piv, C, True
    col, row = C[:, j].clone(), C[i, :].clone()
    row[j] -= 1.
    piv[j] = i
    C.addr_(col, row, alpha=-1. / pivot)
  return piv, C, float(C.abs().max()) <= tol


# maxvol is not differentiable, and torch would otherwise record the in-place
# updates for an input that requires grad
@torch.no_grad()
def maxvol(a, tol=1.05, max_iters=100, rcond=None, warn_nonconvergence=True, device='cpu'):
  """
  Square maximum-volume submatrix: r rows of locally maximal |det|.

  Parameters
  ----------
  a: torch.tensor
    (N, r) matrix with N >= r and full column rank.
  tol: float, optional
    Upper bound for max |C|. Values below 1 mean 1 + tol (legacy spelling).
  max_iters: int, optional
    Maximum number of row swaps.
  rcond: float, optional
    Relative threshold on the LU diagonal for the rank test. Default r * eps.
  warn_nonconvergence: bool, optional
    Warn if max_iters is reached with max |C| > tol.
  device: torch.device or str, optional
    Device the computation runs on. Default is the cpu, since the matrices are
    small and every greedy step synchronizes with the host. None uses the device
    of a.

  Returns
  -------
  piv: np.ndarray
    (r,) selected rows.
  C: torch.tensor
    (N, r) matrix on the device of a with a = C a[piv] and C[piv] = I.
  """
  out_device = torch.as_tensor(a).device
  A = _as_matrix(a, device)
  n, r = A.shape
  if n <= r:
    return np.arange(n), torch.eye(n, dtype=A.dtype, device=out_device)
  tol = float(tol)
  if not tol > 0.:
    raise ValueError(f'maxvol tol must be positive, got {tol}')
  if tol < 1.:
    tol = 1. + tol

  piv, C, converged = _maxvol(A, tol, int(max_iters), rcond)
  if not converged and warn_nonconvergence:
    warnings.warn(
      f'maxvol did not converge after {max_iters} swaps. The factorisation a = C a[piv] '
      'still holds, the submatrix is just not dominant.', RuntimeWarning, stacklevel=2)
  return piv.cpu().numpy(), C.to(out_device)


@torch.no_grad()
def rect_maxvol(a, tol=1.05, maxK=None, min_add_K=None, minK=None,
                start_maxvol_iters=10, rcond=None, warn_budget=True, device='cpu'):
  """
  Rectangular maximum-volume submatrix: K >= r rows with small ||C[i]||_2.

  Starts from the square maxvol rows and greedily adds the row with the largest
  norm of C = a pinv(a[piv]) until all row norms are at most tol. Adding a row is a
  rank-1 update of C (Sherman-Woodbury-Morrison).

  Parameters
  ----------
  a: torch.tensor
    (N, r) matrix with N >= r and full column rank.
  tol: float, optional
    Upper bound for the 2-norm of the rows of C.
  maxK: int, optional
    Maximum number of selected rows. Default is N.
  min_add_K: int, optional
    Minimum number of rows added to the r square ones.
  minK: int, optional
    Minimum number of selected rows. max(minK, r + min_add_K) wins.
  start_maxvol_iters: int, optional
    Number of square maxvol swaps before the rectangular phase.
  rcond: float, optional
    Relative threshold for the rank test of the square phase.
  warn_budget: bool, optional
    Warn if the growth stops at maxK with rows still above tol.
  device: torch.device or str, optional
    Device the computation runs on. Default is the cpu, since the matrices are
    small and every greedy step synchronizes with the host. None uses the device
    of a.

  Returns
  -------
  piv: np.ndarray
    (K,) selected rows.
  C: torch.tensor
    (N, K) matrix on the device of a with a = C a[piv] and C[piv] = I.
  """
  out_device = torch.as_tensor(a).device
  A = _as_matrix(a, device)
  n, r = A.shape
  dtype, device = A.dtype, A.device
  if n <= r:
    return np.arange(n), torch.eye(n, dtype=dtype, device=out_device)
  tol = float(tol)
  if not tol > 0.:
    raise ValueError(f'rect_maxvol tol must be positive, got {tol}')
  tol2 = tol * tol

  # bounds on K
  if maxK is not None and minK is not None and minK > maxK:
    raise ValueError(f'rect_maxvol: minK={minK} > maxK={maxK}, the request is contradictory')
  maxK = n if maxK is None else int(maxK)
  minK = r if minK is None else int(minK)
  if min_add_K is not None:
    minK = max(minK, r + int(min_add_K))
  maxK = min(max(maxK, r), n)
  minK = min(max(minK, r), maxK)

  # truncated square phase as a warm start
  piv0, C0, _ = _maxvol(A, 1., start_maxvol_iters, rcond)

  piv = torch.zeros(maxK, dtype=torch.long, device=device)
  piv[:r] = piv0
  C = torch.zeros((n, maxK), dtype=dtype, device=device)
  C[:, :r] = C0

  row_norm_sqr = (C[:, :r].abs() ** 2).sum(dim=1)
  chosen = torch.ones(n, dtype=row_norm_sqr.dtype, device=device)
  chosen[piv0] = 0.
  row_norm_sqr *= chosen
  # never re-select a row, even when minK forces growth
  row_norm_sqr[piv0] = -1.

  K = r
  i = int(row_norm_sqr.argmax())
  while (row_norm_sqr[i].item() > tol2 and K < maxK) or K < minK:
    piv[K] = i
    chosen[i] = 0.
    c = C[i, :K].clone()
    v = C[:, :K] @ c.conj()
    # v[i] = ||C[i]||^2 is real by construction
    l = 1. / (1. + v[i].real.item())
    C[:, :K].addr_(v, c, alpha=-l)
    C[:, K] = l * v
    row_norm_sqr -= l * (v * v.conj()).real
    row_norm_sqr *= chosen
    row_norm_sqr[piv[:K + 1]] = -1.
    K += 1
    i = int(row_norm_sqr.argmax())

  piv = piv[:K]
  C = C[:, :K].clone()
  C[piv] = torch.eye(K, dtype=dtype, device=device)

  # rows above tol remain when maxK stopped the growth, or always for tol < 1, as
  # the selected rows have norm 1
  remaining = row_norm_sqr[i].item()
  if remaining > tol2 and warn_budget:
    warnings.warn(
      f'rect_maxvol stopped at maxK={maxK} with rows of norm {remaining**.5:.3g} > tol = {tol:.3g}',
      RuntimeWarning, stacklevel=2)
  elif remaining <= tol2 and tol < 1.:
    warnings.warn(
      f'rect_maxvol: tol = {tol:.3g} < 1 cannot hold for the selected rows, which have norm 1',
      RuntimeWarning, stacklevel=2)

  return piv.cpu().numpy(), C.to(out_device)
