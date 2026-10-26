import warnings

import numpy as np
import pytest
import torch
from tt.maxvol import rect_maxvol as rect_maxvol_tt

from util.maxvol_torch import maxvol, rect_maxvol


def orthonormal(n, r, seed, dtype=torch.float64):
  q = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, r)))[0]
  return torch.from_numpy(q).to(dtype)

# (n, r, kickrank) with rf = 2 as in the cross algorithms
CASES = [(n, r, kick) for n in [10, 200, 3000] for r in [1, 5, 9, 40] for kick in [0, 2, 5] if r < n]


@pytest.mark.parametrize('n, r, kick', CASES)
def test_matches_tt(n, r, kick):
  a = orthonormal(n, r, seed=n + r + kick)
  maxK = r + kick + 2
  piv_tt, C_tt = rect_maxvol_tt(a, tol=1.1, maxK=maxK, min_add_K=kick, warn_budget=False)
  piv, C = rect_maxvol(a, tol=1.1, maxK=maxK, min_add_K=kick, warn_budget=False)
  assert np.array_equal(piv, np.asarray(piv_tt))
  assert torch.allclose(C, torch.as_tensor(C_tt), atol=1e-10)

@pytest.mark.parametrize('n, r, kick', CASES)
def test_properties(n, r, kick):
  a = orthonormal(n, r, seed=n * r + kick)
  maxK = r + kick + 2
  piv, C = rect_maxvol(a, tol=1.1, maxK=maxK, min_add_K=kick, warn_budget=False)
  assert isinstance(piv, np.ndarray) and np.issubdtype(piv.dtype, np.integer)
  assert len(np.unique(piv)) == len(piv)
  assert min(r + kick, n) <= len(piv) <= min(maxK, n)
  assert torch.equal(C[piv], torch.eye(len(piv), dtype=C.dtype))
  assert torch.allclose(C @ a[piv], a, atol=1e-12)

def test_row_norms_bounded():
  # without a budget the greedy runs until all rows are within tol
  a = orthonormal(500, 10, seed=0)
  piv, C = rect_maxvol(a, tol=1.1)
  assert float(C.norm(dim=1).max()) <= 1.1 + 1e-12

def test_square_maxvol_dominant():
  a = orthonormal(300, 12, seed=1)
  piv, C = maxvol(a, tol=1.05)
  assert len(piv) == 12
  assert float(C.abs().max()) <= 1.05 + 1e-12
  assert torch.allclose(C @ a[piv], a, atol=1e-12)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_keeps_dtype(dtype):
  a = orthonormal(100, 5, seed=2, dtype=dtype)
  piv, C = rect_maxvol(a, maxK=9, min_add_K=2, warn_budget=False)
  assert C.dtype == dtype
  assert torch.allclose(C @ a[piv], a, atol=1e-5 if dtype == torch.float32 else 1e-12)

def test_few_rows():
  # N <= r selects all rows
  a = orthonormal(4, 4, seed=3)
  piv, C = rect_maxvol(a)
  assert np.array_equal(piv, np.arange(4))
  assert torch.equal(C, torch.eye(4, dtype=a.dtype))

def test_rank_deficient():
  a = orthonormal(50, 4, seed=4)
  a[:, -1] = a[:, 0]
  with pytest.raises(torch.linalg.LinAlgError):
    rect_maxvol(a)

def test_non_finite():
  a = orthonormal(50, 4, seed=5)
  a[3, 1] = float('nan')
  with pytest.raises(ValueError):
    rect_maxvol(a)

@pytest.mark.parametrize('fn', [maxvol, rect_maxvol])
def test_no_autograd(fn):
  a = orthonormal(60, 10, seed=7).requires_grad_(True)
  with warnings.catch_warnings():
    warnings.simplefilter('error')
    piv, C = fn(a)
  assert not C.requires_grad and C.grad_fn is None
  assert torch.allclose(C @ a.detach()[piv], a.detach(), atol=1e-12)

@pytest.mark.parametrize('fn', [maxvol, rect_maxvol])
def test_device_none_matches_cpu(fn):
  a = orthonormal(200, 10, seed=8)
  piv_cpu, C_cpu = fn(a, device='cpu')
  piv_none, C_none = fn(a, device=None)
  assert np.array_equal(piv_cpu, piv_none)
  assert torch.equal(C_cpu, C_none)
  assert C_cpu.device == C_none.device == a.device

@pytest.mark.parametrize('fn', [maxvol, rect_maxvol])
def test_few_rows_device(fn):
  a = orthonormal(4, 4, seed=9)
  piv, C = fn(a, device=None)
  assert C.device == a.device and torch.equal(C, torch.eye(4, dtype=a.dtype))

@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
@pytest.mark.parametrize('fn', [maxvol, rect_maxvol])
def test_device_roundtrip(fn):
  a = orthonormal(200, 10, seed=10)
  piv_ref, C_ref = fn(a)
  # cuda input, computed on the cpu (default), returned on cuda
  piv, C = fn(a.cuda())
  assert C.device.type == 'cuda'
  assert np.array_equal(piv, piv_ref) and torch.allclose(C.cpu(), C_ref)
  # cpu input, computed on cuda, returned on the cpu
  piv, C = fn(a, device='cuda')
  assert C.device.type == 'cpu'
  assert np.array_equal(piv, piv_ref) and torch.allclose(C, C_ref, atol=1e-12)

def test_budget_warning():
  a = orthonormal(500, 10, seed=6)
  with pytest.warns(RuntimeWarning):
    rect_maxvol(a, tol=1.01, maxK=11)
