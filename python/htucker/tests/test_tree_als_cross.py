"""
Tests for tree_als_cross.tree_als_cross_torch.TreeALSCross on a small affine 1d
diffusion / convection / reaction problem, compared with a dense reference solution
over the full parameter grid.
"""
import copy
import itertools

import numpy as np
import pytest
import torch

from tree.tree import Tree
from tree.tree_util import zero_root_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_als_cross.tree_als_cross_torch import TreeALSCross


# dim 0 (spatial) must be the first child of the root
TREES = {
  'pair': (0, 1),                                   # root_node with a single leaf child
  'balanced': zero_root_tupletree(4, 'balanced'),
  'linear': zero_root_tupletree(4, 'linear'),
  'wide_leaf': (0, 1, (2, 3)),                      # root arity 3, leaf below root_node
  'wide': (0, (1, 2), (3, 4)),                      # root arity 3, siblings below root_node
  'ternary': (0, ((1, 2, 3), 4)),                   # interior node with 3 children
}
N = 8   # spatial degrees of freedom
NY = 3  # parameter mode size
EXACT_TOL = 1e-8


class AffineProblem:
  """
  Problem with affine parameter dependence, implementing the assem_solve_fun interface:
  A(c) = sum_k sum_i c_k[i] B[k][i] and b(c) = sum_k sum_i c_k[i] f[k][i].
  """

  def __init__(self, B, f):
    self.B = B # list of (n_k, N, N)
    self.f = f # list of (m_k, N)

  def matrix(self, cores):
    return [list(torch.einsum('ij,ixy->jxy', c, B)) for c, B in zip(cores, self.B)]

  def rhs(self, cores):
    return [[v.reshape(-1, 1) for v in torch.einsum('ij,ix->jx', c, f)] for c, f in zip(cores, self.f)]

  def assemble(self, coeff_A, coeff_b):
    A = sum(torch.einsum('i,ixy->xy', c, B.to(c.dtype)) for c, B in zip(coeff_A, self.B))
    b = sum(torch.einsum('i,ix->x', c, f.to(c.dtype)) for c, f in zip(coeff_b, self.f))
    return A, b

  def solve(self, coeff_A, coeff_b):
    U = []
    for i in range(coeff_A[0].shape[1]):
      A, b = self.assemble([c[:, i] for c in coeff_A], [c[:, i] for c in coeff_b])
      U.append(torch.linalg.solve(A, b).reshape(-1, 1))
    return U


def _stiffness(dtype):
  """ 1d P1 stiffness contribution of each of the N+1 cells, homogeneous Dirichlet BC. """
  h = 1. / (N + 1)
  K = torch.zeros((N + 1, N, N), dtype=dtype)
  for c in range(N + 1):
    for i in (c - 1, c):
      for j in (c - 1, c):
        if 0 <= i < N and 0 <= j < N:
          K[c, i, j] += (1. if i == j else -1.) / h
  return K

def _diffusion_coeff(tree, pshape, seed):
  """ 1 + small random rank 2 perturbation, positive. """
  return TreeBasedTensor.ones(tree, (N + 1,) + pshape) \
    + TreeBasedTensor.randn(tree, (N + 1,) + pshape, 2, seed=seed, scale=0.1)

def _leaf_coeff(tree, pshape, dim, values, dtype):
  """ Rank 1 tensor of shape (1,) + pshape, depending only on parameter dim. """
  x = TreeBasedTensor.ones(tree, (1,) + pshape, dtype=dtype)
  x.cores[tree.dim2leaf(dim)] = torch.as_tensor(values, dtype=dtype).reshape(-1, 1)
  return x


def diffusion_problem(tree, n_A=3, n_b=2, dtype=torch.float64, seed=0):
  """
  Diffusion / convection / reaction problem, with matrix components
    0: diffusion, one stiffness contribution per cell (N+1 coefficients),
    1: convection with central differences, velocity depending on y_1,
    2: reaction (constant coefficient),
  and RHS components
    0: constant load,
    1: random nodal load (N coefficients).
  Returns the problem and the parameter tensors.
  """
  h = 1. / (N + 1)
  off = torch.ones(N - 1, dtype=dtype)
  conv = (torch.diag(off, 1) - torch.diag(off, -1)) / 2
  B = [_stiffness(dtype), conv[None], h * torch.eye(N, dtype=dtype)[None]]
  f = [h * torch.ones((1, N), dtype=dtype), h * torch.eye(N, dtype=dtype)]

  pshape = (NY,) * (tree.order - 1)
  a = TreeBasedTensor(_diffusion_coeff(tree, pshape, seed), dtype=dtype)
  v = _leaf_coeff(tree, pshape, 1, torch.linspace(-2., 2., NY), dtype)
  c = TreeBasedTensor.ones(tree, (1,) + pshape, dtype=dtype)
  f0 = TreeBasedTensor.ones(tree, (1,) + pshape, dtype=dtype)
  f1 = TreeBasedTensor.randn(tree, (N,) + pshape, 2, seed=seed + 1, dtype=dtype)
  return AffineProblem(B[:n_A], f[:n_b]), [a, v, c][:n_A], [f0, f1][:n_b]


def helmholtz_problem(tree, dtype=torch.complex128, seed=0, ny=NY):
  """
  Damped Helmholtz problem -(a u')' - k^2 (1 + 0.1i) u with an impedance term -ik u
  at the last node, i.e. complex symmetric but not hermitian, and indefinite. The
  matrix components are
    0: diffusion with real coefficient a(x, y) (N+1 coefficients),
    1: mass with coefficient -k^2 (1 + 0.1i), k depending on y_1,
    2: impedance at the last node with coefficient -ik', k' depending on y_d,
  and the RHS components are
    0: constant load with complex phase depending on y_1,
    1: random complex nodal load (N coefficients).
  Returns the problem and the parameter tensors.
  """
  h = 1. / (N + 1)
  imp = torch.zeros((1, N, N), dtype=dtype)
  imp[0, -1, -1] = 1.
  B = [_stiffness(dtype), h * torch.eye(N, dtype=dtype)[None], imp]
  f = [h * torch.ones((1, N), dtype=dtype), h * torch.eye(N, dtype=dtype)]

  # k^2 in [9, 36] lies above the smallest eigenvalue ~pi^2, so A is indefinite
  k = torch.linspace(3., 6., ny, dtype=torch.float64)
  pshape = (ny,) * (tree.order - 1)
  a = TreeBasedTensor(_diffusion_coeff(tree, pshape, seed), dtype=dtype)
  m = _leaf_coeff(tree, pshape, 1, -k**2 * (1 + 0.1j), dtype)
  z = _leaf_coeff(tree, pshape, tree.order - 1, -1j * k, dtype)
  f0 = _leaf_coeff(tree, pshape, 1, torch.exp(1j * torch.linspace(0., 2., ny, dtype=torch.float64)), dtype)
  f1 = TreeBasedTensor.randn(tree, (N,) + pshape, 2, seed=seed + 1, dtype=dtype)
  return AffineProblem(B, f), [a, m, z], [f0, f1]


PROBLEMS = {'diffusion': diffusion_problem, 'helmholtz': helmholtz_problem}


def reference(problem, A_params, b_params):
  """ Dense solution for all parameters in double precision, shape (N, NY, ..., NY). """
  ref_dtype = torch.complex128 if A_params[0].dtype.is_complex else torch.float64
  A_dense = [x.to_torch().to(ref_dtype) for x in A_params]
  b_dense = [x.to_torch().to(ref_dtype) for x in b_params]
  pshape = A_dense[0].shape[1:]
  U = torch.zeros((N,) + pshape, dtype=ref_dtype)
  for y in itertools.product(*[range(s) for s in pshape]):
    ind = (slice(None),) + y
    A, b = problem.assemble([x[ind] for x in A_dense], [x[ind] for x in b_dense])
    U[ind] = torch.linalg.solve(A, b)
  return U


def setup(tree_type='balanced', problem='diffusion', problem_kwargs={}, **kwargs):
  tree = Tree.from_tupletree(TREES[tree_type])
  problem, A_params, b_params = PROBLEMS[problem](tree, **problem_kwargs)
  als = TreeALSCross(A_params, b_params, problem, **kwargs)
  return als, problem, A_params, b_params

def rel_err(als, problem, A_params, b_params):
  U = reference(problem, A_params, b_params)
  u = als.get_tensor().to_torch().to(U.dtype)
  return (torch.linalg.norm(u - U) / torch.linalg.norm(U)).item()


def test_reference_problems():
  # sanity check of the test problems themselves
  tree = Tree.from_tupletree(TREES['balanced'])
  _, A_params, _ = diffusion_problem(tree)
  assert A_params[0].to_torch().min() > 0, 'diffusion coefficient must be positive'

  problem, A_params, b_params = helmholtz_problem(tree)
  y = (slice(None),) + (NY - 1,) * 4 # k = 6
  A, _ = problem.assemble([x.to_torch()[y] for x in A_params], [x.to_torch()[y] for x in b_params])
  assert torch.allclose(A, A.T) and not torch.allclose(A, A.mH), 'complex symmetric, not hermitian'
  eig = torch.linalg.eigvalsh(A.real)
  assert eig.min() < 0 < eig.max(), 'indefinite'


@pytest.mark.parametrize('rinit', [0, 3])
@pytest.mark.parametrize('kickrank', [0, 2])
@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('problem', PROBLEMS.keys())
def test_runs(problem, tree_type, kickrank, rinit):
  als, problem, A_params, b_params = setup(tree_type, problem, kickrank=kickrank, rinit=rinit)
  als.run(n_iter=2, tol=1e-8)

  u = als.get_tensor()
  assert u.shape == (N,) + A_params[0].shape[1:]
  assert u.tree.root.n_children == Tree.from_tupletree(TREES[tree_type]).root.n_children
  assert np.isfinite(u.numpy()).all()
  # the approximation should be in the right ballpark even without rank adaption
  assert rel_err(als, problem, A_params, b_params) < 0.5


@pytest.mark.parametrize('rinit', [0, 3])
@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('problem', PROBLEMS.keys())
def test_exact_with_kickrank(problem, tree_type, rinit):
  # the parameter grid is small, so rank adaption reaches the exact solution
  als, *data = setup(tree_type, problem, kickrank=2, rinit=rinit)
  als.run(n_iter=8, tol=1e-10)
  assert rel_err(als, *data) < EXACT_TOL


@pytest.mark.parametrize('n_A, n_b', [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize('kickrank', [0, 2])
def test_components(n_A, n_b, kickrank):
  als, *data = setup('wide', problem_kwargs=dict(n_A=n_A, n_b=n_b), kickrank=kickrank)
  assert (als.M_A, als.M_b) == (n_A, n_b)
  als.run(n_iter=8, tol=1e-10)
  err = rel_err(als, *data)
  assert err < (EXACT_TOL if kickrank > 0 else 0.5)


@pytest.mark.parametrize('tree_type', ['pair', 'balanced', 'wide_leaf', 'wide'])
@pytest.mark.parametrize('problem', PROBLEMS.keys())
def test_exact_full_rank_without_kickrank(problem, tree_type):
  # rinit exceeds all possible ranks (for these trees), so no rank adaption is needed.
  # The result is exact for any consistent projection, but test (conjugated) and
  # trial spaces mixed up in the projections of A and b break this for complex problems.
  als, *data = setup(tree_type, problem, kickrank=0, rinit=9)
  als.run(n_iter=5, tol=1e-12)
  assert rel_err(als, *data) < EXACT_TOL


@pytest.mark.parametrize('kickrank', [1, 5])
def test_kickrank_sizes(kickrank):
  als, *data = setup('balanced', kickrank=kickrank)
  als.run(n_iter=8, tol=1e-10)
  assert rel_err(als, *data) < EXACT_TOL


def test_truncation_tol():
  # coarse tol truncates ranks, fine tol reaches the exact solution
  als_coarse, *data = setup('balanced', kickrank=2)
  als_coarse.run(n_iter=6, tol=1e-1)
  als_fine, *_ = setup('balanced', kickrank=2)
  als_fine.run(n_iter=6, tol=1e-10)

  rank_coarse = sum(c.numel() for c in als_coarse.get_tensor().cores)
  rank_fine = sum(c.numel() for c in als_fine.get_tensor().cores)
  assert rank_coarse <= rank_fine
  assert rel_err(als_fine, *data) < EXACT_TOL
  assert rel_err(als_coarse, *data) < 0.5


def test_float32():
  als, *data = setup('balanced', problem_kwargs=dict(dtype=torch.float32), kickrank=2)
  als.run(n_iter=6, tol=1e-5)
  u = als.get_tensor()
  assert u.dtype == torch.float32
  assert all(c.dtype == torch.float32 for c in u.cores)
  assert rel_err(als, *data) < 1e-3


def test_repeated_run_continues(capsys):
  als, *data = setup('wide', kickrank=2, verbose=2)
  als.run(n_iter=2, tol=1e-10)
  als.run(n_iter=6, tol=1e-10)
  out = capsys.readouterr().out
  # sweep numbering continues across calls
  for i in range(8):
    assert f'= swp={i} finished' in out
  # verbose=2 prints per-node info
  assert 'leaf' in out and 'down' in out and 'up' in out
  assert als.n_eval > 0
  assert rel_err(als, *data) < EXACT_TOL


def test_dx_decreases(capsys):
  als, *_ = setup('balanced', kickrank=2, verbose=1)
  als.run(n_iter=8, tol=1e-10)
  out = capsys.readouterr().out
  max_dx = [float(line.split('max_dx=')[1]) for line in out.splitlines() if 'max_dx=' in line]
  assert len(max_dx) == 8
  assert max_dx[-1] < 1e-6 * max_dx[0]


def test_inputs_not_modified():
  tree = Tree.from_tupletree(TREES['wide'])
  problem, A_params, b_params = diffusion_problem(tree)
  A_copy, b_copy = copy.deepcopy(A_params), copy.deepcopy(b_params)
  als = TreeALSCross(A_params, b_params, problem, kickrank=2)
  als.run(n_iter=2)
  for x, y in zip(A_params + b_params, A_copy + b_copy):
    assert all(torch.equal(cx, cy) for cx, cy in zip(x.cores, y.cores))
    assert x.tree.root.n_children == y.tree.root.n_children


def test_get_tensor_matches_internal_state():
  # get_tensor returns a copy, changing it does not affect the solver
  als, *data = setup('balanced', kickrank=2)
  als.run(n_iter=6, tol=1e-10)
  u = als.get_tensor()
  u.cores[u.tree.root] *= 0
  assert rel_err(als, *data) < EXACT_TOL


@pytest.mark.parametrize('kickrank', [0, 2])
def test_complex_dtype(kickrank):
  als, *data = setup('wide', 'helmholtz', kickrank=kickrank)
  assert als.dtype == torch.complex128
  als.run(n_iter=4, tol=1e-10)
  u = als.get_tensor()
  assert u.dtype == torch.complex128
  assert all(c.dtype == torch.complex128 for c in u.cores)
  # the solution is genuinely complex
  U = u.to_torch()
  assert torch.linalg.norm(U.imag) > 0.1 * torch.linalg.norm(U.real)


def test_complex64():
  als, *data = setup('balanced', 'helmholtz', problem_kwargs=dict(dtype=torch.complex64), kickrank=2)
  als.run(n_iter=6, tol=1e-5)
  assert als.get_tensor().dtype == torch.complex64
  assert rel_err(als, *data) < 1e-3


def test_complex_real_part_problem_matches_real():
  # a complex problem with real data must give the same result as the real problem
  tree = Tree.from_tupletree(TREES['wide'])
  problem, A_params, b_params = diffusion_problem(tree)
  cdtype = torch.complex128
  problem_c = AffineProblem([B.to(cdtype) for B in problem.B], [f.to(cdtype) for f in problem.f])
  A_params_c = [TreeBasedTensor(x, dtype=cdtype) for x in A_params]
  b_params_c = [TreeBasedTensor(x, dtype=cdtype) for x in b_params]
  als = TreeALSCross(A_params, b_params, problem, rinit=9)
  als_c = TreeALSCross(A_params_c, b_params_c, problem_c, rinit=9)
  als.run(n_iter=4, tol=1e-12)
  als_c.run(n_iter=4, tol=1e-12)
  u, u_c = als.get_tensor().to_torch(), als_c.get_tensor().to_torch()
  assert torch.allclose(u_c.real, u, atol=1e-10 * u.abs().max())
  assert torch.allclose(u_c.imag, torch.zeros_like(u), atol=1e-10 * u.abs().max())


@pytest.mark.parametrize('rinit', [0, 3])
def test_helmholtz_low_rank_leaf(rinit):
  # u depends on y_2 only through a complex phase of the RHS, so the leaf of dim 2 has
  # a rank below its mode size and a complex interpolation core (with full rank
  # leaves it is a real permutation matrix).
  # Note: without rank adaption this case occasionally loses a rank in later sweeps
  # (also for real data), so only kickrank > 0 is tested.
  kickrank = 2
  tree = Tree.from_tupletree((0, (1, 2)))
  pshape = (NY, 6)
  dtype = torch.complex128
  problem, _, _ = helmholtz_problem(Tree.from_tupletree(TREES['balanced']))
  problem = AffineProblem(problem.B, problem.f[:1])
  k = torch.linspace(3., 6., NY, dtype=torch.float64)
  A_params = [
    TreeBasedTensor.ones(tree, (N + 1,) + pshape, dtype=dtype),
    _leaf_coeff(tree, pshape, 1, -k**2 * (1 + 0.1j), dtype),
    _leaf_coeff(tree, pshape, 1, -1j * k, dtype),
  ]
  b_params = [_leaf_coeff(tree, pshape, 2, torch.exp(1j * torch.linspace(0., 2., 6, dtype=torch.float64)), dtype)]

  als = TreeALSCross(A_params, b_params, problem, kickrank=kickrank, rinit=rinit)
  als.run(n_iter=6, tol=1e-10)
  assert als.get_tensor().cores[tree.dim2leaf(2)].shape[-1] < pshape[1]
  assert rel_err(als, problem, A_params, b_params) < EXACT_TOL


def subtree_matrix(cores, node):
  """ Matrix (subtree mode sizes, rank) of the subtree below node, first child slowest. """
  if node.isleaf:
    return cores[node]
  nc = node.n_children
  args = []
  for i, child in enumerate(node.children):
    args += [subtree_matrix(cores, child), [nc + 1 + i, i]]
  args += [cores[node], list(range(nc + 1)), [nc + 1 + i for i in range(nc)] + [nc]]
  M = torch.einsum(*args)
  return M.reshape(-1, M.shape[-1])

def subtree_nodes(node):
  yield node
  for child in node.children:
    yield from subtree_nodes(child)

@pytest.mark.parametrize('kickrank, rinit, n_iter', [(0, 0, 2), (0, 2, 2), (1, 2, 1)])
@pytest.mark.parametrize('tree_type', TREES.keys())
def test_projections_conjugate_test_space(tree_type, kickrank, rinit, n_iter):
  # After a sweep, UAU[k][node] and UF[k][node] of every node below the spatial core are
  # the Galerkin projections of the subtree below node, with the conjugated (test)
  # basis in the first slot:
  #   UAU[k][node][i, j, a] = sum_x conj(U[x, i]) U[x, j] A_k[x, a]
  #   UF[k][node][i, b] = sum_x conj(U[x, i]) b_k[x, b]
  # Mode size 5 with ranks <= 3 makes the interpolation cores complex, so that the
  # conjugation matters.
  als, *_ = setup(tree_type, 'helmholtz', problem_kwargs=dict(ny=5), kickrank=kickrank, rinit=rinit)
  als.run(n_iter=n_iter, tol=1e-10)

  n_complex_basis = 0
  for node in subtree_nodes(als.root_node):
    U = subtree_matrix(als.u.cores, node)
    n_complex_basis += U.shape[1] < U.shape[0] and not torch.allclose(U.imag, torch.zeros_like(U.imag))
    for k in range(als.M_A):
      A = subtree_matrix(als.A_params[k].cores, node)
      expected = torch.einsum('xi,xj,xa->ija', U.conj(), U, A)
      assert torch.allclose(als.UAU[k][node], expected, rtol=1e-8, atol=1e-10 * expected.abs().max())
    for k in range(als.M_b):
      b = subtree_matrix(als.b_params[k].cores, node)
      expected = U.conj().T @ b
      assert torch.allclose(als.UF[k][node], expected, rtol=1e-8, atol=1e-10 * expected.abs().max())

  # otherwise the test is not sensitive to the conjugation
  assert n_complex_basis >= 2
