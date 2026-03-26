import tt
import tt.cross
import numpy as np


def target_f(inds):
  return (1 + np.sum(inds, axis=-1))

TT_c = tt.rand(np.array(5 * [8]),r=1)
# compute TT approx using TT-cross
TT_c = tt.cross.rect_cross.cross(target_f, TT_c, nswp=10, eps = 1e-2, kickrank=1)

TT_c = TT_c.round(1e-12)

print(TT_c.r)

inds = np.meshgrid(*[np.arange(d, dtype=int) for d in TT_c.n], indexing='ij')
inds = np.stack(inds, axis=-1)

target_full = target_f(inds)

tt_full = TT_c.full()

print(f'err = {np.linalg.norm(target_full - tt_full)/ np.linalg.norm(target_full):.2e}')