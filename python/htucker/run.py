from benchmarks.alsc_convergence import ARGS_alsc_convergence, alsc_convergence

files = []
for i in range(20):
  args = ARGS_alsc_convergence(
    tree_type = 'random',
    rng_seed = 40+i,
    Ny_decay = True,
    alsc_rinit = 5,
    alsc_kickrank = 5,
    alsc_niter= 30,
    cross_eps = 1e-5,
    alsc_eps = 1e-7,
    verbose = 0
  )
  res = alsc_convergence(args)
  files += [res['file']]

for f in files:
  print(f"'{f}',")