#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/30 

from tiny_q import *
from pprint import pp
import numpy as np

# quantum phase estimation algorithm
# https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm
# https://zhuanlan.zhihu.com/p/84568388
# https://blog.csdn.net/qq_45777142/article/details/109904362


# test arbitary gate
u = U1(0, pi/3, pi/4, pi/5)
eigenvals, eigenvecs = np.linalg.eig(u.v)

for i in range(2):
  # Ax = λx
  assert np.allclose(u.v @ eigenvecs[:, i], eigenvals[i] * eigenvecs[:, i])

  print('eigenval:', eigenvals[i])
  print('eigenvec:', eigenvecs[:, i])
  phi = State(eigenvecs[:, i])
  q = phase_estimate(u, phi)
  q.info()
  r = q > Measure()
  print('estimated:', r)


# test known unitary
for u in [H, X, Y, Z, I]:
  eigenvals, eigenvecs = np.linalg.eig(u.v)

  for i in range(2):
    print('eigenval:', eigenvals[i])
    print('eigenvec:', eigenvecs[:, i])
    phi = State(eigenvecs[:, i])
    q = phase_estimate(u, phi)
    q.info()
    #r = q > Measure()
    #print('estimated:', r)


u = U1(pi/3, pi/4, pi/5, pi/6)
q = phase_estimate(u, n_prec=3)
pp(q > Measure())

phi = v0
q = phase_estimate(u, phi, n_prec=3)
pp(q > Measure())
