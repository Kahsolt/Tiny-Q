#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/30 

from tiny_q import *
from pprint import pp

# quantum phase estimation algorithm
# https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm
# https://zhuanlan.zhihu.com/p/84568388
# https://blog.csdn.net/qq_45777142/article/details/109904362


u = U1(pi/3, pi/4, pi/5, pi/6)
q = phase_estimate(u, n_prec=3)
pp(q > Measure())


phi = v0
q = phase_estimate(u, phi, n_prec=3)
pp(q > Measure())
