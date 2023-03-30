#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/22 

from tiny_q import *

# the two-qubit Deutsch-Jozsa algorithm
# https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
# https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm

# Oracle definition:
# Uf|xy> = |x,f(x)+y>

# (I@I)|xy> = |x,y>
#   f(x)+y = y
#   f(x) = 0
Uf_0 = I @ I

# (I@X)|xy> = |x,~y>
#   f(x)+y = ~y
#   f(x) = 1-2y       # -2y = 0 (mod 2)
#   f(x) = 1
Uf_1 = I @ X

# CNOT|xy> = |x,x+y>
#   f(x)+y = x+y
#   f(x) = x
Uf_I = CNOT

# CNOT|xy> = |x,x+y>
# (I@X)*CNOT|xy> = |x,(x+y)'>   (denotion: q' = 1-q = ~q)
#   f(x)+y = (x+y)'
#   f(x) = 1-x-2y     # -2y = 0 (mod 2)
#   f(x) = x'
Uf_X = (I @ X) * CNOT
# CNOT|x'y> = |x',x'+y>
# CNOT*(X@I)|xy> = |x',x'+y>
# (X@I)*CNOT*(X@I)|xy> = |x,x'+y>
#   f(x) = x'
Uf_X1 = (X @ I) * CNOT * (X @ I)      # another impl


# Circuit:
#   x: |0>--H--|    |--H--Measure
#              | Uf |
#   y: |1>--H--|    |
# 
for Uf in [Uf_0, Uf_1, Uf_I, Uf_X, Uf_X1]:
  DJ = (H @ I) * Uf * (H @ 2) | v('01')
  results = DJ > Measure()
  print(results)
