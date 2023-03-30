#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

# test basic calculation correctness

# => basic states
assert (v0 > v1) < EPS and (v1 > v0) < EPS            # diagonal
assert (h0 > h1) < EPS and (h1 > h0) < EPS
for q1 in [v0, v1, h0, h1]:
  for q2 in [v0, v1, h0, h1]:
    if q1 == q2: assert (q1 > q2) - 1   < EPS
    else:        assert (q1 > q2) - 0.5 < EPS
assert v('0') == v0 and v('1') == v1
assert v('10110') == v1 @ v0 @ v1 @ v1 @ v0
assert v0.dagger == v0

# => global phase gate
assert X.dagger == X
assert Ph(0) == I
assert Ph(pi) == Ph(-pi) == Gate(-I.v)

# => pauli gates
assert X^2 == I and Y^2 == I and Z^2 == I
assert H*Z*H == X and H*X*H == Z
assert X*Y == -Y*X and Y*Z == -Z*Y and Z*X == -X*Z 
assert Gate(((X*Y).v-(Y*X).v)/2) == Gate(i*Z.v)       # σiσj - σjσi = 2*i*σk, where i,j,k is a cyclic permutation of of X,Y,Z

# => phase gates
assert SX^2 == X
assert T^2 == S and S^2 == Z and Z^2 == I
assert Z == P(pi) and S == P(pi/2) and T == P(pi/4)

# => measure operator set
assert MeasureOp.check_completeness([M0, M1])
