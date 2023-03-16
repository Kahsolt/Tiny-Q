#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

## use matmul operator '@' for system expansion (tensor product)

v0 = State.zero()
print('|0>')
print('  state:', v0)
print('  amp:', v0.amp)
print('  prob:', v0.prob)
print('  density:', v0.density)
v1 = State.one()
print('|1>')
print('  state:', v1)
print('  amp:', v1.amp)
print('  prob:', v1.prob)
print('  density:', v1.density)

v00 = v0 @ v0 ; print(v00)
v01 = v0 @ v1 ; print(v01)
v10 = v1 @ v0 ; print(v10)
v11 = v1 @ v1 ; print(v11)

v0011 = v00 @ v11 ; print(v0011)
v011  = v01 @ v1  ; print(v011)

v000 = State.zero(3)
v11  = State.one(2)
v00011 = v000 @ v11
v0110  = v01  @ v10
assert v00011 == v000 @ v11 == v00 @ v011 == v0 @ v0011
assert v0011 @ v0 == v0 @ v0110

u = X @ Y
print(u)
u = X @ CNOT
print(u)
u = H * H
print(u)


## use pipe operator '|' for gate application
q = H | v0
print('|+>')
print('  state:', q)
print('  amp:', q.amp)
print('  prob:', q.prob)
print('  density:', q.density)
q = H | v1
print('|->')
print('  state:', q)
print('  amp:', q.amp)
print('  prob:', q.prob)
print('  density:', q.density)

q = X @ Y | v1 @ v0
print(q)

# gate auto broadcast
q = H | State.one(3)
print(q)


## use mul operator '*' for gate composition
u = X * X
assert u == I

ops: Gate = RY(np.pi/3) * Z * H
q: State = ops | v0
print(q)
q.plot_density('RY(np.pi/3) * Z * H | v0')

ops: Gate = RY(-np.pi/3) * S
q: State = ops | v1
print(q)
q.plot_density('RY(-np.pi/3) * S | v1')


## use > for single measurement
r = q > Measure
print(r)

## call .measure() for batch measurement
res = q.measure(n=100)
print(res)
