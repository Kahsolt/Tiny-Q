#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/15 

from tiny_q import *

## use matmul operator '@' for system expansion (tensor product)

v0 = State.zero()
v0.info('|0>')
v1 = State.one()
v1.info('|1>')

v00 = v0 @ v0 ; print(v00)
v01 = v0 @ v1 ; print(v01)
v10 = v1 @ v0 ; print(v10)
v11 = v1 @ v1 ; print(v11)
print()

v000   = State.zero(3)
v11    = State.one(2)
v0011  = v('0011')
v011   = v('011')
v00011 = v000 @ v11
v0110  = v01  @ v10
assert v00011 == v000 @ v11 == v00 @ v011 == v0 @ v0011
assert v0011 @ v0 == v0 @ v0110

u = X @ Y
u.info('X @ Y')
u = X @ CNOT
u.info('X @ CNOT')
u = H * Z * H
u.info('HZH')


## use pipe operator '|' for gate application
h0 = H | v0
h0.info('|+>')
h1 = H | v1
h1.info('|->')

q = X @ Y | h0 @ h1
q.info('XY|+->')

# single-qubit gate auto broadcast
q = H | State.one(2)
q.info('H|00>')

# local phase cannot be ignored
assert H | v0 != H | v1
assert h0 != S | h0
# but global phase it omittable
h0_gp = Ph(np.pi/3) | h0
h0_gp.info('e^-φi|+> (global phase)')
assert h0 == h0_gp


## use mul operator '*' for gate composition
u = X * X
assert u == I   # special case: I can be auto broadcast
u = X * Y
assert u != Z

ops: Gate = RY(np.pi/3) * Z * H
q: State = ops | v0
q.info('RY(pi/3)ZH|0>')

ops: Gate = RY(-np.pi/3) * S
q: State = ops | v1
q.info('RY(-pi/3)S|1>')


## use > Measure for single measurement
print('Measure:', q > Measure)
print('Measure:', q > Measure)
print('Measure:', q > Measure)
print('Measure:', q > Measure)
print('Measure:', q > Measure)
print('Measure:', q > Measure)
print('Measure:', q > Measure)

## use > Measure() for batch measurement
print('Measure(1000):', q > Measure(1000))
# note that '> Measure' behaves different from '> Measure(1)'
print('Measure(1):', q > Measure(1))

## use > State to project by state
print('q > h1:', q > h1)

## use > MeasureOp to project by measure operator
print('q > M0:', q > M0)
print('q > M1:', q > M1)
